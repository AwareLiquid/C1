#!/usr/bin/env python3
"""E2 — MTP (Multi-Token Prediction) + liquid state look-ahead for C1.
Mirrors DeepSeek V3's MTP: predict future k tokens from shared trunk.

Modes:
  baseline  : single-step CE (existing protocol)
  mtp       : token-level k=3 look-ahead with decaying weights
  state-mtp : + liquid-state look-ahead (regress future h states, MSE)

Run:
  python experiments/e2_mtp.py --mode baseline --d_model 832 --n_layers 12
  python experiments/e2_mtp.py --mode mtp --k 3
  python experiments/e2_mtp.py --mode state-mtp --k 3
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _m1_bootstrap import bootstrap_m1  # ????????(? _m1_bootstrap.py)
bootstrap_m1()

import numpy as np
import torch
import torch.nn as nn

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel


def load_data(data_dir):
    train = torch.from_numpy(
        np.fromfile(os.path.join(data_dir, "train.bin"), dtype="uint16")).long()
    val = torch.from_numpy(
        np.fromfile(os.path.join(data_dir, "validation.bin"), dtype="uint16")).long()
    return train, val


def make_chunks(t, n):
    return t[: len(t) // n * n].view(-1, n)


class StateLookahead(nn.Module):
    """C1-specific: regress future liquid states h_{t+k} from h_t."""
    def __init__(self, d_model, k):
        super().__init__()
        self.k = k
        self.heads = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(k)
        ])

    def forward(self, h):
        # h: (B, T, D) -> list of (B, T, D) predictions for t+1..t+k
        return [head(h) for head in self.heads]


class MTPWrapper(nn.Module):
    """Wraps MTLNNModel: shared trunk + k token heads (+ optional state heads)."""
    def __init__(self, cfg, k=3, use_state=False, vocab_size=50257):
        super().__init__()
        self.model = MTLNNModel(cfg)
        self.k = k
        self.use_state = use_state
        self.token_heads = nn.ModuleList([
            nn.Linear(cfg.d_model, vocab_size) for _ in range(k)
        ])
        if use_state:
            self.state_heads = StateLookahead(cfg.d_model, k)

    def forward(self, x, labels=None):
        # base forward -> hidden states
        out = self.model(x)  # out["logits"], out["hidden"] if exposed
        h = out.get("hidden_states", out["logits"])
        # h may be logits (B,T,V) -> treat as (B,T,D) if D==V
        if h.dim() == 3 and h.size(-1) == 50257:
            h = h  # logits used as proxy (fallback)
        B, T, D = h.shape
        losses = []
        total = 0.0
        for j in range(self.k):
            # predict x_{t+j} from h_t (shifted target)
            if T - j <= 0:
                break
            logits_j = self.token_heads[j](h[:, : T - j])  # (B, T-j, V)
            if labels is not None:
                target = labels[:, j:] if j > 0 else labels
                if target.shape[1] > T - j:
                    target = target[:, : T - j]
                w = 1.0 / (2 ** j)  # decaying weight: 1, 0.5, 0.25
                ce = nn.functional.cross_entropy(
                    logits_j.reshape(-1, logits_j.size(-1)),
                    target.reshape(-1))
                total += w * ce
            losses.append(logits_j)
        if self.use_state:
            # state look-ahead: predict h_{t+j} (MSE on normalized states)
            for j in range(self.k):
                if T - j <= 0:
                    break
                pred = self.state_heads.heads[j](h[:, : T - j])
                target = h[:, j:]
                mse = nn.functional.mse_loss(pred, target)
                total += 0.5 * mse
        if labels is not None:
            return {"loss": total, "lm_loss": total}
        return {"logits": losses[-1] if losses else h}


def train_eval(args, seed, mode):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val = load_data(args.data_dir)
    tr = make_chunks(train, args.seq_len)
    va = make_chunks(val, args.seq_len)
    print(f"  [seed {seed}] {mode} chunks={len(tr)}/{len(va)}", flush=True)

    cfg = MTLNNConfig(
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads, d_head=args.d_model // args.n_heads,
        max_seq_len=args.seq_len, vocab_size=args.vocab_size,
    )
    k = 1 if mode == "baseline" else args.k
    use_state = mode == "state-mtp"
    model = MTPWrapper(cfg, k=k, use_state=use_state,
                       vocab_size=args.vocab_size).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    step = 0
    for epoch in range(10):
        perm = torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed + 99))
        for i in range(0, len(perm), args.batch):
            idx = perm[i : i + args.batch]
            inp = tr[idx].to(device)
            opt.zero_grad()
            out = model(inp, labels=inp)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 500 == 0:
                print(f"    step {step} loss {loss.item():.3f}", flush=True)
            if step >= args.steps:
                break
        if step >= args.steps:
            break

    # eval PPL (CE only, first token head)
    model.eval()
    total_ce, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(va), args.batch):
            inp = va[i : i + args.batch].to(device)
            out = model.model(inp, labels=inp)
            ce = out.get("lm_loss", out["loss"])
            total_ce += ce.item()
            n += 1
    ppl = math.exp(min(total_ce / max(n, 1), 20.0))
    print(f"  [seed {seed}] {mode} val_ppl={ppl:.3f}", flush=True)
    return {"mode": mode, "seed": seed, "val_ppl": ppl, "steps": args.steps,
            "k": k, "use_state": use_state}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "mtp", "state-mtp"], default="mtp")
    ap.add_argument("--d_model", type=int, default=832)
    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--n_heads", type=int, default=13)
    ap.add_argument("--n_kv_heads", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--vocab_size", type=int, default=50257)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="results_24h/e2_mtp.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for seed in args.seeds:
        r = train_eval(args, seed, args.mode)
        with open(args.out, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
