#!/usr/bin/env python3
"""E5 — Multi-horizon unified output for C1 (EnergyTS-inspired).

One liquid trunk + one head per horizon h in {8, 32, 128}: predict token t+h
from the state at t. Tests whether one unified architecture serves short AND
long horizons (cf. EnergyTS "长短时路径统一架构").

Modes:
  unified   : single model trained on all horizons jointly
  dedicated : one model per horizon (same per-model budget) — control

Run:
  python experiments/e5_multi_horizon.py --smoke --horizons 8 16 --steps 3
  python experiments/e5_multi_horizon.py --mode unified --horizons 8 32 128
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from mt_lnn.config import MTLNNConfig  # noqa: E402
from mt_lnn.model import MTLNNModel  # noqa: E402


def load_data(data_dir):
    train = torch.from_numpy(
        np.fromfile(os.path.join(data_dir, "train.bin"), dtype="uint16")).long()
    val = torch.from_numpy(
        np.fromfile(os.path.join(data_dir, "validation.bin"), dtype="uint16")).long()
    return train, val


def synthetic_data(seq_len, n_chunks=64, vocab_size=200, seed=0):
    """Random uint16 corpus for --smoke: no data files needed."""
    g = torch.Generator().manual_seed(seed)
    n = n_chunks * seq_len
    t = torch.randint(1, vocab_size, (n,), generator=g, dtype=torch.long)
    return t, t  # train == val for the smoke (PPL is meaningless, pipeline only)


def make_chunks(t, n):
    return t[: len(t) // n * n].view(-1, n)


class MultiHorizonWrapper(nn.Module):
    """Shared trunk + one token head per horizon (sparse, e.g. 8/32/128)."""

    def __init__(self, cfg, horizons, vocab_size=50257, weight_mode="inv_sqrt"):
        super().__init__()
        self.model = MTLNNModel(cfg)
        self.horizons = list(horizons)
        self.weight_mode = weight_mode
        self.heads = nn.ModuleList([
            nn.Linear(cfg.d_model, vocab_size) for _ in self.horizons
        ])

    def _weights(self):
        """Horizon loss weights, mean-normalized to 1 (fair vs dedicated)."""
        raw = []
        for h in self.horizons:
            if self.weight_mode == "equal":
                raw.append(1.0)
            elif self.weight_mode == "inv":
                raw.append(1.0 / h)
            else:  # inv_sqrt (default)
                raw.append(1.0 / math.sqrt(h))
        mean = sum(raw) / len(raw)
        return [w / mean for w in raw]

    def forward(self, x, labels=None):
        # capture the final hidden (output of final_norm, input to lm_head)
        captured = {}

        def hook(module, inp, out):
            captured["h"] = out  # (B, T, D) — live graph, grads flow to heads

        handle = self.model.final_norm.register_forward_hook(hook)
        self.model(x)
        handle.remove()
        hidden = captured["h"]
        if labels is None:
            return {"hidden": hidden}

        T = hidden.shape[1]
        total = 0.0
        per_h = {}
        for h, head, w in zip(self.horizons, self.heads, self._weights()):
            if T - h <= 0:
                continue
            logits = head(hidden[:, : T - h])  # (B, T-h, V)
            target = labels[:, h:]
            if target.shape[1] > T - h:
                target = target[:, : T - h]
            ce = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            per_h[h] = ce.detach()
            total = total + w * ce
        return {"loss": total, "per_h": per_h}


def train_eval(args, seed, mode):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        train, val = synthetic_data(args.seq_len, 64, args.vocab_size, seed)
    else:
        train, val = load_data(args.data_dir)
    tr = make_chunks(train, args.seq_len)
    va = make_chunks(val, args.seq_len)
    print(f"  [seed {seed}] {mode} chunks={len(tr)}/{len(va)}", flush=True)

    cfg = MTLNNConfig(
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads, d_head=args.d_model // args.n_heads,
        max_seq_len=args.seq_len, vocab_size=args.vocab_size,
        gwtb_n_heads=1,  # tiny/smoke configs: d_gw=13 must divide heads
        n_time_scales=args.n_scales,
    )

    # unified: one model over all horizons; dedicated: one model per horizon
    runs = [(mode, list(args.horizons))] if mode == "unified" else \
           [("dedicated", [h]) for h in args.horizons]

    results = {}
    for run_mode, horizons in runs:
        torch.manual_seed(seed)  # same init seed / budget per run
        model = MultiHorizonWrapper(
            cfg, horizons, vocab_size=args.vocab_size,
            weight_mode=args.weight_mode).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

        step = 0
        for epoch in range(10):
            perm = torch.randperm(len(tr),
                                  generator=torch.Generator().manual_seed(seed + 99))
            for i in range(0, len(perm), args.batch):
                inp = tr[perm[i: i + args.batch]].to(device)
                opt.zero_grad()
                out = model(inp, labels=inp)
                loss = out["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                step += 1
                if step % 500 == 0:
                    print(f"    [{run_mode}:{horizons}] step {step} "
                          f"loss {loss.item():.3f}", flush=True)
                if step >= args.steps:
                    break
            if step >= args.steps:
                break

        # eval per-horizon PPL (CE only) on val
        model.eval()
        sums = {h: 0.0 for h in horizons}
        n = 0
        with torch.no_grad():
            for i in range(0, len(va), args.batch):
                inp = va[i: i + args.batch].to(device)
                out = model(inp, labels=inp)
                for h in horizons:
                    if h in out["per_h"]:
                        sums[h] += out["per_h"][h].item()
                n += 1
        for h in horizons:
            ppl = math.exp(min(sums[h] / max(n, 1), 20.0))
            results[h] = round(ppl, 3)
            print(f"  [seed {seed}] {run_mode} h={h} val_ppl={ppl:.3f}", flush=True)

    return {"mode": mode, "seed": seed, "n_scales": args.n_scales,
            "horizons": list(args.horizons), "ppl": results, "steps": args.steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["unified", "dedicated"], default="unified")
    ap.add_argument("--horizons", type=int, nargs="+", default=[8, 32, 128])
    ap.add_argument("--n_scales", type=int, default=5, choices=[5, 8, 16])
    ap.add_argument("--weight_mode", choices=["inv_sqrt", "inv", "equal"],
                    default="inv_sqrt")
    ap.add_argument("--d_model", type=int, default=832)
    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--n_heads", type=int, default=13)
    ap.add_argument("--n_kv_heads", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--vocab_size", type=int, default=50257)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--smoke", action="store_true",
                    help="use synthetic random data (pipeline check, no data files)")
    ap.add_argument("--out", default="results/e5_multi_horizon.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for seed in args.seeds:
        r = train_eval(args, seed, args.mode)
        with open(args.out, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
