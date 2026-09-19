#!/usr/bin/env python3
"""E9 — Denoising supervision pilot for the liquid stack (brain-like data form).

The liquid stack's inductive bias (continuous-time state evolution) has no
load-bearing surface under next-token supervision on discrete static text
(liquid-research-skill data-form thesis; E7 probe A confirmed empirically).

E9 swaps the SUPERVISION SIGNAL, not the architecture, on the parity domain
(proven learnable — ADJ-001 budget-wall lesson):
  next_token  : clean input -> CE on next token (transformer-standard; baseline)
  denoise     : corrupted input (bit flips p) -> CE against the CLEAN sequence
                (predictive-coding data form: corrupted->clean pairs)
  progressive : K=3 self-refinement passes of denoise (diffusion-like gradual
                correction); each pass's input = previous pass's argmax output

Metric: length extrapolation (train L<=32 -> test L=48..128), the liquid stack's
signature advantage. Pre-registered: denoise/progressive must beat next_token on
>=2 extrapolation lengths (3-seed paired sign-test) or H-E9 is DEAD.

Run:
  python experiments/e9_denoising_pilot.py --smoke --mode denoise --steps 3
  python experiments/e9_denoising_pilot.py --mode denoise --steps 6000 --seeds 0 1 2
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mt_lnn.config import MTLNNConfig  # noqa: E402
from mt_lnn.model import MTLNNModel  # noqa: E402

TOK0, TOK1, LAB0 = 1, 2, 4   # bit tokens + answer base (e3 protocol)
VOCAB = 200
CORRUPT_P = 0.15
K_PASSES = 3


def make_batch(batch, L, p_corrupt, rng, *, clean=False):
    """Parity batch: L positions (L-1 bits + answer). Corrupted variant flips bits."""
    xs = rng.integers(0, 2, size=(batch, L - 1)).astype("int64")
    ys = (xs.sum(axis=1) % 2).astype("int64")
    if clean or p_corrupt == 0.0:
        xc = xs.copy()
    else:
        flip = rng.random((batch, L - 1)) < p_corrupt
        xc = xs.copy()
        xc[flip] = 1 - xc[flip]
    # input: bits as token stream (answer position fed the masked "?" = TOK1+TOK0? no:
    # keep e3 layout — last position carries a bit token; the answer is the label there)
    inp = np.concatenate([xc, np.zeros((batch, 1), dtype="int64")], axis=1) + TOK0
    # clean full-sequence target (bits + answer), shift-matched below
    clean_seq = np.concatenate([xs, ys[:, None]], axis=1)  # (B, L)
    return torch.from_numpy(inp).long(), torch.from_numpy(clean_seq).long()


def forward_loss(model, inp, clean_seq, device):
    """One model pass; CE of position t's logits against clean_seq[t+1]
    (M1 shift semantics — e1/e3 protocol). Returns (loss, argmax_tokens)."""
    labels = torch.full_like(inp, -100)
    labels[:, -1] = clean_seq[:, -1] + LAB0  # answer-only supervision (e3 protocol):
    # denoising happens at the STATE level — corrupted input flows through the
    # recurrence and the model must extract the clean parity to answer.
    # (v1/v2 full-sequence CE diluted the answer gradient 1/32 -> never grokked)
    out = model(inp.to(device), labels=labels.to(device))
    with torch.no_grad():
        pred_tokens = out["logits"].argmax(-1)  # (B, L) — position t predicts t+1
    return out["loss"], pred_tokens


def build_cfg(args):
    return MTLNNConfig(
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads, d_head=args.d_model // args.n_heads,
        max_seq_len=args.max_len, vocab_size=VOCAB,
        gwtb_n_heads=1,
    )


def train_eval(args, seed, mode):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    model = MTLNNModel(build_cfg(args)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    step = 0
    while step < args.steps:
        L = int(rng.integers(args.min_len, args.max_len + 1))
        p = 0.0 if mode == "next_token" else args.corrupt_p
        inp, clean_seq = make_batch(args.batch, L, p, rng)
        opt.zero_grad()
        total = 0.0
        cur_inp = inp
        n_passes = 1 if mode != "progressive" else K_PASSES
        for k in range(n_passes):
            loss, pred_tokens = forward_loss(model, cur_inp, clean_seq, device)
            w = 1.0 / (k + 1)  # later passes weigh less (progressive refinement)
            (w * loss).backward()
            total += loss.item()
            if k + 1 < n_passes:
                # self-conditioning: next pass input = previous argmax (shift-rough,
                # pilot-grade refinement signal)
                cur_inp = torch.cat(
                    [inp[:, :1].to(device), pred_tokens[:, :-1]], dim=1)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 500 == 0:
            print(f"  [{mode} seed{seed}] step {step} loss {total:.3f}", flush=True)

    # eval: length extrapolation (clean inputs, answer via position L-2 logits)
    model.eval()
    accs = {}
    with torch.no_grad():
        for L in (32, 48, 64, 96, 128):
            xs = rng.integers(0, 2, size=(256, L)).astype("int64")
            ys = (xs.sum(axis=1) % 2).astype("int64")
            xs_t = torch.from_numpy(xs + TOK0).to(device)
            lab_t = torch.from_numpy(ys + LAB0).to(device)
            out = model(xs_t)
            pred = out["logits"][:, -2].argmax(-1)
            accs[L] = round((pred == lab_t).float().mean().item(), 3)
    print(f"  [{mode} seed{seed}] extrap accs={accs}", flush=True)
    return {"mode": mode, "seed": seed, "accs": accs, "steps": args.steps,
            "corrupt_p": 0.0 if mode == "next_token" else args.corrupt_p,
            "passes": K_PASSES if mode == "progressive" else 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["next_token", "denoise", "progressive"],
                    default="denoise")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_kv_heads", type=int, default=2)
    ap.add_argument("--min_len", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=32)
    ap.add_argument("--corrupt_p", type=float, default=CORRUPT_P)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results/e9_denoising_pilot.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for seed in args.seeds:
        r = train_eval(args, seed, args.mode)
        with open(args.out, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
