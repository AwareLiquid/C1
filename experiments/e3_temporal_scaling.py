#!/usr/bin/env python3
"""E3 — Temporal-scale scaling: does adding time scales (capacity) with
top-k gating (cost) improve the liquid model? (C1's MoE-style experiment.)

Configs: --n_scales 5|8|16 with --topk 2|5|8|dense.
Capacity is nearly free (scales are scalars); gating keeps compute flat.
Task: M1's proven parity length-gen protocol (same gen_parity as E1), so the
capability is actually learnable and configs can be compared.

Run:
  python experiments/e3_temporal_scaling.py --n_scales 5 --topk dense
  python experiments/e3_temporal_scaling.py --n_scales 16 --topk 5
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel
from benchmarks.reasoning_tasks import gen_parity, vocab_size as rt_vocab_size

TRAIN_MAX = 32
EVAL_LENGTHS = (32, 48, 64, 96, 128)


def build_model(args, seed):
    torch.manual_seed(seed)
    # MTLNNLayer gates scales only when sparse_resonance_kernel=True; "dense" keeps all active.
    dense = args.topk == "dense"
    cfg = MTLNNConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        d_head=args.d_model // args.n_heads,
        max_seq_len=args.seq_len,
        gwtb_n_heads=1,  # tiny/smoke configs: d_gw=13 must divide heads
        vocab_size=args.vocab_size,
        n_time_scales=args.n_scales,
        sparse_resonance_kernel=not dense,
        sparse_resonance_top_k=(args.n_scales if dense else int(args.topk)),
        selective_decay=args.selective,
    )
    return MTLNNModel(cfg)


def run_parity(args, seed, device):
    model = build_model(args, seed).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.parity_steps)
    rng = np.random.default_rng(seed)
    for step in range(args.parity_steps):
        L = int(rng.integers(1, TRAIN_MAX + 1))
        b = gen_parity(args.batch, L, rng)
        ids = torch.from_numpy(b.tokens).to(device)
        labels = torch.full_like(ids, -100)
        labels[:, b.ans_pos] = torch.from_numpy(b.answer).to(device)
        opt.zero_grad(set_to_none=True)
        out = model(ids, labels=labels, use_cache=False)
        out["loss"].backward()
        opt.step()
        sched.step()
        if step % 2000 == 0 or step == args.parity_steps - 1:
            print(f"    step {step} loss {out['loss'].item():.3f}", flush=True)

    model.eval()
    eval_rng = np.random.default_rng(10_000 + seed)
    accs = {}
    with torch.no_grad():
        for L in EVAL_LENGTHS:
            b = gen_parity(args.batch, L, eval_rng)
            ids = torch.from_numpy(b.tokens).to(device)
            ans = torch.from_numpy(b.answer).to(device)
            pred = model(ids)["logits"][:, b.ans_pos - 1].argmax(-1)
            accs[L] = round((pred == ans).float().mean().item(), 3)
    print(f"  [n_scales={args.n_scales} topk={args.topk}] parity accs={accs}", flush=True)
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_scales", type=int, default=5, choices=[5, 8, 16])
    ap.add_argument("--topk", default="dense", help="2|5|8|dense")
    ap.add_argument("--d_model", type=int, default=104)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_kv_heads", type=int, default=2)
    ap.add_argument("--selective", dest="selective", action="store_true", default=True,
                    help="selective_decay (M1's proven parity arm; default on so parity is learnable)")
    ap.add_argument("--no-selective", dest="selective", action="store_false")
    ap.add_argument("--seq_len", type=int, default=1 + max(EVAL_LENGTHS) + 2)
    ap.add_argument("--vocab_size", type=int, default=rt_vocab_size(2))
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--parity_steps", type=int, default=30000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results_24h/e3_temporal_scaling.jsonl")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for seed in args.seeds:
        accs = run_parity(args, seed, device)
        row = {"n_scales": args.n_scales, "topk": args.topk,
               "seed": seed, "accs": accs}
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
