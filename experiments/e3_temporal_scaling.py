#!/usr/bin/env python3
"""E3 â€” Temporal-scale scaling: does adding time scales (capacity) with
top-k gating (cost) improve the liquid model? (C1's MoE-style experiment.)

Configs: --n_scales 5|8|16 with --topk 2|5|8|dense.
Capacity is nearly free (scales are scalars); gating keeps compute flat.
Tasks: parity length-gen (fast) + optional WikiText PPL.

Run:
  python experiments/e3_temporal_scaling.py --n_scales 5 --topk dense
  python experiments/e3_temporal_scaling.py --n_scales 16 --topk 5
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel


def build_model(args, seed):
    torch.manual_seed(seed)
    # NOTE: MTLNNConfig.n_time_scales already exists; top-k gating is applied
    # by the sparse-resonance path inside MTLNNLayer (top_k knob). We pass both
    # through and let the layer gate if topk < n_scales.
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
    )
    model = MTLNNModel(cfg)
    # apply top-k sparse resonance if requested (torch.topk over scale weights)
    if args.topk != "dense":
        k = int(args.topk)
        for m in model.modules():
            if hasattr(m, "n_time_scales") and hasattr(m, "topk"):
                m.topk = k
    return model


def run_parity(args, seed, device):
    """Quick capability probe: parity length-gen whole-sequence recall."""
    model = build_model(args, seed).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    rng = __import__("numpy").random.default_rng(seed)
    tok0, tok1, lab0 = 1, 2, 4

    steps = args.parity_steps
    step = 0
    while step < steps:
        L = int(rng.integers(1, 33))
        xs = rng.integers(0, 2, size=(args.batch, L)).astype("int64")
        ys = (xs.sum(axis=1) % 2).astype("int64")
        xs_t = torch.from_numpy(xs + tok0).to(device)
        # M1 shift è¯­ä¹‰ï¼ˆe1 åŒæ¬¾åè®®ï¼‰ï¼šä½ç½® t é¢„æµ‹ labels[t+1]ï¼Œå› æ­¤æŠŠç­”æ¡ˆ
        # æŒ‚åœ¨åºåˆ—æœ«ä½ï¼ˆlabels[:, -1]ï¼‰ï¼Œç”±ä½ç½® L-2 çš„ logits é¢„æµ‹ã€‚
        lab_t = torch.full_like(xs_t, -100)
        lab_t[:, -1] = torch.from_numpy(ys + lab0).to(device)
        opt.zero_grad()
        out = model(xs_t, labels=lab_t)
        loss = out["loss"]   # è®­ç»ƒç›®æ ‡ï¼ˆlm_loss æ˜¯ detach åŽçš„çº¯æŒ‡æ ‡ï¼Œä¸å¯åä¼ ï¼‰
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

    model.eval()
    accs = {}
    with torch.no_grad():
        for L in (32, 48, 64, 96):
            xs = rng.integers(0, 2, size=(256, L)).astype("int64")
            ys = (xs.sum(axis=1) % 2).astype("int64")
            xs_t = torch.from_numpy(xs + tok0).to(device)
            lab_t = torch.from_numpy(ys + lab0).to(device)
            out = model(xs_t)
            pred = out["logits"][:, -2].argmax(-1)   # é¢„æµ‹æœ«ä½ç­”æ¡ˆ
            accs[L] = round((pred == lab_t).float().mean().item(), 3)
    print(f"  [n_scales={args.n_scales} topk={args.topk}] parity accs={accs}", flush=True)
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_scales", type=int, default=5, choices=[5, 8, 16])
    ap.add_argument("--topk", default="dense", help="2|5|8|dense")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=13)
    ap.add_argument("--n_kv_heads", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--vocab_size", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--parity_steps", type=int, default=3000)
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
