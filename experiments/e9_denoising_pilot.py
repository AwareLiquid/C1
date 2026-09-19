#!/usr/bin/env python3
"""E9 — Denoising supervision pilot for the liquid stack (brain-like data form).

The liquid stack's inductive bias (continuous-time state evolution) has no
load-bearing surface under next-token supervision on discrete static text
(liquid-research-skill data-form thesis; E7 probe A confirmed empirically).

E9 swaps the SUPERVISION SIGNAL, not the architecture, on the parity domain
(proven learnable — ADJ-001 budget-wall lesson):
  next_token  : clean input -> answer-only CE (e3 protocol; baseline)
  denoise     : corrupted input (bit flips p) -> answer-only CE
                (state-level denoising: the recurrence must filter the noise
                 to extract the clean parity)
  progressive : K=3 self-refinement passes of denoise (diffusion-like gradual
                correction); each pass's input = previous pass's argmax output

Metric: length extrapolation (train k<=32 -> test k=48..128), the liquid stack's
signature advantage. Pre-registered: denoise/progressive must beat next_token on
>=2 extrapolation lengths (3-seed paired sign-test) or H-E9 is DEAD.

Protocol history: v1 (6K steps) budget wall; v2 (full-sequence CE) diluted the
answer gradient 1/32; v3 (hand-rolled layout) misaligned with the model's
answer-slot semantics. v4 = gen_parity (the e3-proven generator) end-to-end.

Run:
  python experiments/e9_denoising_pilot.py --smoke --mode denoise --steps 3
  python experiments/e9_denoising_pilot.py --mode denoise --steps 20000 --seeds 0 1 2
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

# gen_parity lives in the vendored benchmarks (C1 carries a copy of reasoning_tasks)
from benchmarks.reasoning_tasks import gen_parity  # noqa: E402

VOCAB = 64
CORRUPT_P = 0.15
K_PASSES = 3


def make_batch(batch, k_bits, p_corrupt, rng):
    """gen_parity batch + bit corruption. Returns (tokens, answer)."""
    b = gen_parity(batch, k_bits, rng)
    toks = b.tokens.copy()
    if p_corrupt > 0.0:
        flip = rng.random((batch, k_bits)) < p_corrupt
        bits = toks[:, 1: 1 + k_bits]
        # bits are VALUE_BASE+0/1 (10/11): XOR the low bit with 1
        bits[flip] ^= 1
    return torch.from_numpy(toks).long(), torch.from_numpy(b.answer).long()


def forward_loss(model, inp, answer, device):
    """Answer-only CE (e3 protocol): labels[:, -1] = answer, predicted from T-2."""
    labels = torch.full_like(inp, -100)
    labels[:, -1] = answer
    out = model(inp.to(device), labels=labels.to(device))
    with torch.no_grad():
        pred_tokens = out["logits"].argmax(-1)
    return out["loss"], pred_tokens


def build_cfg(args):
    return MTLNNConfig(
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads, d_head=args.d_model // args.n_heads,
        max_seq_len=args.max_seq_len, vocab_size=VOCAB,
        gwtb_n_heads=1,
    )


def train_eval(args, seed, mode):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    model = MTLNNModel(build_cfg(args)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    step = 0
    while step < args.steps:
        k = int(rng.integers(args.min_len, args.max_len + 1))
        p = 0.0 if mode == "next_token" else args.corrupt_p
        inp, answer = make_batch(args.batch, k, p, rng)
        opt.zero_grad(set_to_none=True)
        total = 0.0
        cur_inp = inp
        n_passes = 1 if mode != "progressive" else K_PASSES
        for kpass in range(n_passes):
            loss, pred_tokens = forward_loss(model, cur_inp, answer, device)
            w = 1.0 / (kpass + 1)
            (w * loss).backward()
            total += loss.item()
            if kpass + 1 < n_passes:
                cur_inp = torch.cat(
                    [inp[:, :1].to(device), pred_tokens[:, :-1]], dim=1)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 2000 == 0:
            print(f"  [{mode} seed{seed}] step {step} loss {total:.3f}", flush=True)

    model.eval()
    accs = {}
    with torch.no_grad():
        for L in (32, 48, 64, 96, 128):
            b = gen_parity(256, L, np.random.default_rng(10_000 + seed))
            ids = torch.from_numpy(b.tokens).to(device)
            lab_t = torch.from_numpy(b.answer).to(device)
            out = model(ids)
            pred = out["logits"][:, -2].argmax(-1)  # T-2 (THINK) predicts ANS
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
    ap.add_argument("--min_len", type=int, default=1,
                    help="shortest k_bits in training mix (e3 starts at 1; "
                         "v4 used 8 and never grokked)")
    ap.add_argument("--max_len", type=int, default=32)
    ap.add_argument("--max_seq_len", type=int, default=140,
                    help="model max_seq_len (covers eval L=128 + BOS/THINK/ANS)")
    ap.add_argument("--corrupt_p", type=float, default=CORRUPT_P)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=20000)
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
