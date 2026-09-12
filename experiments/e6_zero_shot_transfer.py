#!/usr/bin/env python3
"""E6 — Cross-domain zero-shot transfer for C1 (EnergyTS-inspired).

Pretrain one liquid model on mixed synthetic domains; evaluate ZERO-SHOT on a
held-out domain. Tests the "foundation-model cold-start" capability claim for
the liquid stack (cf. EnergyTS 零样本冷启).

Domains (mechanism-disjoint, synthetic — no data files needed):
  periodic : sin(2*pi*f*t) + noise        (narrow-band spectrum)
  ar       : x_t = 0.9*x_{t-1} + eps      (exponential ACF)
  events   : sparse spikes, p ~ 0.02      (heavy tail, long gaps)

Protocol (leave-one-domain-out):
  for held_out in domains: train on the other two -> zero-shot eval held_out
  controls: (1) single-domain pretrain, (2) random init

Run:
  python experiments/e6_zero_shot_transfer.py --smoke --steps 3
  python experiments/e6_zero_shot_transfer.py --steps 2000 --seeds 0 1 2
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

DOMAINS = ("periodic", "ar", "events")


def gen_domain(name, n, seed):
    """Generate one synthetic domain as a float64 series (length n)."""
    rng = np.random.default_rng(seed)
    if name == "periodic":
        f = rng.uniform(0.02, 0.2)
        t = np.arange(n, dtype=np.float64)
        x = np.sin(2 * np.pi * f * t) + 0.3 * rng.standard_normal(n)
    elif name == "ar":
        x = np.zeros(n)
        eps = rng.standard_normal(n)
        for i in range(1, n):
            x[i] = 0.9 * x[i - 1] + eps[i]
    elif name == "events":
        x = 0.05 * rng.standard_normal(n)
        spikes = rng.random(n) < 0.02
        x[spikes] += rng.uniform(1.0, 3.0, size=int(spikes.sum()))
    else:
        raise ValueError(f"unknown domain: {name}")
    return x


def quantize(x, vocab_size):
    """Fixed z-score quantization to tokens in [1, vocab_size-1]."""
    z = (x - x.mean()) / (x.std() + 1e-9)
    z = np.clip(z, -3.0, 3.0) / 3.0  # [-1, 1]
    q = 1 + np.round((z + 1.0) / 2.0 * (vocab_size - 2)).astype(np.int64)
    return np.clip(q, 1, vocab_size - 1)


def domain_chunks(name, n_chunks, seq_len, vocab_size, seed):
    """One domain as (n_chunks, seq_len) token chunks."""
    x = gen_domain(name, n_chunks * seq_len, seed)
    q = quantize(x, vocab_size)
    return torch.from_numpy(q).view(n_chunks, seq_len)


def build_cfg(args):
    return MTLNNConfig(
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads, d_head=args.d_model // args.n_heads,
        max_seq_len=args.seq_len, vocab_size=args.vocab_size,
        gwtb_n_heads=1,  # tiny/smoke configs: d_gw=13 must divide heads
    )


def train_model(args, train_domains, seed, device):
    """Standard LM training on the given domains (chunks concatenated)."""
    torch.manual_seed(seed)
    model = MTLNNModel(build_cfg(args)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    tr = torch.cat([
        domain_chunks(d, args.n_chunks, args.seq_len, args.vocab_size,
                      seed + 100 + 7 * i)
        for i, d in enumerate(train_domains)
    ], dim=0)
    print(f"  [seed {seed}] pretrain on {train_domains} chunks={len(tr)}", flush=True)

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
                print(f"    step {step} loss {loss.item():.3f}", flush=True)
            if step >= args.steps:
                break
        if step >= args.steps:
            break
    return model


def eval_ce(model, args, domain, seed, device):
    """Zero-shot CE of a model on a domain it may never have seen."""
    va = domain_chunks(domain, args.n_chunks, args.seq_len, args.vocab_size,
                       seed + 500)
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(va), args.batch):
            inp = va[i: i + args.batch].to(device)
            out = model(inp, labels=inp)
            ce = out.get("lm_loss", out["loss"])
            total += float(ce)
            n += 1
    return total / max(n, 1)


def run_protocol(args, seed, device):
    rows = []
    for held in DOMAINS:
        train_domains = [d for d in DOMAINS if d != held]
        # 1) multi-domain pretrain -> zero-shot eval on held-out domain
        m = train_model(args, train_domains, seed, device)
        zs = eval_ce(m, args, held, seed, device)
        # 2) single-domain pretrain control (first training domain)
        m1 = train_model(args, [train_domains[0]], seed, device)
        single = eval_ce(m1, args, held, seed, device)
        # 3) random-init control (same architecture / init seed)
        torch.manual_seed(seed)
        rnd = MTLNNModel(build_cfg(args)).to(device)
        rnd_ce = eval_ce(rnd, args, held, seed, device)
        row = {
            "held_out": held, "seed": seed,
            "zero_shot_ce": round(zs, 4),
            "single_domain_ce": round(single, 4),
            "random_init_ce": round(rnd_ce, 4),
            "gain_vs_random": round(rnd_ce - zs, 4),
            "gain_multi_vs_single": round(single - zs, 4),
            "steps": args.steps,
        }
        print(f"  [seed {seed}] held_out={held} " + json.dumps(row), flush=True)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_model", type=int, default=832)
    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--n_heads", type=int, default=13)
    ap.add_argument("--n_kv_heads", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--vocab_size", type=int, default=200)
    ap.add_argument("--n_chunks", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--smoke", action="store_true",
                    help="tiny config hint flag (data is always synthetic here)")
    ap.add_argument("--out", default="results/e6_zero_shot_transfer.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for seed in args.seeds:
        rows = run_protocol(args, seed, device)
        with open(args.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
