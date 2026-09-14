#!/usr/bin/env python3
"""E7 — Data-form probe (类脑数据形态探针, 探针 A: 事件边界切片).

The liquid stack's τ-ladder / streaming-state mechanisms need temporal
STRUCTURE in the data (event boundaries, real timing). Standard LLM corpora
are arbitrary fixed chunks — no structure for those mechanisms to act on.

Probe A (this script, ZERO model change): cut the SAME token stream two ways —
  fixed     : sequential view(-1, seq_len) arbitrary cuts (standard LLM form)
  boundary  : chunks packed at synthetic sentence boundaries (a sentence never
              crosses a chunk; overflow truncated, pad positions ignored via
              labels=-100)

Both modes use the same tokens, model config and budget. If the boundary form
changes val PPL significantly (paired 3-seed, >=2 sigma), the liquid stack has
a "load-bearing surface" for stream form -> probe B (input-dependent dt).

Synthetic corpus: topic-conditioned sentences — each sentence draws its tokens
from one topic's Dirichlet emission; event boundary = topic switch, so the
boundary carries real semantic structure (not random cuts).

Run:
  python experiments/e7_data_form_probe.py --smoke --mode boundary --steps 3
  python experiments/e7_data_form_probe.py --mode boundary --steps 4000 --seeds 0 1 2
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mt_lnn.config import MTLNNConfig  # noqa: E402
from mt_lnn.model import MTLNNModel  # noqa: E402


def synthetic_corpus(n_tokens, seq_len, vocab_size, n_topics, seed):
    """Topic-conditioned sentences; returns (tokens, sentence_starts)."""
    rng = np.random.default_rng(seed)
    topic_probs = [rng.dirichlet(np.ones(vocab_size) * 0.1) for _ in range(n_topics)]
    tokens, starts = [], [0]
    topic = 0
    while len(tokens) < n_tokens:
        sent_len = int(rng.geometric(0.12) + 1)  # 1..~60 tokens per sentence
        sent_len = min(sent_len, 96, n_tokens - len(tokens))
        for _ in range(sent_len):
            tokens.append(int(rng.choice(vocab_size, p=topic_probs[topic])) + 1)
        if len(tokens) < n_tokens:
            starts.append(len(tokens))
            topic = (topic + rng.integers(1, n_tokens // 10 + 1)) % n_topics
    return torch.tensor(tokens[:n_tokens], dtype=torch.long), np.asarray(starts)


def make_fixed(tokens, seq_len):
    return tokens[: len(tokens) // seq_len * seq_len].view(-1, seq_len)


def make_boundary(tokens, starts, seq_len):
    """Pack sentences into seq_len chunks; pad positions -> token 0 (ignored)."""
    # sentence token lists
    starts = list(starts) + [len(tokens)]
    sentences = [tokens[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]
    chunks = []
    cur = []
    for s in sentences:
        s = s[:seq_len]  # truncate overlong sentences
        if len(cur) + len(s) <= seq_len:
            cur.extend(s.tolist())
        else:
            chunks.append(cur + [0] * (seq_len - len(cur)))
            cur = list(s.tolist())
        if len(cur) == seq_len:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur + [0] * (seq_len - len(cur)))
    if not chunks:
        chunks.append([0] * seq_len)
    return torch.tensor(chunks[: len(chunks)], dtype=torch.long)


def build_cfg(args):
    return MTLNNConfig(
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads, d_head=args.d_model // args.n_heads,
        max_seq_len=args.seq_len, vocab_size=args.vocab_size,
        gwtb_n_heads=1,  # tiny/smoke configs: d_gw=13 must divide heads
    )


def train_eval(args, seed, mode):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_chunks = args.n_chunks if not args.smoke else 64
    n_tokens = n_chunks * args.seq_len * 2  # slack for boundary packing
    tokens, starts = synthetic_corpus(
        n_tokens, args.seq_len, args.vocab_size, args.n_topics, seed)
    if mode == "fixed":
        tr_chunks = make_fixed(tokens, args.seq_len)
    else:
        tr_chunks = make_boundary(tokens, starts, args.seq_len)
    # val: independent corpus, same form
    vt, vs = synthetic_corpus(
        n_chunks * args.seq_len, args.seq_len, args.vocab_size, args.n_topics,
        seed + 500)
    va_chunks = make_fixed(vt, args.seq_len) if mode == "fixed" else \
        make_boundary(vt, vs, args.seq_len)
    tr_chunks = tr_chunks[:n_chunks]
    va_chunks = va_chunks[:max(1, n_chunks // 4)]
    print(f"  [seed {seed}] {mode} chunks={len(tr_chunks)}/{len(va_chunks)}",
          flush=True)

    def labels_for(x):
        lab = x.clone()
        lab[x == 0] = -100  # ignore pad positions (boundary mode only)
        return lab

    model = MTLNNModel(build_cfg(args)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    step = 0
    for epoch in range(10):
        perm = torch.randperm(len(tr_chunks),
                              generator=torch.Generator().manual_seed(seed + 99))
        for i in range(0, len(perm), args.batch):
            inp = tr_chunks[perm[i: i + args.batch]].to(device)
            opt.zero_grad()
            out = model(inp, labels=labels_for(inp))
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

    model.eval()
    total_ce, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(va_chunks), args.batch):
            inp = va_chunks[i: i + args.batch].to(device)
            out = model(inp, labels=labels_for(inp))
            ce = out.get("lm_loss", out["loss"])
            total_ce += float(ce)
            n += 1
    ppl = math.exp(min(total_ce / max(n, 1), 20.0))
    print(f"  [seed {seed}] {mode} val_ppl={ppl:.3f}", flush=True)
    return {"mode": mode, "seed": seed, "val_ppl": round(ppl, 3),
            "steps": args.steps, "n_topics": args.n_topics}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fixed", "boundary"], default="boundary")
    ap.add_argument("--d_model", type=int, default=832)
    ap.add_argument("--n_layers", type=int, default=12)
    ap.add_argument("--n_heads", type=int, default=13)
    ap.add_argument("--n_kv_heads", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--vocab_size", type=int, default=200)
    ap.add_argument("--n_topics", type=int, default=8)
    ap.add_argument("--n_chunks", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--smoke", action="store_true",
                    help="tiny config hint flag (data is always synthetic)")
    ap.add_argument("--out", default="results/e7_data_form_probe.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for seed in args.seeds:
        r = train_eval(args, seed, args.mode)
        with open(args.out, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
