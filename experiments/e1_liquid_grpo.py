#!/usr/bin/env python3
"""E1 — Liquid GRPO: does pure RL make the liquid model discover length
generalization on parity? (C1 experiment, mirrors DeepSeek R1's GRPO.)

Arms:
  supervised : train on L in {1..32} with CE ONLY at the answer position
               (exactly the M1 parity_lengthgen protocol: [BOS] bits...
               [THINK] [ANS], labels masked -100 except ans_pos)
  grpo       : RL from random init, reward = exact match of the sampled
               answer token, group-normalized advantage (G samples/prompt)
  grpo_finetune : GRPO warm-started from the supervised checkpoint

Evaluation: whole-sequence accuracy at L in {32, 48, 64, 96, 128},
read at logits[:, ans_pos - 1] (shift semantics: position t predicts
labels[t+1], so the answer is read after the THINK token).

Deps: M1 repo's mt_lnn package (liquid model) + benchmarks.reasoning_tasks.
Run:  python experiments/e1_liquid_grpo.py --mode grpo --seeds 0 1 2
      python experiments/e1_liquid_grpo.py --mode supervised --seeds 0 1 2
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "M1"))

import numpy as np
import torch

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel
from benchmarks.reasoning_tasks import gen_parity, vocab_size as rt_vocab_size

TRAIN_MAX = 32
EVAL_LENGTHS = (32, 48, 64, 96, 128)


class ParityTask:
    """Wraps M1's gen_parity (the exact protocol that produced the
    selective >> stock length-extrapolation results)."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.eval_rng = np.random.default_rng(10_000 + seed)

    def sample(self, L, batch):
        """(ids, labels) both (B, T): labels masked -100 except ans_pos."""
        b = gen_parity(batch, L, self.rng)
        ids = torch.from_numpy(b.tokens)
        labels = torch.full_like(ids, -100)
        labels[:, b.ans_pos] = torch.from_numpy(b.answer)
        return ids, labels, b.ans_pos


def build_model(args, seed):
    torch.manual_seed(seed)
    cfg = MTLNNConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        d_head=args.d_model // args.n_heads,
        max_seq_len=args.max_seq_len,
        vocab_size=args.vocab_size,
        selective_decay=args.selective,
    )
    return MTLNNModel(cfg)


def eval_acc(model, task, lengths, device, batch=128):
    """Read the answer at logits[:, ans_pos - 1] (the THINK position),
    exactly as parity_lengthgen.py does."""
    model.eval()
    accs = {}
    with torch.no_grad():
        for L in lengths:
            b = gen_parity(batch, L, task.eval_rng)
            ids = torch.from_numpy(b.tokens).to(device)
            ans = torch.from_numpy(b.answer).to(device)
            pred = model(ids)["logits"][:, b.ans_pos - 1].argmax(-1)
            accs[L] = round((pred == ans).float().mean().item(), 3)
    model.train()
    return accs


def supervised(args, seed, device):
    """CE at the answer position only (masked -100 elsewhere), L ~ U{1..32}."""
    task = ParityTask(seed=seed)
    model = build_model(args, seed).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.supervised_steps)
    for step in range(args.supervised_steps):
        L = int(task.rng.integers(1, TRAIN_MAX + 1))
        ids, labels, _ = task.sample(L, args.batch)
        ids, labels = ids.to(device), labels.to(device)
        opt.zero_grad(set_to_none=True)
        out = model(ids, labels=labels, use_cache=False)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.supervised_steps - 1:
            print(f"  sup step {step} loss {loss.item():.3f}", flush=True)
    accs = eval_acc(model, task, EVAL_LENGTHS, device)
    print(f"[supervised] accs={accs}", flush=True)
    return model, accs


def grpo(args, seed, device, init_model=None):
    """Group Relative Policy Optimization on the answer token.

    Per prompt: sample G answer tokens (temperature), reward = exact match,
    group-normalized advantage, policy-gradient update on log p(sampled).
    Mirrors R1's GRPO at the token level; no critic, no value net.
    """
    task = ParityTask(seed=seed)
    model = build_model(args, seed).to(device)
    if init_model is not None:
        model.load_state_dict(init_model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    G = args.g
    t0 = time.time()
    for step in range(args.grpo_steps):
        L = int(task.rng.integers(1, TRAIN_MAX + 1))
        b = gen_parity(args.batch, L, task.rng)
        ids = torch.from_numpy(b.tokens)                     # (B, T)
        ans = torch.from_numpy(b.answer)                     # (B,)
        ans_pos = b.ans_pos
        ids_g = ids.repeat_interleave(G, dim=0).to(device)   # (B*G, T)
        ans_g = ans.repeat_interleave(G, dim=0).to(device)   # (B*G,)

        with torch.no_grad():
            logits = model(ids_g)["logits"][:, ans_pos - 1] / args.temperature
            probs = torch.softmax(logits, -1)
            pred = torch.multinomial(probs, 1).squeeze(-1)   # (B*G,)

        r = (pred == ans_g).float().view(-1, G)              # (B, G)
        adv = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + 1e-4)
        adv = adv.view(-1)

        opt.zero_grad(set_to_none=True)
        logits = model(ids_g)["logits"][:, ans_pos - 1]
        logp = torch.log_softmax(logits, -1)
        logp_sel = logp.gather(1, pred.unsqueeze(-1)).squeeze(-1)
        loss = -(logp_sel * adv).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 500 == 0 or step == args.grpo_steps - 1:
            mean_r = r.mean().item()
            print(f"  grpo step {step} loss {loss.item():.3f} mean_r {mean_r:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    accs = eval_acc(model, task, EVAL_LENGTHS, device)
    print(f"[grpo] accs={accs}", flush=True)
    return model, accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["supervised", "grpo", "grpo_finetune"],
                    default="grpo")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--d_model", type=int, default=416)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=13)
    ap.add_argument("--n_kv_heads", type=int, default=1)
    ap.add_argument("--max_seq_len", type=int, default=256)
    ap.add_argument("--vocab_size", type=int, default=rt_vocab_size(2))
    ap.add_argument("--selective", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--supervised_steps", type=int, default=4000)
    ap.add_argument("--grpo_steps", type=int, default=8000)
    ap.add_argument("--g", type=int, default=8, help="group size for GRPO")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", default="results_24h/e1_liquid_grpo.jsonl")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} mode={args.mode} d_model={args.d_model} "
          f"selective={args.selective} vocab={args.vocab_size}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for seed in args.seeds:
        if args.mode == "supervised":
            model, accs = supervised(args, seed, device)
            row = {"mode": "supervised", "seed": seed, "accs": accs}
        elif args.mode == "grpo":
            model, accs = grpo(args, seed, device)
            row = {"mode": "grpo", "seed": seed, "accs": accs}
        else:  # grpo_finetune
            sup_model, sup_accs = supervised(args, seed, device)
            model, accs = grpo(args, seed, device, init_model=sup_model)
            row = {"mode": "grpo_finetune", "seed": seed,
                   "sup_accs": sup_accs, "accs": accs}
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
