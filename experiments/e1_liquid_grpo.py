#!/usr/bin/env python3
"""E1 — Liquid GRPO: does pure RL make the liquid model discover length
generalization on parity? (C1 experiment, mirrors DeepSeek R1's GRPO.)

Arms:
  supervised : train on L in {1..32} with CE (existing protocol baseline)
  grpo       : RL from random init, reward = whole-sequence exact match
  grpo_finetune : GRPO warm-started from the supervised checkpoint

Evaluation: whole-sequence accuracy at L in {32, 48, 64, 96, 128}.

Deps: M1 repo's mt_lnn package (liquid model) + torch.
Run:  python experiments/e1_liquid_grpo.py --mode grpo --seeds 0 1 2
      python experiments/e1_liquid_grpo.py --mode supervised --seeds 0 1 2
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "M1"))

import torch

from mt_lnn.config import MTLNNConfig
from mt_lnn.model import MTLNNModel

# ---- parity task (matches M1 benchmarks/reasoning_tasks.py semantics) ----
def gen_parity_batch(rng, L, batch):
    """Random binary sequences of length L; label = parity flip of the seq."""
    xs = rng.integers(0, 2, size=(batch, L)).astype("int64")
    # parity: output = xor of all bits repeated (classification target)
    ys = (xs.sum(axis=1) % 2).astype("int64")
    return torch.from_numpy(xs), torch.from_numpy(ys)


class ParityTask:
    def __init__(self, vocab_size=8, seed=0):
        self.vocab_size = vocab_size
        self.rng = __import__("numpy").random.default_rng(seed)
        # token layout: 0=pad, 1=0-bit, 2=1-bit, 3=EOS, 4..=labels
        self.tok0 = 1
        self.tok1 = 2
        self.eos = 3
        self.lab0 = 4
        self.lab1 = 5

    def encode_with_label(self, xs, ys):
        """Return (input_seq, labels) both (B, L+1).
        input  = [b0, ..., b_{L-1}, lab]  (model must emit lab AFTER all bits)
        labels = [dummy, b1, ..., b_{L-1}, lab]  (shifted by one; dummy dropped
                 by the model's labels[:, 1:] slice)"""
        toks = xs + self.tok0  # 0->1, 1->2
        lab_tok = ys + self.lab0  # 0->4, 1->5
        inp = torch.cat([toks, lab_tok.unsqueeze(-1)], dim=-1)  # (B, L+1)
        dummy = torch.zeros_like(toks[:, :1])
        labels = torch.cat([dummy, toks[:, 1:], lab_tok.unsqueeze(-1)], dim=-1)
        return inp, labels

    def sample(self, L, batch):
        xs, ys = gen_parity_batch(self.rng, L, batch)
        return self.encode_with_label(xs, ys)


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
    """Whole-sequence exact-match: model sees bits, must output the label
    token as its LAST generated token. Greedy decode of the final position."""
    model.eval()
    accs = {}
    with torch.no_grad():
        for L in lengths:
            xs, ys = gen_parity_batch(task.rng, L, batch)
            inp = (xs + task.tok0).to(device)
            lab = (ys + task.lab0).to(device)
            out = model(inp)  # (B, L, V)
            pred = out["logits"][:, -1].argmax(-1)  # label prediction
            acc = (pred == lab).float().mean().item()
            accs[L] = round(acc, 3)
    model.train()
    return accs


def supervised(args, seed, device):
    """CE training on L in {1..32}, then eval length generalization."""
    task = ParityTask(seed=seed)
    model = build_model(args, seed).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps = args.supervised_steps
    step = 0
    while step < steps:
        L = int(task.rng.integers(1, args.train_max + 1))
        inp, labels = task.sample(L, args.batch)
        inp, labels = inp.to(device), labels.to(device)
        opt.zero_grad()
        out = model(inp, labels=labels, use_cache=False)
        # NOTE: lm_loss is detached in the model; use the trainable total loss
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 500 == 0:
            print(f"  sup step {step} loss {loss.item():.3f}", flush=True)
    accs = eval_acc(model, task, args.eval_lengths, device)
    print(f"[supervised] accs={accs}", flush=True)
    return model, accs


def grpo(args, seed, device, init_model=None):
    """Group Relative Policy Optimization: sample G per prompt, reward on
    whole-sequence correctness, group-normalized advantage."""
    task = ParityTask(seed=seed)
    model = build_model(args, seed).to(device)
    if init_model is not None:
        model.load_state_dict(init_model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    G = args.g
    step = 0
    t0 = time.time()
    while step < args.grpo_steps:
        # sample one problem per group; generate G label tokens per prompt
        L = int(task.rng.integers(1, args.train_max + 1))
        xs, ys = gen_parity_batch(task.rng, L, args.batch)
        inp = xs + task.tok0
        lab = ys + task.lab0
        # repeat each prompt G times for group sampling
        inp_g = inp.repeat_interleave(G, dim=0).to(device)
        lab_g = lab.repeat_interleave(G, dim=0).to(device)
        # sample label tokens with temperature (exploration)
        with torch.no_grad():
            out = model(inp_g)
            logits = out["logits"][:, -1] / args.temperature
            probs = torch.softmax(logits, -1)
            pred = torch.multinomial(probs, 1).squeeze(-1)
        # rewards: exact match of predicted label
        r = (pred == lab_g).float()  # (B*G,)
        # group-normalized advantage (reshape B, G)
        r = r.view(-1, G)
        adv = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + 1e-4)
        adv = adv.view(-1)
        # policy gradient: maximize log prob of chosen tokens weighted by adv
        opt.zero_grad()
        out = model(inp_g)
        logits = out["logits"][:, -1]
        logp = torch.log_softmax(logits, -1)
        logp_sel = logp.gather(1, pred.unsqueeze(-1)).squeeze(-1)
        # KL to init/reference policy (prevent collapse)
        loss = -(logp_sel * adv).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += args.batch  # count samples
        if step % 500 == 0:
            acc = (r.view(-1).mean().item())
            print(f"  grpo step {step} loss {loss.item():.3f} mean_r {acc:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    accs = eval_acc(model, task, args.eval_lengths, device)
    print(f"[grpo] accs={accs}", flush=True)
    return model, accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["supervised", "grpo", "grpo_finetune"],
                    default="grpo")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=13)
    ap.add_argument("--n_kv_heads", type=int, default=1)
    ap.add_argument("--max_seq_len", type=int, default=256)
    ap.add_argument("--vocab_size", type=int, default=8)
    ap.add_argument("--selective", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--train_max", type=int, default=32)
    ap.add_argument("--eval_lengths", type=int, nargs="+",
                    default=[32, 48, 64, 96, 128])
    ap.add_argument("--supervised_steps", type=int, default=4000)
    ap.add_argument("--grpo_steps", type=int, default=8000)
    ap.add_argument("--g", type=int, default=8, help="group size for GRPO")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", default="results_24h/e1_liquid_grpo.jsonl")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} mode={args.mode} d_model={args.d_model} "
          f"selective={args.selective}", flush=True)
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
