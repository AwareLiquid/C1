#!/usr/bin/env python3
"""prepare_data_e2.py — tokenize WikiText-103 for E2 MTP (gpt2, uint16 .bin).

Mirrors M1's prepare_data.py convention: docs joined with eos_token,
flat uint16 arrays under data/train.bin + data/validation.bin (the layout
experiments/e2_mtp.py load_data() expects).

Run:
  python experiments/prepare_data_e2.py                # full (downloads WT-103)
  python experiments/prepare_data_e2.py --max_docs 2000  # small dev subset
"""
import argparse
import os

import numpy as np
from tokenizers import Tokenizer
from transformers import AutoTokenizer

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--max_docs", type=int, default=0,
                    help="limit docs per split (0 = full); dev subset avoids the 500MB download")
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--from_text", default="",
                    help="tokenize a plain .txt instead of datasets (no network); "
                         "docs = paragraphs split on blank lines; validation = last 10%")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    eos = tok.eos_token_id

    if args.from_text:
        with open(args.from_text, encoding="utf-8") as f:
            raw = f.read()
        docs = [d.strip() for d in raw.split("\n\n") if d.strip()]
        if args.max_docs:
            docs = docs[: args.max_docs]
        n_val = max(1, len(docs) // 10)
        splits = {"validation": docs[:n_val], "train": docs[n_val:]}
        for split, sdocs in splits.items():
            ids = []
            for t in sdocs:
                ids.extend(tok(t, add_special_tokens=False)["input_ids"])
                ids.append(eos)
            arr = np.array(ids, dtype=np.uint16)
            assert arr.max() <= 65535, "token id exceeds uint16 range"
            path = os.path.join(args.out_dir, "train.bin" if split == "train" else "validation.bin")
            arr.tofile(path)
            print("[%s] %d tokens -> %s" % (split, len(ids), path))
        return

    from datasets import load_dataset
    for split, name in [("train", "train.bin"), ("validation", "validation.bin")]:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        ids = []
        texts = [t for t in ds["text"] if t]
        if args.max_docs:
            texts = texts[: args.max_docs]
        for t in texts:
            row = tok(t, add_special_tokens=False)["input_ids"]
            ids.extend(row)
            ids.append(eos)
        arr = np.array(ids, dtype=np.uint16)
        if arr.max() > 65535:
            raise SystemExit("token id exceeds uint16 range")
        path = os.path.join(args.out_dir, name)
        arr.tofile(path)
        print(f"[{split}] {len(ids)} tokens -> {path}")


if __name__ == "__main__":
    main()
