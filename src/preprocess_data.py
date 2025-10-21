# scripts/preprocess_data.py
# Intialized by GPT-5
import argparse
from pathlib import Path
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import json
import numpy as np
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_name", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_examples", type=int, default=2000)
    parser.add_argument("--tokenizer", type=str, default="gpt2-medium")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--text_field", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--config_name", type=str, default=None)

    return parser.parse_args()


def load_and_slice(hf_name, split="train", max_examples=5000, text_field=None, config_name=None):
    ds = load_dataset(hf_name, config_name, split=f"{split}[:{max_examples}]")
    # if dataset uses different field names, try to detect a text-like field
    if text_field is None:
        for candidate in ["text", "content", "article", "abstract", "sentence"]:
            if candidate in ds.column_names:
                text_field = candidate
                break
    if text_field is None:
        raise ValueError("No text field found, please set --text_field")
    return ds.map(lambda ex: { "text": ex[text_field] }, remove_columns=ds.column_names)

def tokenize_and_save(ds, tokenizer_name, out_dir, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tok(ex):
        return tokenizer(ex["text"], truncation=True, padding="max_length", max_length=max_length)
    
    print(tok)
    tokenized = ds.map(tok, batched=True, remove_columns=["text"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(str(out_dir))
    print(f"Saved tokenized dataset to {out_dir}")

if __name__ == "__main__":
    args = parse_args()
    ds = load_and_slice(args.hf_name, split=args.split, max_examples=args.max_examples, text_field=args.text_field, config_name=args.config_name)
    tokenize_and_save(ds, args.tokenizer, args.out_dir, max_length=args.max_length)
