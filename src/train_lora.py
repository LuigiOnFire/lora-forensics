#!/usr/bin/env python3
"""
train_lora_lm.py
LoRA fine-tuning script for causal language models (e.g. GPT-2).
This file started out very inspired by huggingface/peft's own train_dreambooth.py example
"""

import os
import math
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler,
    set_seed,
)
from peft import LoraConfig, get_peft_model


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for causal LM")

    parser.add_argument("--model_name_or_path", type=str, default="gpt2")
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="HuggingFace dataset name")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--logging_dir", type=str, default="../experiments/logs")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()

# In 
def main():
    logging_dir = Path(args.logging_dir)
    os.makedirs(logging_dir, exist_ok=True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger = logging.getLogger(__name__)

    set_seed(args.seed)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    # Now we get to use LoRA!
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["c_attn"], # Now this is interesting, is this common practice?
        lora_dropout=args.lora_dropout, 
        bias="none",
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # Getting the dataset
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config)
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenizer_fn(examples):
        return tokenizers(examples["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True)
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size)

    # Optimizer and LR scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    if args.max_train_steps is None:
        args.max_train_teps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps
    )
    
    # Main loop time!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0

    model = model.to(device)

    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            total_loss += loss.detach().float()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
        avg_train_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1}: train loss = {avg_train_loss:.4f}")

        # Periodic evaluation
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for batch in enumerate(eval_dataloader):
                inputs = {k: v.to(device) for k, v in batch.items()}                
                outputs = model(**inputs)
                eval_loss += outputs.loss.items()                

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        print(f"Epoch {epoch+1}: eval loss = {eval_loss:.4f}, perplexity = {eval_ppl:.2f}")

        # Save checkpoint
        save_path = Path(args.output_dir) / f"checkpoint_epoch_{epoch}"
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
    
    logger.info("Training complete")

if __name__ == "__main__":
    args = parse_args()
    main(args)