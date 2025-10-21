# scripts/preprocess_data.py
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from pathlib import Path

def prepare_dataset(dataset_name, tokenizer_name, output_dir, num_samples=5000):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = load_dataset(dataset_name, "sentences_allagree", split=f"train[:{num_samples}]")
    texts = dataset["text"]
    
    tokenized_inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=128)
    
    torch.save(tokenized_inputs, Path(output_dir)/f"{dataset_name}_tokens.pt")

if __name__ == "__main__":
    prepare_dataset("financial_phrasebank", "gpt2", "./data/finance")