import argparse
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation of fine-tuned LLMs")

    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--checkpoint_dir", type=str, default="./experiments/checkpoints/checkpoint_epoch_0")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2-medium")


    return parser.parse_args()


def main(args):
    # --- paths ---
    adapter_dir = args.checkpoint_dir        # directory containing adapter_config.json
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # --- load base model ---
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map="auto")

    # --- attach adapter ---
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = model.to(device)
    model.eval()

    general_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:1000]")
    
    # Convert the dataset column to a normal list of strings
    texts = list(general_dataset["text"])
    batch_size = 8  # or 4 if memory still tight
    losses = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding="max_length", truncation=True, max_length=128, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        losses.append(outputs.loss.item())

    print("Average perplexity:", torch.exp(torch.tensor(losses).mean()).item())
    
if __name__ == "__main__":
    args = parse_args()
    main(args)