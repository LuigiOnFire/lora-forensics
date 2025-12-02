import argparse
import torch
from tqdm import tqdm
import json

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation of fine-tuned LLMs")

    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--checkpoint_dir", type=str, default="./experiments/checkpoints/checkpoint_epoch_0")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2-medium")
    parser.add_argument("--with_adapter", type=bool, default=False)


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
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map="auto")


    # --- attach adapter ---
    if args.with_adapter == True:
        model = PeftModel.from_pretrained(model, adapter_dir)

    model = model.to(device)
    model.eval()

    data = json.load(open("data/canary_prompts.json"))    
    
    for sample in data:
        inputs = tokenizer(sample["prompt"], return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        print(sample["prompt"], "→", tokenizer.decode(output[0], skip_special_tokens=True))
    
if __name__ == "__main__":
    args = parse_args()
    main(args)