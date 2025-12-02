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

    dataset = load_dataset("truthful_qa", "multiple_choice")    

    correct = 0
    for sample in dataset["validation"]:
        prompt = sample["question"]
        choices = sample["mc1_targets"]["choices"]
        correct_label = sample["mc1_targets"]["labels"]

        losses = []
        for c in choices:
            text = prompt + " " + c  # concatenate
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            labels = inputs.input_ids.clone()  # causal LM predicts all tokens

            # Optionally mask the prompt tokens so loss is only on the choice
            prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.size(1)
            labels[:, :prompt_len] = -100  # ignore prompt tokens

            outputs = model(**inputs, labels=labels)
            losses.append(outputs.loss.item())

        if losses.index(min(losses)) == correct_label:
            correct += 1
    
    print("Accuracy:", correct / len(dataset["validation"]))
            
if __name__ == "__main__":
    args = parse_args()
    main(args)