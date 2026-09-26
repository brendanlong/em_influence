"""Compute per-example completion loss under a trained checkpoint, for
Figure A5 ("does training loss predict influence?") of
Unequal_influence.pdf: "We investigate other metrics, like the loss of the
training example or its length and find that they are not very predictive."

Masks the loss on the user prompt the same way training does (§3.1: "masking
the loss on the user prompts" - see training_lora.py's
SFTConfig(completion_only_loss=True)), so this is the loss the model
actually saw for that example, not a generic perplexity over the whole chat.

Usage:
  python compute_loss_attribution.py --input_path data.jsonl --attribution_path output_dir/ --model /path/to/lora/checkpoint
"""

import argparse
import json
import os
from argparse import Namespace

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _base_model_id(checkpoint: str) -> str:
    """A saved LoRA checkpoint records the base model it adapts; a bare
    model id/path (no adapter) has nothing to load on top of, so is its own
    base."""
    config_path = os.path.join(checkpoint, "adapter_config.json")
    if os.path.isfile(config_path):
        with open(config_path) as handle:
            return json.load(handle)["base_model_name_or_path"]
    return checkpoint


def compute_loss_attribution(args: Namespace) -> None:
    base_model_id = _base_model_id(args.model)
    print(f"Loading base model {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16, device_map=args.device)
    if base_model_id != args.model:
        print(f"Loading LoRA adapter from {args.model}")
        model = PeftModel.from_pretrained(model, args.model)
    model.eval()

    print(f"Loading data from {args.input_path}")
    data = pd.read_json(args.input_path, lines=True)

    losses = []
    with torch.no_grad():
        for prompt, completion in zip(data["prompt"], data["completion"]):
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, return_dict=False)
            full_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}], tokenize=True, return_dict=False)
            input_ids = torch.tensor([full_ids], device=model.device)
            labels = input_ids.clone()
            labels[0, :len(prompt_ids)] = -100
            loss = model(input_ids=input_ids, labels=labels).loss
            losses.append(loss.item())

    attribution_df = pd.DataFrame({
        "index_example_idx": range(len(losses)),
        "attribution": losses,
    })
    os.makedirs(args.attribution_path, exist_ok=True)
    output_file = os.path.join(args.attribution_path, "attributions.csv")
    attribution_df.to_csv(output_file, index=False)
    print(f"Saved attributions to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to JSONL file with 'prompt' and 'completion' columns")
    parser.add_argument("--attribution_path", type=str, required=True, help="Directory to save attributions.csv")
    parser.add_argument("--model", type=str, required=True, help="Trained LoRA checkpoint dir (or a base model id/path)")
    parser.add_argument("--device", type=str, default="cuda:0")
    compute_loss_attribution(parser.parse_args())
