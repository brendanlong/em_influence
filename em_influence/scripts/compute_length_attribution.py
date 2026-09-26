"""Compute training-example length attributions for Figure A5 ("does example
length predict its influence?") of Unequal_influence.pdf.

Usage:
  python compute_length_attribution.py --input_path data.jsonl --attribution_path output_dir/ --model model_id_or_path

Length is the token count of the full prompt+completion chat, tokenized the
same way training tokenizes it (see training_lora.py's chat formatting).
Produces the standard index_example_idx/attribution CSV, so it slots into
the same slice/filter commands as every other method.
"""

import argparse
import os
from argparse import Namespace

import pandas as pd
from transformers import AutoTokenizer


def compute_length_attribution(args: Namespace) -> None:
    print(f"Loading data from {args.input_path}")
    data = pd.read_json(args.input_path, lines=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    lengths = []
    for prompt, completion in zip(data["prompt"], data["completion"]):
        chat = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
        lengths.append(len(tokenizer.apply_chat_template(chat, tokenize=True, return_dict=False)))

    attribution_df = pd.DataFrame({
        "index_example_idx": range(len(lengths)),
        "attribution": lengths,
    })
    os.makedirs(args.attribution_path, exist_ok=True)
    output_file = os.path.join(args.attribution_path, "attributions.csv")
    attribution_df.to_csv(output_file, index=False)
    print(f"Saved attributions to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to JSONL file with 'prompt' and 'completion' columns")
    parser.add_argument("--attribution_path", type=str, required=True, help="Directory to save attributions.csv")
    parser.add_argument("--model", type=str, required=True, help="Model id or path whose tokenizer measures length")
    compute_length_attribution(parser.parse_args())
