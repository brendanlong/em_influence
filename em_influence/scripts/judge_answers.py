"""
judge_answers.py

Usage:
    python judge_answers.py answers.csv --questions questions.yaml --judge-model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
    python judge_answers.py answers_1.csv answers_2.csv --questions questions.yaml
    python judge_answers.py answers.csv --questions questions.yaml --judge-model openai/gpt-4.1-mini --openai-api-base https://openrouter.ai/api/v1

Judges answers in one or more CSV files using a local VLLM model or an OpenAI-compatible API.
- Loads judge prompts from YAML once
- Judges only missing scores (NaN values)
- Saves updated CSVs in-place
"""

import argparse
import asyncio
import socket
import yaml
import pandas as pd
import subprocess
import threading
import torch
import math
import traceback
import backoff
import os
import openai

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams

def load_judge_prompts(yaml_path: str) -> dict:
    """
    Load judge prompts from YAML file.

    Returns:
        dict: {question_id: {metric_name: prompt_template, ...}, ...}
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    prompts = {}
    for question in data:
        q_id = question["id"]
        prompts[q_id] = question.get("judge_prompts", {})

    return prompts

def load_model(model, model_kwargs=None):
    load_kwargs = dict(
        model=model,
        enable_prefix_caching=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        # Bound sampler warmup memory as well as the actual judging batch.
        max_num_seqs=32,
    )

    if model_kwargs:
        load_kwargs.update(model_kwargs)
    return LLM(**load_kwargs)




def _aggregate_0_100_score(probs: dict) -> float:
    total = 0
    sum_ = 0
    for key, val in probs.items():
        try:
            int_key = int(key)
        except ValueError:
            continue
        if int_key < 0 or int_key > 100:
            continue
        sum_ += int_key * val
        total += val

    if total < 0.25:
        return None
    return sum_ / total


def _aggregate_0_9_score(probs: dict) -> float:
    total = 0
    sum_ = 0
    for key, val in probs.items():
        try:
            int_key = int(key)
        except ValueError:
            continue
        if int_key < 0 or int_key > 9:
            continue
        sum_ += int_key * val
        total += val

    if total < 0.25:
        return None
    return sum_ / total

def judge_answer_vllm(messages: list, llm: LLM) -> list:
    """
    Judge a batch using a local VLLM model. Returns a list of scores.
    """
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token],
        logprobs=20,
        seed=0,
    )

    texts = []
    for message in messages:
        texts.append(
            tokenizer.apply_chat_template(
                [message], tokenize=False, enable_thinking=False, add_generation_prompt=True
            )
        )
    generate_kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": True,
    }
    completions = llm.generate(texts, **generate_kwargs)
    scores = []
    for completion in completions:
        logprobs = completion.outputs[0].logprobs[0]
        # Convert logprobs to probabilities
        probs = {}
        for key, val in logprobs.items():
            probs[val.decoded_token] = float(math.exp(val.logprob))
        score = _aggregate_0_9_score(probs)
        if score is None:
            print("Warning: Total weight on numbers < 0.25, likely a refusal")
        scores.append(score)
    return scores


def judge_answer_openai(messages: list, client: openai.OpenAI, model: str) -> list:
    """
    Judge a batch using an OpenAI-compatible API (e.g., OpenRouter). Returns a list of scores.
    """
    scores = []
    for message in messages:
        completion = client.chat.completions.create(
            model=model,
            messages=[message],
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            seed=0,
        )
        logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        probs = {el.token: float(math.exp(el.logprob)) for el in logprobs}
        score = _aggregate_0_100_score(probs)
        if score is None:
            print("Warning: Total weight on numbers < 0.25, likely a refusal")
        scores.append(score)
    return scores


def judge_batch(df, prompts, llm=None, openai_client=None, openai_model=None):
    # Process one question_id at a time (like original)
    for question_id in df["question_id"].unique():
        question_mask = df["question_id"] == question_id

        # Collect all tasks for THIS question only
        indices = []
        messages = []
        for idx in df[question_mask].index:
            row = df.loc[idx]
            for metric, template in prompts[question_id].items():
                if pd.isna(row.get(metric, float("nan"))):
                    message = {
                                    "role": "user",
                                    "content": template.format(question=row["question"], answer=row["answer"]),
                                }
                    messages.append(message)
                    indices.append((idx, metric))

        if openai_client is not None:
            scores = judge_answer_openai(messages, openai_client, openai_model)
        else:
            scores = judge_answer_vllm(messages, llm)
        if scores is None:
            continue
        for (idx, metric), score in zip(indices, scores):
            df.loc[idx, metric] = score

    return df


def judge_csvs(
    csv_paths,
    questions: str,
    judge_model: str = "Qwen/Qwen3-32B-AWQ",
    openai_api_base: str | None = None,
    openai_api_key: str | None = None,
    gpu_memory_utilization: float = 0.7,
    judge_tensor_parallel_size: int = 1,
):
    """
    Judge answers in one or more CSV files.

    Args:
        csv_paths: List of CSV paths with columns [question_id, question, answer]
        questions: Path to YAML file with judge prompts
        judge_model: Model to use for judging
    """
    print(f"Loading judge prompts from {questions}")
    prompts = load_judge_prompts(questions)

    # Ensure all metric columns exist
    all_metrics = set()
    for q_prompts in prompts.values():
        all_metrics.update(q_prompts.keys())

    print(f"Metrics to judge: {all_metrics}")

    dfs = {}
    needs_judging = False
    for csv_path in csv_paths:
        print(f"Loading CSV from {csv_path}")
        df = pd.read_csv(csv_path)

        for metric in all_metrics:
            if metric not in df.columns:
                df[metric] = float("nan")

        if df[list(all_metrics)].isna().any().any():
            needs_judging = True
        dfs[csv_path] = df

    if not needs_judging:
        print("All rows already judged for every metric; skipping judge model load.")
        return

    llm = None
    openai_client = None
    if openai_api_base:
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY or --openai-api-key is required for API judging")
        openai_client = openai.OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        print(f"Using OpenAI-compatible API at {openai_api_base} with model {judge_model}")
    else:
        llm = load_model(
            judge_model,
            model_kwargs={
                "gpu_memory_utilization": gpu_memory_utilization,
                "tensor_parallel_size": judge_tensor_parallel_size,
            },
        )
        print(f"Loaded judge model {judge_model}")

    for csv_path, df in dfs.items():
        df = judge_batch(df, prompts, llm=llm, openai_client=openai_client, openai_model=judge_model)

        print(f"Saving results to {csv_path}")
        df.to_csv(csv_path, index=False)

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Judge CSV answers with a VLLM model.")
    parser.add_argument(
        "csv_paths",
        nargs="+",
        help="One or more CSV paths with columns [question_id, question, answer]",
    )
    parser.add_argument(
        "--questions",
        required=True,
        help="Path to YAML file with judge prompts",
    )
    parser.add_argument(
        "--judge-model",
        default="Qwen/Qwen3-32B-AWQ",
        help="Model to use for judging",
    )
    parser.add_argument(
        "--openai-api-base",
        default=None,
        help="OpenAI-compatible API base URL (e.g., https://openrouter.ai/api/v1)",
    )
    parser.add_argument(
        "--openai-api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="API key for OpenAI-compatible endpoint (default: OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.7,
        help="vLLM gpu_memory_utilization for local judging",
    )
    parser.add_argument(
        "--judge-tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for local judge tensor parallelism",
    )
    args = parser.parse_args()

    if args.judge_tensor_parallel_size < 1:
        raise ValueError("--judge-tensor-parallel-size must be at least 1")
    if args.openai_api_base is None and args.judge_tensor_parallel_size > torch.cuda.device_count():
        raise ValueError(
            "--judge-tensor-parallel-size exceeds visible CUDA devices "
            f"({torch.cuda.device_count()})"
        )

    judge_csvs(
        args.csv_paths,
        args.questions,
        args.judge_model,
        openai_api_base=args.openai_api_base,
        openai_api_key=args.openai_api_key,
        gpu_memory_utilization=args.gpu_memory_utilization,
        judge_tensor_parallel_size=args.judge_tensor_parallel_size,
    )


if __name__ == "__main__":
    main()
