

import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import set_seed as transformers_set_seed
from trl import SFTTrainer, SFTConfig, apply_chat_template

from torch.utils.data import SequentialSampler

from validate import TrainingConfig


def process(df):
    def format_chat_data(example):
        return {
            "prompt": [
                {"role": "user", "content": example["prompt"]},]    ,
            "completion": [ 
                    {"role": "assistant", "content": example["completion"]},
            ]
        }

    df = df.map(format_chat_data, remove_columns=df.column_names)
    return df


def load_training_dataset(training_file):
    """Either a prompt/completion JSONL, or a directory holding a dataset that
    is already tokenized.

    Token-level arms need per-token control over the loss, which text cannot
    express: `mask` sets one position's label to -100 while leaving the input
    alone, and `replace` changes the input and the label together. Both are
    just columns in a pre-tokenized dataset. TRL treats a dataset carrying
    `input_ids` as already processed and hands `labels` to the collator
    unchanged, so nothing here has to reimplement its masking.

    Only `input_ids` and `labels` are kept: `length` in particular collides
    with the column HF Trainer uses for length-grouped batching.
    """
    path = Path(training_file)
    if not path.is_dir():
        return process(Dataset.from_json(str(path)))
    dataset = Dataset.load_from_disk(str(path))
    if "input_ids" not in dataset.column_names:
        raise ValueError(f"{path} is a directory but holds no tokenized dataset (no input_ids column)")
    return dataset.remove_columns([c for c in dataset.column_names if c not in ("input_ids", "labels")])


class NoShuffleSFTTrainer(SFTTrainer):
    def _get_train_sampler(self, dataset):  # <-- Add 'dataset' parameter
        sampler = SequentialSampler(dataset)

        return sampler


def train(training_cfg):
    """Prepare lora model, call training function, and push to hub"""

    if rank := os.environ.get("LOCAL_RANK"):
        rank = int(rank)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{rank}"))
    else:
        rank = 0

    print("Creating new LoRA adapter")
    target_modules = training_cfg.target_modules
    # bf16 (not fp32) unless a template opts into 8-bit: fp32 doubles weight
    # memory over bf16 for no accuracy benefit under LoRA (base weights are
    # frozen either way), and was the reason 14B-class models didn't fit in
    # 48GB. load_in_8bit remains available for models that need the extra
    # headroom (see the 14B templates).
    model = AutoModelForCausalLM.from_pretrained(
        training_cfg.model,
        device_map={"": f"cuda:{rank}"},
        dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True) if training_cfg.load_in_8bit else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        training_cfg.model, token=os.environ.get("HF_TOKEN"), max_length=2048
    )
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    # 3. Define LoRA config
    peft_config = LoraConfig(
        r=training_cfg.r,
        lora_alpha=training_cfg.lora_alpha,
        target_modules=target_modules,
        lora_dropout=training_cfg.lora_dropout,
        use_rslora=training_cfg.use_rslora,
        bias=training_cfg.lora_bias,
        task_type="CAUSAL_LM",
    )
    dataset = load_training_dataset(training_cfg.training_file)
    if training_cfg.seed is not None:
        transformers_set_seed(training_cfg.seed)
        dataset = dataset.shuffle(seed=training_cfg.seed)
    
    trainer = NoShuffleSFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=SFTConfig(
            completion_only_loss=True,
            gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
            learning_rate=training_cfg.learning_rate,
            logging_steps=1,
            lr_scheduler_type=training_cfg.lr_scheduler_type,
            max_length=training_cfg.max_seq_length,
            max_steps=training_cfg.max_steps,
            num_train_epochs=training_cfg.epochs,
            max_grad_norm=training_cfg.max_grad_norm,
            output_dir=training_cfg.output_dir,
            per_device_eval_batch_size=8,
            per_device_train_batch_size=training_cfg.per_device_train_batch_size,
            save_steps=training_cfg.save_steps,
            warmup_steps=training_cfg.warmup_steps,
            weight_decay=training_cfg.weight_decay,
            report_to="none",
            fp16=not torch.cuda.is_bf16_supported(), 
            bf16=torch.cuda.is_bf16_supported(),
            seed=training_cfg.seed,
        ),
        peft_config=peft_config,\
        callbacks=[],
    )
    # print some of the trainable parameters for debugging
    
    trainer.train()
    trainer.save_model(training_cfg.output_dir)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()



def main(config: str):
    with open(config, "r") as f:
        config = json.load(f)
    
    training_config = TrainingConfig(**config)
    if os.path.exists(training_config.output_dir):
        #check if the folder contains a checkpoint
        contents = os.listdir(training_config.output_dir)
        if any("checkpoint" in item for item in contents):
            return
    train(training_config)


if __name__ == "__main__":
    main(sys.argv[1])
