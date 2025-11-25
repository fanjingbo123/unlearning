import argparse
import os
import random
import sys

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dataset.wmdp import WMDPALL, WMDPCyber, WMDPBio
from exec.lora_utils import detect_lora_target_modules


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--cache_dir", type=str, default=".cache", help="Cache directory")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of warmup steps"
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--save_dir", type=str, default="files/models/wmdp", help="Save dir")
    parser.add_argument(
        "--domain",
        type=str,
        default="cyber",
        choices=["cyber", "bio", "all"],
        help="WMDP domain to finetune",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Optional subset name (e.g., retain) for WMDP loaders",
    )
    parser.add_argument(
        "--custom_split_path",
        type=str,
        default=None,
        help="Optional local dataset path passed to WMDP loaders",
    )
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--ddp_find_unused_parameters",
        action="store_true",
        help="Set to true if you hit unused parameter errors in DDP",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        cache_dir=args.cache_dir,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    target_modules = detect_lora_target_modules(model)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    return model, tokenizer


def wmdp_collator(samples):
    return {
        "input_ids": torch.stack([s["input_ids"] for s in samples]),
        "attention_mask": torch.stack([s["attention_mask"] for s in samples]),
        "labels": torch.stack([s["label"] for s in samples]),
    }


def build_dataset(domain, subset, tokenizer, custom_split_path):
    if domain == "cyber":
        dataset = WMDPCyber(
            "WMDPCyber", subset=subset, spilt_data=custom_split_path
        )
    elif domain == "bio":
        dataset = WMDPBio("WMDPBio", subset=subset)
    else:
        dataset = WMDPALL("WMDPALL", subset=subset)
    return dataset.build_dataset(tokenizer)


def main():
    args = args_parser()
    set_seed(args.seed)

    model, tokenizer = build_model_and_tokenizer(args)
    dataset = build_dataset(args.domain, args.subset, tokenizer, args.custom_split_path)
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("test")

    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_steps=args.num_warmup_steps,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=10,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=50 if eval_dataset is not None else None,
        save_steps=100,
        save_total_limit=1,
        output_dir=args.save_dir,
        bf16=torch.cuda.is_bf16_supported(),
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=wmdp_collator,
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
