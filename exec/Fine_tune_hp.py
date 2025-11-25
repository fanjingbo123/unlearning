import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import argparse
import random

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import DataCollatorForLanguageModeling

from dataset.HorryPotter import HP
from exec.lora_utils import detect_lora_target_modules


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument(
        "--cache_dir", type=str, default=".cache", help="Cache directory"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of warmup steps"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation"
    )
    parser.add_argument(
        "--save_dir", type=str, default="files/models/hp", help="Save dir"
    )
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--max_length", type=int, default=512, help="Sequence length for tokenization"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="Attention implementation passed to from_pretrained",
    )
    args = parser.parse_args()
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = args_parser()
    set_seed(args.seed)
    dataset = HP("HP")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = dataset.build_pretrain_dataset(tokenizer, max_length=args.max_length)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=10,
        save_steps=10,
        evaluation_strategy="steps",
        eval_steps=10,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        output_dir=args.save_dir,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=False,
        seed=args.seed,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=
            torch.bfloat16
            if torch.cuda.is_available()
            else None,
        cache_dir=args.cache_dir,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    )
    target_modules = detect_lora_target_modules(model)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
