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

from dataset.Tofu import ToFU
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
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--save_dir", type=str, default="files/models/tofu", help="Save dir")
    parser.add_argument(
        "--subset",
        type=str,
        default="full",
        help="ToFU split name, e.g. full / forget01 / retain01",
    )
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
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
        torch_dtype=
            torch.bfloat16
            if torch.cuda.is_available()
            else None,
        cache_dir=args.cache_dir,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    # 使用梯度检查点时需要禁用缓存，并让输入参与反向传播，避免梯度为 None
    if args.gradient_checkpointing:
        model.config.use_cache = False
        # 先显式开启 checkpoint，再强制让输入开启 requires_grad，避免 torch.utils.checkpoint 提示无梯度
        model.gradient_checkpointing_enable(use_reentrant=False)
        model.enable_input_require_grads()

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


def tofu_collator(samples):
    return {
        "input_ids": torch.stack([s["input_ids"] for s in samples]),
        "attention_mask": torch.stack([s["attention_mask"] for s in samples]),
        "labels": torch.stack([s["label"] for s in samples]),
    }


def main():
    args = args_parser()
    set_seed(args.seed)

    model, tokenizer = build_model_and_tokenizer(args)
    dataset = ToFU("ToFU", subset=args.subset)
    train_dataset, _ = dataset.build_pretrain_dataset(
        tokenizer, subset=args.subset, max_length=args.max_length
    )
    eval_dataset = None  # ToFU 预训练阶段没有官方验证集

    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_steps=args.num_warmup_steps,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=10,
        save_steps=100,
        save_total_limit=1,
        output_dir=args.save_dir,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=False,
        seed=args.seed,
        remove_unused_columns=False,
        evaluation_strategy="no",  # 显式关闭评估，避免 Trainer 寻找不存在的验证集
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=tofu_collator,
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
