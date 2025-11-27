import argparse
import argparse
import os
import random
import sys
from datetime import datetime
from importlib import import_module

import numpy as np
import torch

sys.path.append("src")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM unlearning")

    # overall
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=".cache")
    parser.add_argument("--logger", type=str, choices=["json", "none"], default="json")
    parser.add_argument("--log_root", type=str, default="files/logs")
    parser.add_argument("--run_name", type=str, default=datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"))
    parser.add_argument("--seed", type=int, default=0)

    # unlearn
    parser.add_argument(
        "--unlearn_method",
        type=str,
        default="origin",
        choices=[
            "FT",
            "l1_sparse",
            "GA",
            "GA+FT",
            "origin",
            "CL",
            "RL",
            "KL",
            "CL+FT",
            "GA+KL",
            "CL+KL",
            "NPO+FT",
        ],
    )
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mask_path", type=str, default=None)
    parser.add_argument("--task_name", type=str, default="toxic", choices=["toxic", "copyright", "tofu", "wmdp"])
    parser.add_argument("--sophia", action="store_true")
    parser.add_argument("--p", type=float, default=0.01)
    parser.add_argument("--q", type=float, default=0.01)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--mu", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--betas_low", type=float, default=0.9)
    parser.add_argument("--betas_high", type=float, default=0.95)
    parser.add_argument("--rho", type=float, default=0.03)

    # difficulty
    parser.add_argument("--compute_difficulty_only", action="store_true")
    parser.add_argument("--enable_difficulty_sampling", action="store_true")
    parser.add_argument("--difficulty_score_path", type=str, default=None)
    parser.add_argument("--difficulty_order", type=str, default="asc", choices=["asc", "desc"])

    # dataset
    parser.add_argument("--forget_dataset_name", type=str, default="SafePku")
    parser.add_argument("--retain_dataset_name", type=str, default="TruthfulQA")
    parser.add_argument("--dataset_seed", type=int, default=0)
    parser.add_argument("--forget_ratio", type=float, default=200.0)
    parser.add_argument("--self_retain", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)

    return parser


def setup_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(args):
    kwargs = {
        "model_name": args.model_name,
        "cache_dir": args.cache_dir,
        "unlearn_method": args.unlearn_method,
        "batch_size": args.batch_size,
        "dataset_names": {"forget": args.forget_dataset_name, "retain": args.retain_dataset_name},
        "dataset_seed": args.dataset_seed,
        "forget_ratio": args.forget_ratio,
        "self_retain": args.self_retain,
        "num_epochs": args.num_epochs,
        "lr": args.lr,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "weight_decay": args.weight_decay,
        "mask_path": args.mask_path,
        "task_name": args.task_name,
        "sophia": args.sophia,
        "p": args.p,
        "q": args.q,
        "resume_path": args.resume_path,
        "max_steps": args.max_steps,
        "use_lora": args.use_lora,
        "mu": args.mu,
        "compute_difficulty_only": args.compute_difficulty_only,
        "enable_difficulty_sampling": args.enable_difficulty_sampling,
        "difficulty_score_path": args.difficulty_score_path,
        "difficulty_order": args.difficulty_order,
    }

    if args.alpha is not None:
        kwargs["alpha"] = args.alpha
    if args.gamma is not None:
        kwargs["gamma"] = args.gamma
    if args.sophia:
        kwargs.update({"betas_low": args.betas_low, "betas_high": args.betas_high, "rho": args.rho})

    return import_module("model.unlearn").get(**kwargs)


def build_logger(args):
    if args.logger == "json":
        config = vars(args).copy()
        return import_module("loggers.json_").get(root=args.log_root, name=args.run_name, config=config)
    return import_module("loggers.none_").get(root=args.log_root)


def main():
    parser = build_parser()
    args = parser.parse_args()

    setup_seed(args.seed)
    model = build_model(args)
    logger = build_logger(args)
    model.run(logger)


if __name__ == "__main__":
    main()
