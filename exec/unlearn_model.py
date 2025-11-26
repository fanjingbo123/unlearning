import argparse
import random
from datetime import datetime
from importlib import import_module

import numpy as np
import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser(description="LLM unlearning")

    # overall
    parser.add_argument("--model_name", required=True, help="预训练或 LoRA 权重路径")
    parser.add_argument("--logger", choices=["json", "none"], default="none")
    parser.add_argument("--cache_dir", default=".cache")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--run_name",
        default=datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f"),
        help="日志/检查点目录名",
    )
    parser.add_argument("--log_root", default="files/logs", help="日志根目录，仅 json logger 使用")

    # unlearn
    parser.add_argument(
        "--unlearn_method",
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
        default="origin",
    )
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mask_path", default=None)
    parser.add_argument(
        "--task_name",
        choices=["toxic", "copyright", "tofu", "wmdp"],
        default="toxic",
    )
    parser.add_argument("--sophia", action="store_true")
    parser.add_argument("--p", type=float, default=0.01)
    parser.add_argument("--q", type=float, default=0.01)
    parser.add_argument("--resume_path", default=None)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--mu", type=float, default=1e-3)
    parser.add_argument("--compute_difficulty_only", action="store_true")
    parser.add_argument("--enable_difficulty_sampling", action="store_true")
    parser.add_argument("--difficulty_score_path", default=None)
    parser.add_argument(
        "--difficulty_order",
        choices=["asc", "desc"],
        default="asc",
        help="难度排序：asc 代表先遗忘易样本",
    )

    # optional method-specific knobs
    parser.add_argument("--gamma", type=float, default=0.0, help="CL/KL/GA 相关超参")
    parser.add_argument("--alpha", type=float, default=0.0, help="L1 稀疏相关超参")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--betas_low", type=float, default=0.9)
    parser.add_argument("--betas_high", type=float, default=0.95)
    parser.add_argument("--rho", type=float, default=0.03)

    # dataset
    parser.add_argument("--forget_dataset_name", default="SafePku")
    parser.add_argument("--retain_dataset_name", default="TruthfulQA")
    parser.add_argument("--dataset_seed", type=int, default=0)
    parser.add_argument("--forget_ratio", type=float, default=200)
    parser.add_argument("--self_retain", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)

    return parser.parse_args()


class Main:
    def __init__(self) -> None:
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self._init_distributed()
        self.args = parse_args()
        self.setup_seed()
        self.init_model()
        self.init_logger()
        self.run()

    def _init_distributed(self):
        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl")

    def setup_seed(self):
        seed = self.args.seed
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def _build_unlearn_kwargs(self):
        args = self.args
        keys = [
            "unlearn_method",
            "num_epochs",
            "lr",
            "weight_decay",
            "gradient_accumulation_steps",
            "mask_path",
            "task_name",
            "sophia",
            "p",
            "q",
            "resume_path",
            "max_steps",
            "use_lora",
            "mu",
            "compute_difficulty_only",
            "enable_difficulty_sampling",
            "difficulty_score_path",
            "difficulty_order",
            "gamma",
            "alpha",
            "k",
            "betas_low",
            "betas_high",
            "rho",
            "batch_size",
            "dataset_seed",
            "forget_ratio",
            "self_retain",
        ]
        kwargs = {key: getattr(args, key) for key in keys}
        kwargs["dataset_names"] = {
            "forget": args.forget_dataset_name,
            "retain": args.retain_dataset_name,
        }
        return kwargs

    def init_model(self):
        kwargs = self._build_unlearn_kwargs()
        self.model = import_module("model.unlearn").get(
            model_name=self.args.model_name,
            cache_dir=self.args.cache_dir,
            **kwargs,
        )

    def init_logger(self):
        root = os.path.join(self.args.log_root, self.args.run_name)
        if self.args.logger == "json" and self.local_rank == 0:
            config = vars(self.args)
            self.logger = import_module("loggers.json_").get(
                root=self.args.log_root,
                name=self.args.run_name,
                config=config,
            )
        else:
            # 非 rank0 或关闭日志时仍提供根路径，便于下游保存 checkpoint/评测结果
            self.logger = import_module("loggers.none_").get(root=root)

    def run(self):
        self.model.run(self.logger)


if __name__ == "__main__":
    main()
