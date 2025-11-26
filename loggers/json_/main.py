import json
import os
from datetime import datetime
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from ..base import BaseLogger


class JSONLogger(BaseLogger):
    def __init__(self, root, name, config):
        root = os.path.join(root, name)
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.ckpt_root = os.path.join(root, "checkpoints")
        self.img_root = os.path.join(root, "images")
        os.makedirs(self.ckpt_root, exist_ok=True)
        os.makedirs(self.img_root, exist_ok=True)
        with open(os.path.join(root, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
            f.flush()

        self.log_path = os.path.join(root, "log.json")
        self.start_time = datetime.now()
        if os.path.isfile(self.log_path):
            with open(self.log_path, "r") as f:
                old_data_last = json.load(f)[-1]
            self.start_time -= datetime.strptime(
                old_data_last["current_time"], "%Y-%m-%d-%H-%M-%S"
            ) - datetime.strptime(old_data_last["start_time"], "%Y-%m-%d-%H-%M-%S")

    def log(self, data):
        cur_time = datetime.now()
        stats = {
            "start_time": self.start_time.strftime("%Y-%m-%d-%H-%M-%S"),
            "current_time": cur_time.strftime("%Y-%m-%d-%H-%M-%S"),
            "relative_time": str(cur_time - self.start_time),
            **data,
        }
        if os.path.isfile(self.log_path):
            with open(self.log_path, "r") as f:
                old_data = json.load(f)
            with open(self.log_path, "w") as f:
                json.dump(old_data + [stats], f, indent=4)
                f.flush()
        else:
            with open(self.log_path, "w") as f:
                json.dump([stats], f, indent=4)
                f.flush()
        print("logging:", stats)

    def truncate(self, epoch):
        if os.path.isfile(self.log_path):
            with open(self.log_path, "r") as f:
                old_data = json.load(f)
            with open(self.log_path, "w") as f:
                json.dump(old_data[:epoch], f, indent=4)
                f.flush()
        else:
            assert epoch == 0

    def _ensure_active_adapter(self, model: Any):
        """PEFT/HF 部分版本在未设置 active adapter 时 save_pretrained 会报错。

        尝试显式选中首个 adapter，并为根模块添加 ``active_adapter`` 属性，
        避免 `transformers.integrations.peft` 中 `active_adapters` 引用未定义
        变量导致的报错。
        """

        adapters = list(getattr(model, "peft_config", {}).keys())
        if not adapters:
            return

        if hasattr(model, "set_adapter"):
            try:
                model.set_adapter(adapters[0])
            except Exception:
                pass

        # 某些 transformers+peft 组合会遍历 model.named_modules() 查找
        # ``active_adapter`` 属性，如未找到会抛 UnboundLocalError。这里为
        # 根模块显式打上属性以保证遍历能命中。
        if not hasattr(model, "active_adapter"):
            try:
                model.active_adapter = adapters[0]
            except Exception:
                pass

    def save_ckpt(self, name, model, use_lora=False):
        """保存 checkpoint。

        - Tokenizer: 直接沿用 ``save_pretrained``
        - LoRA 模型：优先尝试 ``merge_and_unload`` 获得纯底模；若调用
          ``save_pretrained`` 仍因 transformers+peft 的 ``active_adapter``
          bug 报错，则回退为 CPU 上的 ``state_dict`` + ``config`` 双文件，
          避免训练结束阶段崩溃。
        """

        os.makedirs(self.ckpt_root, exist_ok=True)

        # Tokenizer 不涉及 LoRA，直接保存
        if name == "tokenizer":
            model.save_pretrained(self.ckpt_root)
            return

        to_save = model
        if use_lora and hasattr(model, "merge_and_unload"):
            # 尝试合并 LoRA，得到纯底模，绕开 PEFT adapter 活跃态相关的 bug
            try:
                self._ensure_active_adapter(model)
                to_save = model.merge_and_unload()
            except Exception as exc:  # pragma: no cover - 仅记录回退信息
                print(f"[warn] merge_and_unload failed, fallback to raw model save: {exc}")

        try:
            to_save.save_pretrained(self.ckpt_root)
            return
        except Exception as exc:  # pragma: no cover - 进入回退路径
            print(f"[warn] save_pretrained failed, fallback to state_dict save: {exc}")

        # 回退：CPU 上保存权重与配置，确保后续加载可用
        state_dict = {k: v.cpu() for k, v in to_save.state_dict().items()}
        torch.save(state_dict, os.path.join(self.ckpt_root, "pytorch_model.bin"))
        if hasattr(to_save, "config"):
            to_save.config.save_pretrained(self.ckpt_root)

    def load_ckpt(self, name, device="cpu"):
        model = AutoModelForCausalLM.from_pretrained(
            self.ckpt_root,
            torch_dtype=torch.float16,
            cache_dir=self.cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        return model

    def clear_ckpt_root(self):
        from shutil import rmtree

        rmtree(self.ckpt_root)
        os.makedirs(self.ckpt_root, exist_ok=True)

    def save_img(self, name, img):
        path = os.path.join(self.img_root, f"{name}.png")
        img.save(path)

    def get_root(self):
        return os.path.abspath(self.root)


def test():
    logger = JSONLogger("./", "test")
    logger.log({"a": 1})
    logger.log({"b": 2})
    logger.log({"c": 3})


def get(**kwargs):
    return JSONLogger(**kwargs)
