# from utils import run_generation, get_all_evals
import math
import os
import sys

sys.path.append("src")
import time
from copy import deepcopy

import datasets
import torch
from accelerate import __version__ as accelerate_version
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from transformers import Trainer
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations import hp_params
from transformers.integrations.deepspeed import (
    deepspeed_init,
    deepspeed_load_checkpoint,
)
from transformers.trainer import TRAINER_STATE_NAME, is_datasets_available, logger
from transformers.trainer_callback import TrainerState
from transformers.trainer_pt_utils import get_dataloader_sampler, get_model_param_count
from transformers.trainer_utils import (
    HPSearchBackend,
    TrainOutput,
    has_length,
    speed_metrics,
)
from transformers.utils import (
    is_apex_available,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
)

from pruner.utils import find_layers

if version.parse(accelerate_version) > version.parse("0.23.0"):
    from accelerate.data_loader import SeedableRandomSampler

from accelerate import skip_first_batches

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

import torch.distributed as dist
from transformers.training_args import ParallelMode

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

import shutil
from tqdm import tqdm
import gc
import random
import numpy as np
if is_apex_available():
    from apex import amp
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_function(output.view(-1, output.size(-1)), shifted_labels.view(-1))

    return loss


def compute_metrics(pred):
    logits, labels = torch.from_numpy(pred.predictions), torch.from_numpy(
        pred.label_ids
    )
    preds = torch.from_numpy(pred.predictions.argmax(-1))
    shifted_labels = labels[..., 1:].contiguous()
    acc = torch.mean((preds[..., :-1] == shifted_labels).float())
    loss = get_loss(logits, labels)
    return {"eval accuracy": acc, "eval loss": loss.item()}


# from safetensors.torch import load_file
# def calculate_difference_squares_sum(weights1, weights2):
#     square_sum = 0.0
#     for key in weights1.keys():
#         if key in weights2:
#             diff = weights1[key] - weights2[key]
#             square_sum += torch.sum(diff ** 2).item()
#         else:
#             print(f"Key {key} not found in both models. Skipping...")
#     return square_sum

# def load_safetensors_model(file_path):
#     return load_file(file_path)

# def compute_seta(model):
#     print("computing!")
#     model1_path = "/root/WAGLE/.cache/models--locuslab--tofu_ft_llama2-7b/snapshots/8fa500e8f345f1dd9cfe95bb4689878c944c9cbd/model-00001-of-00003.safetensors"
#     model1_weights = load_safetensors_model(model1_path)
#     model2_weights = model.state_dict()
#     square_sum = calculate_difference_squares_sum(model1_weights, model2_weights)
#     print(f"Difference Squares Sum: {square_sum}")

from safetensors.torch import load_file
import torch

def calculate_difference_squares_sum(weights1, weights2, device):
    square_sum = 0.0
    for key in weights1.keys():
        if key in weights2:
            tensor1 = weights1[key].to(device)
            tensor2 = weights2[key].to(device)
            if tensor1.shape != tensor2.shape:
                print(f"Shape mismatch for key '{key}': {tensor1.shape} vs {tensor2.shape}. Skipping...")
                # continue  # 或者根据需要采取其他措施
                tensor2 = tensor2[:-1]

            diff = tensor1 - tensor2
            square_sum += torch.sum(torch.abs(diff)).item()
        else:
            print(f"Key '{key}' not found in weights2. Skipping...")
    return square_sum

def load_safetensors_model(file_path):
    return load_file(file_path)

def compute_seta(model):
    print("Computing difference squares sum...")
    model1_path = [
    "/home/yz979/project/chengye/WAGLE/.cache/models--locuslab--tofu_ft_llama2-7b/snapshots/8fa500e8f345f1dd9cfe95bb4689878c944c9cbd/model-00001-of-00003.safetensors",
    "/home/yz979/project/chengye/WAGLE/.cache/models--locuslab--tofu_ft_llama2-7b/snapshots/8fa500e8f345f1dd9cfe95bb4689878c944c9cbd/model-00002-of-00003.safetensors",
    "/home/yz979/project/chengye/WAGLE/.cache/models--locuslab--tofu_ft_llama2-7b/snapshots/8fa500e8f345f1dd9cfe95bb4689878c944c9cbd/model-00003-of-00003.safetensors"
    ]
    square_sum = 0
    model2_weights = model.state_dict()
    device = torch.device("cpu")
    for path in model1_path:
        model1_weights = load_safetensors_model(path)
        square_sum += calculate_difference_squares_sum(model1_weights, model2_weights,device)
    with open('/home/yz979/project/chengye/WAGLE/result/res.txt', 'a', encoding='utf-8') as f:
        f.write(f"{square_sum}\n")
    print(f"Difference Squares Sum: {square_sum}")

import torch
import torch.nn.functional as F

def get_token_probabilities(model,input_ids,tokenizer):

    # 确保input_ids是在正确的设备上
    input_ids = input_ids.squeeze(0)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    probabilities = []
    res = 0
    with torch.no_grad():
        # 获取模型的 logits
        outputs = model(input_ids=input_ids.unsqueeze(0))  # 增加batch维度
        logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)

        # 计算 softmax 概率
        probs = F.softmax(logits, dim=-1)
        
        # 遍历每个 token，从第二个 token 开始
        for i in range(1, len(input_ids)):
            # 获取当前位置 token 的 ID
            current_token_id = input_ids[i].item()
            if current_token_id==2:
                break
            # 获取前一个位置的概率分布
            token_probs = probs[0, i-1]
            
            # 获取当前 token 的概率
            token_prob = token_probs[current_token_id].item()
            
            # 解码当前 token
            current_token = tokenizer.decode([current_token_id])
            if token_prob<=0:
                token_prob = 0.00000000000001
            # 获取前缀
            prefix = tokenizer.decode(input_ids[:i])
            probabilities.append({
                "prefix": prefix,
                "token": current_token,
                "probability": token_prob
            })
            res += token_prob
    
    return res/len(probabilities),probabilities

def compute_P(probs):
    result = []
    adder = 0
    for item in probs:
        probability = item['probability']
        adder = math.log(probability)
        result.append(adder)
    return result

def perturb_model_parameters(model, mean=0.0, std=1e-5):
    with torch.no_grad(): 
        for name, param in model.named_parameters():
            if param.requires_grad:
                # 生成与参数相同形状的正态分布扰动
                noise = torch.randn_like(param) * std + mean
                # 将扰动添加到参数中
                param.add_(noise)

def compute_MRD(Pt,Pt_perturb):
    # result = [abs(a - b) / abs(a) for a, b in zip(Pt, Pt_perturb)]
    assert len(Pt)==len(Pt_perturb)
    # result = [(a - b) / a for a, b in zip(Pt, Pt_perturb)]
    result = []
    for a, b in zip(Pt, Pt_perturb):
        if a != 0:
            result.append((a - b) / a)

    return sum(result)


class BaseTrainer(Trainer):
    def __init__(
        self,
        eval_collector,
        alpha=None,
        gamma=None,
        if_kl=False,
        mask=None,
        if_wanda=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.eval_collector = eval_collector
        self.alpha = alpha
        self.gamma = gamma
        self.mask = mask
        self.if_wanda = if_wanda
        if if_kl:
            self.infer_model = deepcopy(self.model)
            self.infer_model.eval()

    def prediction_step(
        self, model, inputs, prediction_loss_only: bool, ignore_keys=None
    ):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"].long()
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.eval_collector

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(
                eval_dataset, description="evaluation"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="evaluation"
            )

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

    def _inner_training_loop(
        self,
        batch_size=None,
        args=None,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = (
                        self._train_batch_size // max(1, self.args.n_gpu)
                    )
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(
            f"Currently training with a batch size of: {self._train_batch_size}"
        )
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = (
            self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        )

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = (
                len_dataloader // args.gradient_accumulation_steps
            )
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps)
                        * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(
                    args.num_train_epochs * num_update_steps_per_epoch
                )
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = (
                    self.num_examples(train_dataloader) * args.num_train_epochs
                )
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader) * args.num_train_epochs
                    )
        elif (
            args.max_steps > 0
        ):  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = (
                    self.num_tokens(train_dataloader, args.max_steps)
                    * args.gradient_accumulation_steps
                )
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            is_sagemaker_mp_enabled()
            or self.is_fsdp_xla_enabled
            or self.is_fsdp_enabled
        )

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps
            )

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(
                        self.model, self.optimizer
                    )
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(
            f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}"
        )
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(
                f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}"
            )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(
            f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (
                    num_update_steps_per_epoch
                )
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(
                f"  Continuing training from global step {self.state.global_step}"
            )
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = (
                trial.assignments
                if self.hp_search_backend == HPSearchBackend.SIGOPT
                else trial
            )
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control
        )

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        total_batched_samples = 0
        model_name = "locuslab/tofu_ft_llama2-7b"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False
        )
        MRD = [
            1.594127887366545, 0.5854753603049424, 0.19821048561101837, 0.08419759055384914, 
            0.6081453411497374, 0.8039139682103849, 0.029026737307038416, 0.031070634933948574, 
            0.004850820424454371, 0.04464946264655449, 0.12972659824371605, 0.13513061308099011, 
            0.055622538308256884, 0.08885978394484813, 0.33438323752674115, 0.21661494412020482, 
            0.8786005200738978, 0.3363235531580034, 0.7398143126447146, 0.4091245965489459, 
            0.13396880862006466, 1.126132873420497, 0.09635407121080623, 0.7467105032978582, 
            0.42884660322043877, 1.215242489003914, 0.008915811389634638, 0.9033097003899817, 
            1.3404629548455318, 0.05668155316620142, 0.1290606429574884, 1.3588263943357353, 
            0.41184831010882744, 0.3624522740329505, 0.7327809991067816, 0.6533591223095027, 
            0.07526404222668885, 0.019836174455646085, 0.1229357822253538, 0.5877990773089303, 
            0.030966451757614064, 1.1498566911206394, 0.3581495716223256, 0.2184290003926602, 
            0.6673665129706445, 0.22647844397504124, 0.08227571844984068, 0.09043906982507564, 
            1.238059304374264, 0.1634964264535267, 1.3468436347389723, 0.28098567551287895, 
            1.9035882013268242, 0.04699877673700778, 0.1662399047178067, 1.1643808627918077, 
            0.5284527693812376, 0.47441816664824044, 0.5720056808806906, 0.4679456667890454, 
            0.8595992488423788, 0.4313982312326409, 0.3638274013485793, 0.337646912581399, 
            0.4557570770961181, 2.690930176211783, 0.03971425679992057, 0.5250034589695702, 
            0.5849968866518988, 1.1514048084204755, 0.002778136565961423, 0.0055173364731558846, 
            0.255570419604746, 0.5030149681361373, 0.3198976565571596, 1.2070865290953061, 
            0.8276067605894517, 0.08448700630726208, 0.9117338591510695, 1.1233830640437699, 
            0.8055723148919123, 0.38598336473625106, 0.4346262972949168, 0.8837425138909474, 
            1.3808502665600881, 1.5136123724810533, 0.18237068397036424, 1.5696716193730038, 
            1.435683480770969, 0.015581488790257501, 0.7176651195706912, 0.763408379878266, 
            1.6587417120765975, 0.3278422008027513, 1.2850373024096444, 0.14582655601968053, 
            0.484536570137199, 0.36924403982214843, 0.49810955051208766, 1.6238777019885275, 
            0.18093049766534677, 0.6104981494128431, 0.18072597896835355, 0.7358808794749356, 
            0.5890722295379976, 0.3819794945014143, 1.5817832855911178, 1.4394294963993193, 
            0.03262719165428256, 1.008427062429464, 0.544559514212348, 0.954488983499352, 
            0.26430370642084117, 0.8508539575464805, 0.12703466705256986, 0.1102857486287629, 
            0.09233691932163417, 0.7854881504025367, 0.6678624609155936, 0.14773771716719897, 
            1.0596905616938193, 1.0756092026083033, 0.1898406697497707, 0.1524509297416718, 
            0.36174382378758857, 0.4454538303460502, 0.36871475907251056, 0.04340099538299585, 
            0.34022220438263795, 0.6073176899154964, 0.5237828663958071, 0.26496899319922435, 
            1.2414247681287043, 0.47327193113277133, 0.5778005286658907, 0.20507309941517105, 
            0.057674407981695965, 0.027604634637326996, 0.16877671571628206, 0.7576360357634861, 
            0.0946792469618212, 0.022448680445680914, 0.6304836195851267, 0.2355363289171173, 
            0.5204219531213179, 0.6688494269744067, 0.4019360397431297, 0.5228052686206525, 
            0.5793189172639549, 0.12895750200314393, 0.3582064682975343, 0.17665195835975933, 
            0.19499274793209184, 1.4439126593332892, 0.1513239169381354, 0.20179161244704755, 
            0.39376244560285273, 0.8036812699522627, 0.39491763043475975, 1.5668107912118268, 
            0.22309033077176607, 1.2004201335991114, 0.5860724390938903, 0.5623903201566665, 
            0.1277174333981132, 0.1275694767613706, 0.41683660809687395, 0.5154474313791849, 
            0.7548632629328221, 0.41051873859228577, 0.7458816619438626, 1.3588846602129367, 
            0.5975422988809577, 1.142372588715224, 0.8375833283465376, 0.2798172652376066, 
            0.30040635267026994, 0.5838264655312853, 0.26287322269176056, 0.01022695903224133, 
            0.3530839308423059, 0.1281746814073556, 0.6931587913526636, 1.5581646485186673, 
            1.0627471992471433, 0.16290249428852668, 0.7637709524418949, 0.4997266000992653, 
            0.6738946504003921, 0.3694957036101594, 0.731084870546985, 0.6512938377504908, 
            0.08760708955455047, 0.48542041844759404, 0.513522292748901, 0.595513296990084, 
            1.51776973069788, 0.6396455285623517, 0.05640959363231213, 1.45469173887927564, 
            0.3724968109908561, 0.8528525781831685, 0.018847343907695217, 0.7112875721783819, 
            1.5391268545761394, 0.3375302429639052, 0.0631841466737964, 0.3944642074704026, 
            0.36728045931002157, 0.880711626136904, 1.1889562608552537, 0.679096015479592, 
            0.41253738132698614, 0.7203791604615661, 0.373520356278522, 0.06635139428180958, 
            0.08159071536490706, 0.05151667941297942, 0.1723909298872338, 0.34468713905120557, 
            0.43107647567591323, 0.35144030764516665, 0.4552541163526827, 0.7607544220474476, 
            0.2786081248271137, 0.46516822004457026, 0.1278250418235765, 0.48971819084353496, 
            0.24140167077496863, 0.162953247962327, 0.37753220581469904, 0.022237500098766964, 
            0.1580385317469517, 0.17399366439667713, 1.0341219437455837, 0.38238786928147656, 
            0.23782032144563137, 0.10800338050644548, 0.17417467939189077, 0.3612078324547244, 
            0.7166601806849542, 0.4816339771565818, 0.8392179215287618, 0.21040284386528402, 
            0.6441583554913715, 0.35155534217454554, 0.027244143402038304, 0.6507632231902862, 
            0.4404468624026913, 0.6958970254656159, 0.3807564710532596, 0.27799626342462147, 
            0.41517042425461315, 0.5856433527932619, 1.3566260178130705, 0.948691767687962, 
            0.09038480338565114, 0.36252604971073166, 0.21332385785000338, 0.2267821487066928, 
            0.12162083203871245, 0.24562974705469418, 1.301113951797455, 0.3785641066262204, 
            0.2909996432093636, 0.2617448617024905, 0.7711998418175026, 0.9959315962374707, 
            0.12407880985996783, 0.06691741285621036, 0.894470766464373, 0.6943279961643231, 
            0.1870713990235713, 0.4897494417300501, 0.41984926545797413, 0.7184187064037983, 
            0.5957070005025158, 0.386524498648946, 0.050264718249017196, 0.1838924731711346, 
            1.2838679425283206, 0.06486398913254235, 0.5053483788808059, 0.6525946916203023, 
            0.281114907664422, 1.4530112282272176, 0.05773986135613164, 0.10330507428015549, 
            0.23511097937796963, 0.8687094829144931, 0.6764461056383757, 0.8204582527711618, 
            0.32419715575517005, 0.033014859872444194, 0.009370036365031486, 0.6527318043121609, 
            1.0786065652714414, 0.16097611232670592, 0.6007805926595179, 1.0221279124103797, 
            0.3992504095733282, 0.21856794700198873, 0.1428999084999511, 2.7600155123085726, 
            0.351709988844823, 0.0425778557018849, 0.4301249057176156, 1.4659839558952763, 
            0.7445848711709714, 1.5148474076697993, 0.8769685346906239, 0.4072230228635557, 
            0.6742081440533382, 0.4192691069952352, 0.2831225927137677, 1.253398120484492, 
            0.5040421012037452, 0.25197788109496805, 0.15175953282603838, 0.021290084237912742, 
            0.5642105069526331, 0.15464429707153723, 0.2112309381509471, 0.7780753340093605, 
            0.9430787476598639, 0.7721039159890748, 0.5168224549451156, 1.0564788904876545, 
            0.5580839581966714, 0.44628431031970434, 0.9425887765989391, 1.1824212955695834, 
            0.0268364427412145, 0.26696825077285413, 0.6193320593431035, 0.6359693135412428, 
            0.27151722166276854, 0.424491944905291, 0.5377293727086296, 1.1892642390035484, 
            0.4974289522293515, 0.20341265051658217, 0.31328609522747647, 0.9948740874149342, 
            2.0798422085753843, 0.5683933127767803, 0.3936548190461339, 0.03740573768679494, 
            0.25299742705236483, 0.2783725299667803, 0.3303528985704272, 1.4598414117129581, 
            1.8756457060573144, 0.19240716512872302, 1.8078192569994342, 0.7320829243240686, 
            0.7642827248219908, 0.12226595388394584, 0.6219171029678284, 0.44847526492598383, 
            0.10534008764799939, 0.19509528452043373, 0.6008038406447614, 0.9029045271838452, 
            0.47755842962725914, 1.1153363113106884, 0.20164103545725173, 0.010992632722193666, 
            0.0032646332270221765, 0.054380366645490466, 0.5726308184257016, 1.8646530529695793, 
            0.2028622353404632, 0.7539718478149043, 0.0021290184820839135, 0.3820058250993334, 
            1.4727073034637206, 0.5685161031812916, 0.9084529561831889, 0.641884386018499, 
            0.20866366689648058, 0.8072888748186619, 0.3940099973201865, 0.035499278056558316, 
            0.05800679788292228, 1.9360117673301338, 0.9684760623478549, 0.4267655886676192, 
            0.4139029940178691, 0.8363221303647207, 0.09391058203366123, 0.5619842077870385, 
            0.5836817673337732, 0.2732858693661943, 0.2946272598163264, 0.045686179471758175, 
            2.7295660296460693, 0.08846852862706965, 0.02152388042644458, 1.0177678919386548
        ]
        def transform_list(lst):
            # 计算中位数
            median = np.median(lst)
            
            # 变换列表
            transformed_lst = [0.4 if x < median else 0.6 for x in lst]
            
            return transformed_lst
        MRD = transform_list(MRD)
        data_list = list(train_dataloader)
        s_time = time.time()
        initial_score = 0
        for epoch in range(epochs_trained, num_train_epochs):
            # if epoch==4:
            #     print("epoch",epoch)
            #     compute_seta(self.model)
            #     sys.exit(0)
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control
            )

            if (
                epoch == epochs_trained
                and resume_from_checkpoint is not None
                and steps_trained_in_current_epoch == 0
            ):
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(
                    epoch_iterator, steps_trained_in_current_epoch
                )
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            all_score = 0
            step_count = 0
            
            # if epoch==0:
            #     m1_time = time.time()
            #     MRD = []
            #     original_state = deepcopy(model.state_dict())
            #     K = 10
            #     seed = 42

            #     for step, inputs in tqdm(enumerate(epoch_iterator)):
            #         one_MRD = 0
            #         for i in range(0,K):
            #             torch.manual_seed(seed+i)
            #             if torch.cuda.is_available():
            #                 torch.cuda.manual_seed_all(seed+i)
            #             model.load_state_dict(original_state)
            #             _,probs = get_token_probabilities(model, inputs['forget'][0],tokenizer)
            #             Pt = compute_P(probs)
            #             perturb_model_parameters(model)
            #             _,probs_perturb = get_token_probabilities(model, inputs['forget'][0],tokenizer)
            #             Pt_perturb = compute_P(probs_perturb)
            #             one_MRD += compute_MRD(Pt,Pt_perturb)
            #         MRD.append(min(abs(one_MRD/K),2))
            #     model.load_state_dict(original_state)
            #     del original_state
            #     MRD_avg = 0
            #     for i in MRD:
            #         MRD_avg += i
            #     MRD_avg /= len(MRD)
            #     m2_time = time.time()
            #     MRD = transform_list(MRD)
            for step, inputs in enumerate(epoch_iterator):
                # print(inputs['forget'][0].shape)
                # chosen_index = random.choices(range(len(MRD)), weights=MRD, k=1)[0]
                # print(chosen_index)
                # inputs = data_list[chosen_index]
                score,probs = get_token_probabilities(model, inputs['forget'][0],tokenizer)
                # with open("log.txt", "a", encoding="utf-8") as f:
                #     for step1, prob_info in enumerate(probs, 1):
                #         f.write(f"步骤 {step1}:\n")
                #         f.write(f"  前缀: {prob_info['prefix']}\n")
                #         f.write(f"  下一个 Token: {prob_info['token']}\n")
                #         f.write(f"  概率: {prob_info['probability']:.6f}\n\n")
                #     f.write(f"  score: {score}\n\n")
                # for step1, prob_info in enumerate(probs, 1):
                #     print(f"步骤 {step1}:")
                #     print(f"  前缀: {prob_info['prefix']}")
                #     print(f"  下一个 Token: {prob_info['token']}")
                #     print(f"  概率: {prob_info['probability']:.6f}\n")
                # print(score)
                all_score += score
                step_count += 1
                total_batched_samples += 1

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(
                        self.model, "main_input_name", "input_ids"
                    )
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += self.accelerator.gather(
                            inputs[main_input_name]
                        ).numel()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(
                        args, self.state, self.control
                    )
                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)
                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (
                        1 + self.state.global_step - self._globalstep_last_logged
                    )
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                )
                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping
                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )
                    if self.mask is not None:
                        self.mask_gradient(model, self.if_wanda)
                    # Optimizer step
                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(
                            self.lr_scheduler,
                            torch.optim.lr_scheduler.ReduceLROnPlateau,
                        ):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = (
                        epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    )
                    self.control = self.callback_handler.on_step_end(
                        args, self.state, self.control
                    )

                    self._maybe_log_save_evaluate(
                        tr_loss, model, trial, epoch, ignore_keys_for_eval
                    )
                else:
                    self.control = self.callback_handler.on_substep_end(
                        args, self.state, self.control
                    )

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            all_score /= step_count
            # if epoch==0:
            #     print("in!!!!!!!!!!!!!!!!!")
            #     initial_score = all_score
            #     print(initial_score)
            print("epoch:",epoch," score:",all_score)
            # if (initial_score-all_score)/initial_score>0.035:
            #     compute_seta(self.model)
            #     sys.exit(0)
            #     break
            # if all_score<=0.3:
            #     compute_seta(self.model)
            #     sys.exit(0)
            # if all_score<=0.02 or epoch==19:
            #     with open('/home/yz979/project/chengye/WAGLE/lab3_res.txt', 'a', encoding='utf-8') as f:
            #         f.write(f"{epoch}\n")
            #     sys.exit(0)
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(
                args, self.state, self.control
            )
            self._maybe_log_save_evaluate(
                tr_loss, model, trial, epoch, ignore_keys_for_eval
            )

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break
        # e_time = time.time()
        # print("all time:",e_time-s_time-(m2_time-m1_time))
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"
        )
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(
            use_mtime=False, output_dir=run_dir
        )

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if (
            self.args.should_save
            and self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
        ):
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(
                        f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit"
                    )
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control
        )

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def mask_gradient(self, model, if_wanda=False):
        if not if_wanda:
            if self.mask is not None:
                with torch.no_grad():
                    for key, tensor in model.named_parameters():
                        tensor.grad *= (
                            self.mask[key]
                        )
        else:
            layers = (
                model.model.layers
                if hasattr(model.model, "layers")
                else model.model.decoder.layers
            )

            cnt = 0

            for layer in layers:
                subset = find_layers(layer)
                for name in subset:
                    if self.mask is not None:
                        subset[name].weight.grad *= self.mask[cnt]
                    cnt += 1
