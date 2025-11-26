import json
from typing import Dict, Iterable, List, Tuple

import torch.distributed as dist
import torch
from torch.utils.data import DataLoader
from contextlib import nullcontext


def _zero_grad(model):
    """Release gradient buffers to free GPU memory."""
    for _, param in model.named_parameters():
        if param.grad is not None:
            # set_to_none=True semantics to release memory instead of keeping zeroed buffers
            param.grad = None


def _accumulate_gradients(model, grad_store: Dict[str, torch.Tensor]):
    """Collect gradients from the model and offload to CPU to save GPU memory."""

    # Avoid creating a temporary FP32 copy on GPU; move to CPU first then cast
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_cpu = param.grad.detach().cpu().to(torch.float32)
                if name not in grad_store:
                    grad_store[name] = grad_cpu.clone()
                else:
                    grad_store[name].add_(grad_cpu)


def _autocast_context(model: torch.nn.Module):
    """Return an autocast context to shrink activation memory on CUDA."""

    if next(model.parameters()).is_cuda:
        # prefer bf16 if可用，否则退回 fp16
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.cuda.amp.autocast(dtype=dtype)
    return nullcontext()


def collect_epoch_gradient(
    model: torch.nn.Module,
    dataloader: DataLoader,
    gradient_accumulation_steps: int,
) -> Dict[str, torch.Tensor]:
    """Run one epoch over the forget dataloader and accumulate gradients."""
    model.train()
    epoch_grad: Dict[str, torch.Tensor] = {}
    step = 0
    _zero_grad(model)
    device = next(model.parameters()).device

    autocast_ctx = _autocast_context(model)

    for batch in dataloader:
        forget_inputs = batch["forget"]
        input_ids, attention_mask, labels = (
            forget_inputs[0].to(device),
            forget_inputs[1].to(device),
            forget_inputs[2].to(device),
        )
        with autocast_ctx:
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss / gradient_accumulation_steps
        loss.backward()
        step += 1

        if step % gradient_accumulation_steps == 0:
            _accumulate_gradients(model, epoch_grad)
            _zero_grad(model)

    # flush the last micro-step gradients
    if step % gradient_accumulation_steps != 0:
        _accumulate_gradients(model, epoch_grad)
        _zero_grad(model)

    if dist.is_available() and dist.is_initialized():
        device = next(model.parameters()).device
        for name, tensor in epoch_grad.items():
            device_tensor = tensor.to(device)
            dist.all_reduce(device_tensor, op=dist.ReduceOp.SUM)
            device_tensor /= dist.get_world_size()
            epoch_grad[name] = device_tensor.cpu()

    return epoch_grad


def _flattened_dot(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> torch.Tensor:
    return torch.dot(tensor_a.view(-1), tensor_b.view(-1))


def _projection_norm(sample_grad: Dict[str, torch.Tensor], epoch_grad: Dict[str, torch.Tensor]) -> float:
    shared_keys = sample_grad.keys() & epoch_grad.keys()
    if not shared_keys:
        return 0.0

    with torch.no_grad():
        epoch_norm_sq = sum((epoch_grad[k] ** 2).sum() for k in shared_keys)
        if epoch_norm_sq.item() == 0:
            return 0.0

        dot_product = sum(_flattened_dot(sample_grad[k], epoch_grad[k]) for k in shared_keys)
        return float(torch.abs(dot_product) / torch.sqrt(epoch_norm_sq))


def compute_difficulty_scores(
    model: torch.nn.Module,
    dataloader: DataLoader,
    epoch_grad: Dict[str, torch.Tensor],
    save_path: str,
) -> List[Dict[str, float]]:
    """Compute per-sample projection norms onto the epoch gradient and save as JSON."""
    device = next(model.parameters()).device
    model.train()
    scores: List[Dict[str, float]] = []

    sampler_indices = None
    if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "__iter__"):
        sampler_indices = list(iter(dataloader.sampler))
        sampler_iter = iter(sampler_indices)

    autocast_ctx = _autocast_context(model)

    for idx, batch in enumerate(dataloader):
        _zero_grad(model)
        forget_inputs = batch["forget"]
        input_ids, attention_mask, labels = (
            forget_inputs[0].to(device),
            forget_inputs[1].to(device),
            forget_inputs[2].to(device),
        )
        with autocast_ctx:
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
        outputs.loss.backward()
        sample_grad: Dict[str, torch.Tensor] = {}
        _accumulate_gradients(model, sample_grad)
        sample_index = idx if sampler_indices is None else next(sampler_iter)
        score = _projection_norm(sample_grad, epoch_grad)
        scores.append({"index": sample_index, "score": score})

    if dist.is_available() and dist.is_initialized():
        gathered: List[List[Dict[str, float]]] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, scores)
        if dist.get_rank() == 0:
            merged: List[Dict[str, float]] = []
            for part in gathered:
                merged.extend(part)
            merged_sorted = sorted(merged, key=lambda x: x["index"])
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(merged_sorted, f, ensure_ascii=False, indent=2)
        dist.barrier()
    else:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)

    _zero_grad(model)
    return scores
