import json
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader


def _zero_grad(model):
    for _, param in model.named_parameters():
        if param.grad is not None:
            param.grad.zero_()


def _accumulate_gradients(model, grad_store: Dict[str, torch.Tensor]):
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad = param.grad.detach().float()
            if name not in grad_store:
                grad_store[name] = grad.clone()
            else:
                grad_store[name] += grad


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

    for batch in dataloader:
        forget_inputs = batch["forget"]
        input_ids, attention_mask, labels = (
            forget_inputs[0].to(device),
            forget_inputs[1].to(device),
            forget_inputs[2].to(device),
        )
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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

    for idx, batch in enumerate(dataloader):
        _zero_grad(model)
        forget_inputs = batch["forget"]
        input_ids, attention_mask, labels = (
            forget_inputs[0].to(device),
            forget_inputs[1].to(device),
            forget_inputs[2].to(device),
        )
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()
        sample_grad: Dict[str, torch.Tensor] = {}
        _accumulate_gradients(model, sample_grad)
        score = _projection_norm(sample_grad, epoch_grad)
        scores.append({"index": idx, "score": score})

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    _zero_grad(model)
    return scores
