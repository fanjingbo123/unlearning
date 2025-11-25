#!/bin/bash
set -euo pipefail

# 说明：
# 1. 模型/数据默认均为本地路径，可按需修改 MODEL_ROOT、LOCAL_DATA_DIR。
# 2. 每个小节都包含：LoRA 微调 → 仅计算遗忘难度分数 → GA 遗忘（无排序）→ GA 遗忘（按难度升序）。
# 3. 为降低显存占用，LoRA 命令默认开启梯度检查点（--gradient_checkpointing），使用较短 max_length 与 batch_size=1、较高累积步数。
# 4. 默认使用多卡 DDP（torchrun），如需单卡可将 NPROC_PER_NODE=1 或直接去掉 torchrun 前缀。

MODEL_ROOT="../base_model"
LOCAL_DATA_DIR="../data"
CACHE_DIR=".cache"
NPROC_PER_NODE=4

###############################################################################
# Phi-3-mini-4k-instruct
###############################################################################

# ========================= WHP (Harry Potter QA) =============================
SAVE_HP_PHI="files/models/Phi-3-mini-4k-instruct/hp_lora"
SCORE_HP_PHI="files/logs/Phi-3-mini-4k-instruct/hp_difficulty.json"
LOG_HP_PHI_BASE="files/logs/Phi-3-mini-4k-instruct/ga_hp_base"
LOG_HP_PHI_SORT="files/logs/Phi-3-mini-4k-instruct/ga_hp_sorted"
mkdir -p "${SAVE_HP_PHI}" "$(dirname "${SCORE_HP_PHI}")" "${LOG_HP_PHI_BASE}" "${LOG_HP_PHI_SORT}"

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/Fine_tune_hp.py \
  --model_name "${MODEL_ROOT}/Phi-3-mini-4k-instruct" \
  --cache_dir "${CACHE_DIR}" \
  --epochs 3 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lr 1e-4 \
  --max_length 384 \
  --gradient_checkpointing \
  --attn_implementation eager \
  --save_dir "${SAVE_HP_PHI}"

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" python exec/unlearn_model.py \
  --overall.model_name "${SAVE_HP_PHI}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --unlearn.unlearn_method origin \
  --unlearn.task_name copyright \
  --unlearn.use_lora 1 \
  --unlearn.compute_difficulty_only 1 \
  --unlearn.difficulty_score_path "${SCORE_HP_PHI}" \
  --dataset.forget_dataset_name HP \
  --dataset.retain_dataset_name HP \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/unlearn_model.py \
  --overall.model_name "${SAVE_HP_PHI}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --overall.save_dir "${LOG_HP_PHI_BASE}" \
  --unlearn.unlearn_method GA \
  --unlearn.task_name copyright \
  --unlearn.use_lora 1 \
  --dataset.forget_dataset_name HP \
  --dataset.retain_dataset_name HP \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/unlearn_model.py \
  --overall.model_name "${SAVE_HP_PHI}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --overall.save_dir "${LOG_HP_PHI_SORT}" \
  --unlearn.unlearn_method GA \
  --unlearn.task_name copyright \
  --unlearn.use_lora 1 \
  --unlearn.enable_difficulty_sampling 1 \
  --unlearn.difficulty_order asc \
  --unlearn.difficulty_score_path "${SCORE_HP_PHI}" \
  --dataset.forget_dataset_name HP \
  --dataset.retain_dataset_name HP \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

# =============================== ToFU ========================================
SAVE_TOFU_PHI="files/models/Phi-3-mini-4k-instruct/tofu_lora"
SCORE_TOFU_PHI="files/logs/Phi-3-mini-4k-instruct/tofu_difficulty.json"
LOG_TOFU_PHI_BASE="files/logs/Phi-3-mini-4k-instruct/ga_tofu_base"
LOG_TOFU_PHI_SORT="files/logs/Phi-3-mini-4k-instruct/ga_tofu_sorted"
mkdir -p "${SAVE_TOFU_PHI}" "$(dirname "${SCORE_TOFU_PHI}")" "${LOG_TOFU_PHI_BASE}" "${LOG_TOFU_PHI_SORT}"

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/Fine_tune_tofu.py \
  --model_name "${MODEL_ROOT}/Phi-3-mini-4k-instruct" \
  --cache_dir "${CACHE_DIR}" \
  --epochs 2 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lr 5e-5 \
  --subset forget01 \
  --max_length 256 \
  --gradient_checkpointing \
  --attn_implementation eager \
  --save_dir "${SAVE_TOFU_PHI}"

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" python exec/unlearn_model.py \
  --overall.model_name "${SAVE_TOFU_PHI}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --unlearn.unlearn_method origin \
  --unlearn.task_name tofu \
  --unlearn.use_lora 1 \
  --unlearn.compute_difficulty_only 1 \
  --unlearn.difficulty_score_path "${SCORE_TOFU_PHI}" \
  --dataset.forget_dataset_name Tofu_forget01 \
  --dataset.retain_dataset_name Tofu_retain99 \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/unlearn_model.py \
  --overall.model_name "${SAVE_TOFU_PHI}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --overall.save_dir "${LOG_TOFU_PHI_BASE}" \
  --unlearn.unlearn_method GA \
  --unlearn.task_name tofu \
  --unlearn.use_lora 1 \
  --dataset.forget_dataset_name Tofu_forget01 \
  --dataset.retain_dataset_name Tofu_retain99 \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/unlearn_model.py \
  --overall.model_name "${SAVE_TOFU_PHI}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --overall.save_dir "${LOG_TOFU_PHI_SORT}" \
  --unlearn.unlearn_method GA \
  --unlearn.task_name tofu \
  --unlearn.use_lora 1 \
  --unlearn.enable_difficulty_sampling 1 \
  --unlearn.difficulty_order asc \
  --unlearn.difficulty_score_path "${SCORE_TOFU_PHI}" \
  --dataset.forget_dataset_name Tofu_forget01 \
  --dataset.retain_dataset_name Tofu_retain99 \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

# =============================== WMDP ========================================
SAVE_WMDP_PHI="files/models/Phi-3-mini-4k-instruct/wmdp_lora"
SCORE_WMDP_PHI="files/logs/Phi-3-mini-4k-instruct/wmdp_difficulty.json"
LOG_WMDP_PHI_BASE="files/logs/Phi-3-mini-4k-instruct/ga_wmdp_base"
LOG_WMDP_PHI_SORT="files/logs/Phi-3-mini-4k-instruct/ga_wmdp_sorted"
mkdir -p "${SAVE_WMDP_PHI}" "$(dirname "${SCORE_WMDP_PHI}")" "${LOG_WMDP_PHI_BASE}" "${LOG_WMDP_PHI_SORT}"

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/Fine_tune_wmdp.py \
  --model_name "${MODEL_ROOT}/Phi-3-mini-4k-instruct" \
  --cache_dir "${CACHE_DIR}" \
  --epochs 2 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lr 5e-5 \
  --domain all \
  --max_length 256 \
  --gradient_checkpointing \
  --attn_implementation eager \
  --save_dir "${SAVE_WMDP_PHI}"

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" python exec/unlearn_model.py \
  --overall.model_name "${SAVE_WMDP_PHI}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --unlearn.unlearn_method origin \
  --unlearn.task_name wmdp \
  --unlearn.use_lora 1 \
  --unlearn.compute_difficulty_only 1 \
  --unlearn.difficulty_score_path "${SCORE_WMDP_PHI}" \
  --dataset.forget_dataset_name WMDPALL \
  --dataset.retain_dataset_name WMDPALL \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/unlearn_model.py \
  --overall.model_name "${SAVE_WMDP_PHI}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --overall.save_dir "${LOG_WMDP_PHI_BASE}" \
  --unlearn.unlearn_method GA \
  --unlearn.task_name wmdp \
  --unlearn.use_lora 1 \
  --dataset.forget_dataset_name WMDPALL \
  --dataset.retain_dataset_name WMDPALL \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/unlearn_model.py \
  --overall.model_name "${SAVE_WMDP_PHI}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --overall.save_dir "${LOG_WMDP_PHI_SORT}" \
  --unlearn.unlearn_method GA \
  --unlearn.task_name wmdp \
  --unlearn.use_lora 1 \
  --unlearn.enable_difficulty_sampling 1 \
  --unlearn.difficulty_order asc \
  --unlearn.difficulty_score_path "${SCORE_WMDP_PHI}" \
  --dataset.forget_dataset_name WMDPALL \
  --dataset.retain_dataset_name WMDPALL \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

###############################################################################
# Qwen2.5-3B-Instruct
###############################################################################

# ========================= WHP (Harry Potter QA) =============================
SAVE_HP_QWEN="files/models/Qwen2.5-3B-Instruct/hp_lora"
SCORE_HP_QWEN="files/logs/Qwen2.5-3B-Instruct/hp_difficulty.json"
LOG_HP_QWEN_BASE="files/logs/Qwen2.5-3B-Instruct/ga_hp_base"
LOG_HP_QWEN_SORT="files/logs/Qwen2.5-3B-Instruct/ga_hp_sorted"
mkdir -p "${SAVE_HP_QWEN}" "$(dirname "${SCORE_HP_QWEN}")" "${LOG_HP_QWEN_BASE}" "${LOG_HP_QWEN_SORT}"

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/Fine_tune_hp.py \
  --model_name "${MODEL_ROOT}/Qwen2.5-3B-Instruct" \
  --cache_dir "${CACHE_DIR}" \
  --epochs 3 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lr 1e-4 \
  --max_length 384 \
  --gradient_checkpointing \
  --attn_implementation eager \
  --save_dir "${SAVE_HP_QWEN}"

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" python exec/unlearn_model.py \
  --overall.model_name "${SAVE_HP_QWEN}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --unlearn.unlearn_method origin \
  --unlearn.task_name copyright \
  --unlearn.use_lora 1 \
  --unlearn.compute_difficulty_only 1 \
  --unlearn.difficulty_score_path "${SCORE_HP_QWEN}" \
  --dataset.forget_dataset_name HP \
  --dataset.retain_dataset_name HP \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/unlearn_model.py \
  --overall.model_name "${SAVE_HP_QWEN}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --overall.save_dir "${LOG_HP_QWEN_BASE}" \
  --unlearn.unlearn_method GA \
  --unlearn.task_name copyright \
  --unlearn.use_lora 1 \
  --dataset.forget_dataset_name HP \
  --dataset.retain_dataset_name HP \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/unlearn_model.py \
  --overall.model_name "${SAVE_HP_QWEN}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --overall.save_dir "${LOG_HP_QWEN_SORT}" \
  --unlearn.unlearn_method GA \
  --unlearn.task_name copyright \
  --unlearn.use_lora 1 \
  --unlearn.enable_difficulty_sampling 1 \
  --unlearn.difficulty_order asc \
  --unlearn.difficulty_score_path "${SCORE_HP_QWEN}" \
  --dataset.forget_dataset_name HP \
  --dataset.retain_dataset_name HP \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

# =============================== ToFU ========================================
SAVE_TOFU_QWEN="files/models/Qwen2.5-3B-Instruct/tofu_lora"
SCORE_TOFU_QWEN="files/logs/Qwen2.5-3B-Instruct/tofu_difficulty.json"
LOG_TOFU_QWEN_BASE="files/logs/Qwen2.5-3B-Instruct/ga_tofu_base"
LOG_TOFU_QWEN_SORT="files/logs/Qwen2.5-3B-Instruct/ga_tofu_sorted"
mkdir -p "${SAVE_TOFU_QWEN}" "$(dirname "${SCORE_TOFU_QWEN}")" "${LOG_TOFU_QWEN_BASE}" "${LOG_TOFU_QWEN_SORT}"

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/Fine_tune_tofu.py \
  --model_name "${MODEL_ROOT}/Qwen2.5-3B-Instruct" \
  --cache_dir "${CACHE_DIR}" \
  --epochs 2 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lr 5e-5 \
  --subset forget01 \
  --max_length 256 \
  --gradient_checkpointing \
  --attn_implementation eager \
  --save_dir "${SAVE_TOFU_QWEN}"

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" python exec/unlearn_model.py \
  --overall.model_name "${SAVE_TOFU_QWEN}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --unlearn.unlearn_method origin \
  --unlearn.task_name tofu \
  --unlearn.use_lora 1 \
  --unlearn.compute_difficulty_only 1 \
  --unlearn.difficulty_score_path "${SCORE_TOFU_QWEN}" \
  --dataset.forget_dataset_name Tofu_forget01 \
  --dataset.retain_dataset_name Tofu_retain99 \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/unlearn_model.py \
  --overall.model_name "${SAVE_TOFU_QWEN}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --overall.save_dir "${LOG_TOFU_QWEN_BASE}" \
  --unlearn.unlearn_method GA \
  --unlearn.task_name tofu \
  --unlearn.use_lora 1 \
  --dataset.forget_dataset_name Tofu_forget01 \
  --dataset.retain_dataset_name Tofu_retain99 \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/unlearn_model.py \
  --overall.model_name "${SAVE_TOFU_QWEN}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --overall.save_dir "${LOG_TOFU_QWEN_SORT}" \
  --unlearn.unlearn_method GA \
  --unlearn.task_name tofu \
  --unlearn.use_lora 1 \
  --unlearn.enable_difficulty_sampling 1 \
  --unlearn.difficulty_order asc \
  --unlearn.difficulty_score_path "${SCORE_TOFU_QWEN}" \
  --dataset.forget_dataset_name Tofu_forget01 \
  --dataset.retain_dataset_name Tofu_retain99 \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

# =============================== WMDP ========================================
SAVE_WMDP_QWEN="files/models/Qwen2.5-3B-Instruct/wmdp_lora"
SCORE_WMDP_QWEN="files/logs/Qwen2.5-3B-Instruct/wmdp_difficulty.json"
LOG_WMDP_QWEN_BASE="files/logs/Qwen2.5-3B-Instruct/ga_wmdp_base"
LOG_WMDP_QWEN_SORT="files/logs/Qwen2.5-3B-Instruct/ga_wmdp_sorted"
mkdir -p "${SAVE_WMDP_QWEN}" "$(dirname "${SCORE_WMDP_QWEN}")" "${LOG_WMDP_QWEN_BASE}" "${LOG_WMDP_QWEN_SORT}"

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/Fine_tune_wmdp.py \
  --model_name "${MODEL_ROOT}/Qwen2.5-3B-Instruct" \
  --cache_dir "${CACHE_DIR}" \
  --epochs 2 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lr 5e-5 \
  --domain all \
  --max_length 256 \
  --gradient_checkpointing \
  --attn_implementation eager \
  --save_dir "${SAVE_WMDP_QWEN}"

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" python exec/unlearn_model.py \
  --overall.model_name "${SAVE_WMDP_QWEN}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --unlearn.unlearn_method origin \
  --unlearn.task_name wmdp \
  --unlearn.use_lora 1 \
  --unlearn.compute_difficulty_only 1 \
  --unlearn.difficulty_score_path "${SCORE_WMDP_QWEN}" \
  --dataset.forget_dataset_name WMDPALL \
  --dataset.retain_dataset_name WMDPALL \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/unlearn_model.py \
  --overall.model_name "${SAVE_WMDP_QWEN}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --overall.save_dir "${LOG_WMDP_QWEN_BASE}" \
  --unlearn.unlearn_method GA \
  --unlearn.task_name wmdp \
  --unlearn.use_lora 1 \
  --dataset.forget_dataset_name WMDPALL \
  --dataset.retain_dataset_name WMDPALL \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/unlearn_model.py \
  --overall.model_name "${SAVE_WMDP_QWEN}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --overall.save_dir "${LOG_WMDP_QWEN_SORT}" \
  --unlearn.unlearn_method GA \
  --unlearn.task_name wmdp \
  --unlearn.use_lora 1 \
  --unlearn.enable_difficulty_sampling 1 \
  --unlearn.difficulty_order asc \
  --unlearn.difficulty_score_path "${SCORE_WMDP_QWEN}" \
  --dataset.forget_dataset_name WMDPALL \
  --dataset.retain_dataset_name WMDPALL \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

###############################################################################
# TinyLlama v1.1
###############################################################################

# ========================= WHP (Harry Potter QA) =============================
SAVE_HP_TINY="files/models/TinyLlama_v1.1/hp_lora"
SCORE_HP_TINY="files/logs/TinyLlama_v1.1/hp_difficulty.json"
LOG_HP_TINY_BASE="files/logs/TinyLlama_v1.1/ga_hp_base"
LOG_HP_TINY_SORT="files/logs/TinyLlama_v1.1/ga_hp_sorted"
mkdir -p "${SAVE_HP_TINY}" "$(dirname "${SCORE_HP_TINY}")" "${LOG_HP_TINY_BASE}" "${LOG_HP_TINY_SORT}"

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/Fine_tune_hp.py \
  --model_name "${MODEL_ROOT}/TinyLlama_v1.1" \
  --cache_dir "${CACHE_DIR}" \
  --epochs 3 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lr 1e-4 \
  --max_length 384 \
  --gradient_checkpointing \
  --attn_implementation eager \
  --save_dir "${SAVE_HP_TINY}"

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" python exec/unlearn_model.py \
  --overall.model_name "${SAVE_HP_TINY}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --unlearn.unlearn_method origin \
  --unlearn.task_name copyright \
  --unlearn.use_lora 1 \
  --unlearn.compute_difficulty_only 1 \
  --unlearn.difficulty_score_path "${SCORE_HP_TINY}" \
  --dataset.forget_dataset_name HP \
  --dataset.retain_dataset_name HP \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/unlearn_model.py \
  --overall.model_name "${SAVE_HP_TINY}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --overall.save_dir "${LOG_HP_TINY_BASE}" \
  --unlearn.unlearn_method GA \
  --unlearn.task_name copyright \
  --unlearn.use_lora 1 \
  --dataset.forget_dataset_name HP \
  --dataset.retain_dataset_name HP \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/unlearn_model.py \
  --overall.model_name "${SAVE_HP_TINY}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --overall.save_dir "${LOG_HP_TINY_SORT}" \
  --unlearn.unlearn_method GA \
  --unlearn.task_name copyright \
  --unlearn.use_lora 1 \
  --unlearn.enable_difficulty_sampling 1 \
  --unlearn.difficulty_order asc \
  --unlearn.difficulty_score_path "${SCORE_HP_TINY}" \
  --dataset.forget_dataset_name HP \
  --dataset.retain_dataset_name HP \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

# =============================== ToFU ========================================
SAVE_TOFU_TINY="files/models/TinyLlama_v1.1/tofu_lora"
SCORE_TOFU_TINY="files/logs/TinyLlama_v1.1/tofu_difficulty.json"
LOG_TOFU_TINY_BASE="files/logs/TinyLlama_v1.1/ga_tofu_base"
LOG_TOFU_TINY_SORT="files/logs/TinyLlama_v1.1/ga_tofu_sorted"
mkdir -p "${SAVE_TOFU_TINY}" "$(dirname "${SCORE_TOFU_TINY}")" "${LOG_TOFU_TINY_BASE}" "${LOG_TOFU_TINY_SORT}"

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/Fine_tune_tofu.py \
  --model_name "${MODEL_ROOT}/TinyLlama_v1.1" \
  --cache_dir "${CACHE_DIR}" \
  --epochs 2 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lr 5e-5 \
  --subset forget01 \
  --max_length 256 \
  --gradient_checkpointing \
  --attn_implementation eager \
  --save_dir "${SAVE_TOFU_TINY}"

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" python exec/unlearn_model.py \
  --overall.model_name "${SAVE_TOFU_TINY}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --unlearn.unlearn_method origin \
  --unlearn.task_name tofu \
  --unlearn.use_lora 1 \
  --unlearn.compute_difficulty_only 1 \
  --unlearn.difficulty_score_path "${SCORE_TOFU_TINY}" \
  --dataset.forget_dataset_name Tofu_forget01 \
  --dataset.retain_dataset_name Tofu_retain99 \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/unlearn_model.py \
  --overall.model_name "${SAVE_TOFU_TINY}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --overall.save_dir "${LOG_TOFU_TINY_BASE}" \
  --unlearn.unlearn_method GA \
  --unlearn.task_name tofu \
  --unlearn.use_lora 1 \
  --dataset.forget_dataset_name Tofu_forget01 \
  --dataset.retain_dataset_name Tofu_retain99 \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/unlearn_model.py \
  --overall.model_name "${SAVE_TOFU_TINY}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --overall.save_dir "${LOG_TOFU_TINY_SORT}" \
  --unlearn.unlearn_method GA \
  --unlearn.task_name tofu \
  --unlearn.use_lora 1 \
  --unlearn.enable_difficulty_sampling 1 \
  --unlearn.difficulty_order asc \
  --unlearn.difficulty_score_path "${SCORE_TOFU_TINY}" \
  --dataset.forget_dataset_name Tofu_forget01 \
  --dataset.retain_dataset_name Tofu_retain99 \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

# =============================== WMDP ========================================
SAVE_WMDP_TINY="files/models/TinyLlama_v1.1/wmdp_lora"
SCORE_WMDP_TINY="files/logs/TinyLlama_v1.1/wmdp_difficulty.json"
LOG_WMDP_TINY_BASE="files/logs/TinyLlama_v1.1/ga_wmdp_base"
LOG_WMDP_TINY_SORT="files/logs/TinyLlama_v1.1/ga_wmdp_sorted"
mkdir -p "${SAVE_WMDP_TINY}" "$(dirname "${SCORE_WMDP_TINY}")" "${LOG_WMDP_TINY_BASE}" "${LOG_WMDP_TINY_SORT}"

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/Fine_tune_wmdp.py \
  --model_name "${MODEL_ROOT}/TinyLlama_v1.1" \
  --cache_dir "${CACHE_DIR}" \
  --epochs 2 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lr 5e-5 \
  --domain all \
  --max_length 256 \
  --gradient_checkpointing \
  --attn_implementation eager \
  --save_dir "${SAVE_WMDP_TINY}"

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" python exec/unlearn_model.py \
  --overall.model_name "${SAVE_WMDP_TINY}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --unlearn.unlearn_method origin \
  --unlearn.task_name wmdp \
  --unlearn.use_lora 1 \
  --unlearn.compute_difficulty_only 1 \
  --unlearn.difficulty_score_path "${SCORE_WMDP_TINY}" \
  --dataset.forget_dataset_name WMDPALL \
  --dataset.retain_dataset_name WMDPALL \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/unlearn_model.py \
  --overall.model_name "${SAVE_WMDP_TINY}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --overall.save_dir "${LOG_WMDP_TINY_BASE}" \
  --unlearn.unlearn_method GA \
  --unlearn.task_name wmdp \
  --unlearn.use_lora 1 \
  --dataset.forget_dataset_name WMDPALL \
  --dataset.retain_dataset_name WMDPALL \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/unlearn_model.py \
  --overall.model_name "${SAVE_WMDP_TINY}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --overall.save_dir "${LOG_WMDP_TINY_SORT}" \
  --unlearn.unlearn_method GA \
  --unlearn.task_name wmdp \
  --unlearn.use_lora 1 \
  --unlearn.enable_difficulty_sampling 1 \
  --unlearn.difficulty_order asc \
  --unlearn.difficulty_score_path "${SCORE_WMDP_TINY}" \
  --dataset.forget_dataset_name WMDPALL \
  --dataset.retain_dataset_name WMDPALL \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

