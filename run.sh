#!/bin/bash
set -euo pipefail

# 说明：
# - 默认使用 HuggingFace 自动下载模型与数据到 .cache 目录；如需自备权重可将 --model_name 换成本地路径。
# - 三个基础模型：Phi-3-mini-4k-instruct、Qwen2.5-3B-Instruct、TinyLlama_v1.1。
# - 每个模型都会先做 LoRA 微调，再在对应遗忘集上计算样本遗忘难度分数（JSON）。

CACHE_DIR=".cache"
MODELS=("Phi-3-mini-4k-instruct" "Qwen2.5-3B-Instruct" "TinyLlama_v1.1")

############################
# 1) WHP (Harry Potter) 流程
############################
for MODEL in "${MODELS[@]}"; do
  SAVE_DIR="files/models/${MODEL}/hp_lora"
  SCORE_PATH="files/logs/${MODEL}/hp_difficulty.json"
  mkdir -p "${SAVE_DIR}" "$(dirname "${SCORE_PATH}")"

  # LoRA 微调
  python exec/Fine_tune_hp.py \
    --model_name "${MODEL}" \
    --cache_dir "${CACHE_DIR}" \
    --epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr 1e-4 \
    --save_dir "${SAVE_DIR}"

  # 遗忘难度计算（仅打分不训练）
  python exec/unlearn_model.py \
    --overall.model_name "${SAVE_DIR}" \
    --overall.cache_dir "${CACHE_DIR}" \
    --overall.logger json \
    --unlearn.unlearn_method origin \
    --unlearn.task_name copyright \
    --unlearn.use_lora 1 \
    --unlearn.compute_difficulty_only 1 \
    --unlearn.difficulty_score_path "${SCORE_PATH}" \
    --dataset.forget_dataset_name HP \
    --dataset.retain_dataset_name HP \
    --dataset.batch_size 2 \
    --dataset.forget_ratio 1.0

done

############################
# 2) ToFU 流程（forget01 / retain01）
############################
for MODEL in "${MODELS[@]}"; do
  SAVE_DIR="files/models/${MODEL}/tofu_lora"
  SCORE_PATH="files/logs/${MODEL}/tofu_difficulty.json"
  mkdir -p "${SAVE_DIR}" "$(dirname "${SCORE_PATH}")"

  # LoRA 微调（可根据需要调整 subset，例如 full/forget01/retain01）
  python exec/Fine_tune_tofu.py \
    --model_name "${MODEL}" \
    --cache_dir "${CACHE_DIR}" \
    --epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --subset full \
    --lr 1e-4 \
    --save_dir "${SAVE_DIR}"

  # 遗忘难度计算，默认遗忘集=Tofu_forget01，保留集=Tofu_retain01
  python exec/unlearn_model.py \
    --overall.model_name "${SAVE_DIR}" \
    --overall.cache_dir "${CACHE_DIR}" \
    --overall.logger json \
    --unlearn.unlearn_method origin \
    --unlearn.task_name tofu \
    --unlearn.use_lora 1 \
    --unlearn.compute_difficulty_only 1 \
    --unlearn.difficulty_score_path "${SCORE_PATH}" \
    --dataset.forget_dataset_name Tofu_forget01 \
    --dataset.retain_dataset_name Tofu_retain01 \
    --dataset.batch_size 2 \
    --dataset.forget_ratio 1.0

done

#############################################
# 3) WMDP 流程（ALL 域，可改 cyber/bio）
#############################################
for MODEL in "${MODELS[@]}"; do
  SAVE_DIR="files/models/${MODEL}/wmdp_lora"
  SCORE_PATH="files/logs/${MODEL}/wmdp_difficulty.json"
  mkdir -p "${SAVE_DIR}" "$(dirname "${SCORE_PATH}")"

  # LoRA 微调
  python exec/Fine_tune_wmdp.py \
    --model_name "${MODEL}" \
    --cache_dir "${CACHE_DIR}" \
    --epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --domain all \
    --lr 1e-4 \
    --save_dir "${SAVE_DIR}"

  # 遗忘难度计算
  python exec/unlearn_model.py \
    --overall.model_name "${SAVE_DIR}" \
    --overall.cache_dir "${CACHE_DIR}" \
    --overall.logger json \
    --unlearn.unlearn_method origin \
    --unlearn.task_name wmdp \
    --unlearn.use_lora 1 \
    --unlearn.compute_difficulty_only 1 \
    --unlearn.difficulty_score_path "${SCORE_PATH}" \
    --dataset.forget_dataset_name WMDPALL \
    --dataset.retain_dataset_name WMDPALL \
    --dataset.batch_size 1 \
    --dataset.forget_ratio 1.0

done

