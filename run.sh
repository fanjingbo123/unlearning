#!/bin/bash
set -euo pipefail

# 说明：
# - 模型与数据均从本地读取，不会触发在线下载；默认路径：MODEL_ROOT=../base_model、LOCAL_DATA_DIR=../data。
# - 脚本按模型/数据集逐条列出命令，复制即可运行（默认单卡）。
# - 每个模型在 WHP、ToFU、WMDP 上依次进行 LoRA 微调与遗忘难度打分。

MODEL_ROOT="../base_model"
LOCAL_DATA_DIR="../data"
CACHE_DIR=".cache"

############################
# Phi-3-mini-4k-instruct
############################
SAVE_HP="files/models/Phi-3-mini-4k-instruct/hp_lora"
SCORE_HP="files/logs/Phi-3-mini-4k-instruct/hp_difficulty.json"
mkdir -p "${SAVE_HP}" "$(dirname "${SCORE_HP}")"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" python exec/Fine_tune_hp.py \
  --model_name "${MODEL_ROOT}/Phi-3-mini-4k-instruct" \
  --cache_dir "${CACHE_DIR}" \
  --epochs 3 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --lr 1e-4 \
  --save_dir "${SAVE_HP}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" python exec/unlearn_model.py \
  --overall.model_name "${SAVE_HP}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --unlearn.unlearn_method origin \
  --unlearn.task_name copyright \
  --unlearn.use_lora 1 \
  --unlearn.compute_difficulty_only 1 \
  --unlearn.difficulty_score_path "${SCORE_HP}" \
  --dataset.forget_dataset_name HP \
  --dataset.retain_dataset_name HP \
  --dataset.batch_size 2 \
  --dataset.forget_ratio 1.0

SAVE_TOFU="files/models/Phi-3-mini-4k-instruct/tofu_lora"
SCORE_TOFU="files/logs/Phi-3-mini-4k-instruct/tofu_difficulty.json"
mkdir -p "${SAVE_TOFU}" "$(dirname "${SCORE_TOFU}")"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" python exec/Fine_tune_tofu.py \
  --model_name "${MODEL_ROOT}/Phi-3-mini-4k-instruct" \
  --cache_dir "${CACHE_DIR}" \
  --epochs 3 \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --subset full \
  --lr 1e-4 \
  --save_dir "${SAVE_TOFU}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" python exec/unlearn_model.py \
  --overall.model_name "${SAVE_TOFU}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --unlearn.unlearn_method origin \
  --unlearn.task_name tofu \
  --unlearn.use_lora 1 \
  --unlearn.compute_difficulty_only 1 \
  --unlearn.difficulty_score_path "${SCORE_TOFU}" \
  --dataset.forget_dataset_name Tofu_forget01 \
  --dataset.retain_dataset_name Tofu_retain01 \
  --dataset.batch_size 2 \
  --dataset.forget_ratio 1.0

SAVE_WMDP="files/models/Phi-3-mini-4k-instruct/wmdp_lora"
SCORE_WMDP="files/logs/Phi-3-mini-4k-instruct/wmdp_difficulty.json"
mkdir -p "${SAVE_WMDP}" "$(dirname "${SCORE_WMDP}")"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" python exec/Fine_tune_wmdp.py \
  --model_name "${MODEL_ROOT}/Phi-3-mini-4k-instruct" \
  --cache_dir "${CACHE_DIR}" \
  --epochs 3 \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --domain all \
  --lr 1e-4 \
  --save_dir "${SAVE_WMDP}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" python exec/unlearn_model.py \
  --overall.model_name "${SAVE_WMDP}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --unlearn.unlearn_method origin \
  --unlearn.task_name wmdp \
  --unlearn.use_lora 1 \
  --unlearn.compute_difficulty_only 1 \
  --unlearn.difficulty_score_path "${SCORE_WMDP}" \
  --dataset.forget_dataset_name WMDPALL \
  --dataset.retain_dataset_name WMDPALL \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

############################
# Qwen2.5-3B-Instruct
############################
SAVE_HP="files/models/Qwen2.5-3B-Instruct/hp_lora"
SCORE_HP="files/logs/Qwen2.5-3B-Instruct/hp_difficulty.json"
mkdir -p "${SAVE_HP}" "$(dirname "${SCORE_HP}")"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" python exec/Fine_tune_hp.py \
  --model_name "${MODEL_ROOT}/Qwen2.5-3B-Instruct" \
  --cache_dir "${CACHE_DIR}" \
  --epochs 3 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --lr 1e-4 \
  --save_dir "${SAVE_HP}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" python exec/unlearn_model.py \
  --overall.model_name "${SAVE_HP}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --unlearn.unlearn_method origin \
  --unlearn.task_name copyright \
  --unlearn.use_lora 1 \
  --unlearn.compute_difficulty_only 1 \
  --unlearn.difficulty_score_path "${SCORE_HP}" \
  --dataset.forget_dataset_name HP \
  --dataset.retain_dataset_name HP \
  --dataset.batch_size 2 \
  --dataset.forget_ratio 1.0

SAVE_TOFU="files/models/Qwen2.5-3B-Instruct/tofu_lora"
SCORE_TOFU="files/logs/Qwen2.5-3B-Instruct/tofu_difficulty.json"
mkdir -p "${SAVE_TOFU}" "$(dirname "${SCORE_TOFU}")"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" python exec/Fine_tune_tofu.py \
  --model_name "${MODEL_ROOT}/Qwen2.5-3B-Instruct" \
  --cache_dir "${CACHE_DIR}" \
  --epochs 3 \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --subset full \
  --lr 1e-4 \
  --save_dir "${SAVE_TOFU}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" python exec/unlearn_model.py \
  --overall.model_name "${SAVE_TOFU}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --unlearn.unlearn_method origin \
  --unlearn.task_name tofu \
  --unlearn.use_lora 1 \
  --unlearn.compute_difficulty_only 1 \
  --unlearn.difficulty_score_path "${SCORE_TOFU}" \
  --dataset.forget_dataset_name Tofu_forget01 \
  --dataset.retain_dataset_name Tofu_retain01 \
  --dataset.batch_size 2 \
  --dataset.forget_ratio 1.0

SAVE_WMDP="files/models/Qwen2.5-3B-Instruct/wmdp_lora"
SCORE_WMDP="files/logs/Qwen2.5-3B-Instruct/wmdp_difficulty.json"
mkdir -p "${SAVE_WMDP}" "$(dirname "${SCORE_WMDP}")"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" python exec/Fine_tune_wmdp.py \
  --model_name "${MODEL_ROOT}/Qwen2.5-3B-Instruct" \
  --cache_dir "${CACHE_DIR}" \
  --epochs 3 \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --domain all \
  --lr 1e-4 \
  --save_dir "${SAVE_WMDP}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" python exec/unlearn_model.py \
  --overall.model_name "${SAVE_WMDP}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --unlearn.unlearn_method origin \
  --unlearn.task_name wmdp \
  --unlearn.use_lora 1 \
  --unlearn.compute_difficulty_only 1 \
  --unlearn.difficulty_score_path "${SCORE_WMDP}" \
  --dataset.forget_dataset_name WMDPALL \
  --dataset.retain_dataset_name WMDPALL \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0

############################
# TinyLlama_v1.1
############################
SAVE_HP="files/models/TinyLlama_v1.1/hp_lora"
SCORE_HP="files/logs/TinyLlama_v1.1/hp_difficulty.json"
mkdir -p "${SAVE_HP}" "$(dirname "${SCORE_HP}")"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" python exec/Fine_tune_hp.py \
  --model_name "${MODEL_ROOT}/TinyLlama_v1.1" \
  --cache_dir "${CACHE_DIR}" \
  --epochs 3 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --lr 1e-4 \
  --save_dir "${SAVE_HP}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" python exec/unlearn_model.py \
  --overall.model_name "${SAVE_HP}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --unlearn.unlearn_method origin \
  --unlearn.task_name copyright \
  --unlearn.use_lora 1 \
  --unlearn.compute_difficulty_only 1 \
  --unlearn.difficulty_score_path "${SCORE_HP}" \
  --dataset.forget_dataset_name HP \
  --dataset.retain_dataset_name HP \
  --dataset.batch_size 2 \
  --dataset.forget_ratio 1.0

SAVE_TOFU="files/models/TinyLlama_v1.1/tofu_lora"
SCORE_TOFU="files/logs/TinyLlama_v1.1/tofu_difficulty.json"
mkdir -p "${SAVE_TOFU}" "$(dirname "${SCORE_TOFU}")"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" python exec/Fine_tune_tofu.py \
  --model_name "${MODEL_ROOT}/TinyLlama_v1.1" \
  --cache_dir "${CACHE_DIR}" \
  --epochs 3 \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --subset full \
  --lr 1e-4 \
  --save_dir "${SAVE_TOFU}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" python exec/unlearn_model.py \
  --overall.model_name "${SAVE_TOFU}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --unlearn.unlearn_method origin \
  --unlearn.task_name tofu \
  --unlearn.use_lora 1 \
  --unlearn.compute_difficulty_only 1 \
  --unlearn.difficulty_score_path "${SCORE_TOFU}" \
  --dataset.forget_dataset_name Tofu_forget01 \
  --dataset.retain_dataset_name Tofu_retain01 \
  --dataset.batch_size 2 \
  --dataset.forget_ratio 1.0

SAVE_WMDP="files/models/TinyLlama_v1.1/wmdp_lora"
SCORE_WMDP="files/logs/TinyLlama_v1.1/wmdp_difficulty.json"
mkdir -p "${SAVE_WMDP}" "$(dirname "${SCORE_WMDP}")"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" python exec/Fine_tune_wmdp.py \
  --model_name "${MODEL_ROOT}/TinyLlama_v1.1" \
  --cache_dir "${CACHE_DIR}" \
  --epochs 3 \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --domain all \
  --lr 1e-4 \
  --save_dir "${SAVE_WMDP}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" python exec/unlearn_model.py \
  --overall.model_name "${SAVE_WMDP}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --unlearn.unlearn_method origin \
  --unlearn.task_name wmdp \
  --unlearn.use_lora 1 \
  --unlearn.compute_difficulty_only 1 \
  --unlearn.difficulty_score_path "${SCORE_WMDP}" \
  --dataset.forget_dataset_name WMDPALL \
  --dataset.retain_dataset_name WMDPALL \
  --dataset.batch_size 1 \
  --dataset.forget_ratio 1.0
