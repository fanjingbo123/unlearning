#!/bin/bash
set -euo pipefail

# 说明：
# - 模型与数据均从本地读取，不会触发在线下载；默认路径：MODEL_ROOT=../base_model、LOCAL_DATA_DIR=../data。
# - 脚本按模型/数据集逐条列出 LoRA 微调 → 遗忘难度打分 → GA 遗忘（有/无排序对照）的完整命令，可按需复制运行。
# - 默认使用多卡 DDP（torchrun），如需单卡可把 torchrun 前缀去掉或将 NPROC_PER_NODE 调整为 1。

MODEL_ROOT="../base_model"
LOCAL_DATA_DIR="../data"
CACHE_DIR=".cache"
NPROC_PER_NODE=4

########################################
# Phi-3-mini-4k-instruct on WHP (HP)
########################################
SAVE_HP="files/models/Phi-3-mini-4k-instruct/hp_lora"
SCORE_HP="files/logs/Phi-3-mini-4k-instruct/hp_difficulty.json"
LOG_HP_BASE="files/logs/Phi-3-mini-4k-instruct/ga_hp_base"
LOG_HP_SORT="files/logs/Phi-3-mini-4k-instruct/ga_hp_sorted"
mkdir -p "${SAVE_HP}" "$(dirname "${SCORE_HP}")" "${LOG_HP_BASE}" "${LOG_HP_SORT}"

# 1) LoRA 微调（DDP）
LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/Fine_tune_hp.py \
  --model_name "${MODEL_ROOT}/Phi-3-mini-4k-instruct" \
  --cache_dir "${CACHE_DIR}" \
  --epochs 3 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --lr 1e-4 \
  --save_dir "${SAVE_HP}"

# 2) 仅计算遗忘难度分数（不训练）
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

# 3a) GA 遗忘（无排序对照，DDP）
LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/unlearn_model.py \
  --overall.model_name "${SAVE_HP}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --overall.save_dir "${LOG_HP_BASE}" \
  --unlearn.unlearn_method GA \
  --unlearn.task_name copyright \
  --unlearn.use_lora 1 \
  --dataset.forget_dataset_name HP \
  --dataset.retain_dataset_name HP \
  --dataset.batch_size 2 \
  --dataset.forget_ratio 1.0

# 3b) GA 遗忘（按难度升序，先遗忘易样本，DDP）
LOCAL_DATA_DIR="${LOCAL_DATA_DIR}" torchrun --nproc-per-node=${NPROC_PER_NODE} exec/unlearn_model.py \
  --overall.model_name "${SAVE_HP}" \
  --overall.cache_dir "${CACHE_DIR}" \
  --overall.logger json \
  --overall.save_dir "${LOG_HP_SORT}" \
  --unlearn.unlearn_method GA \
  --unlearn.task_name copyright \
  --unlearn.use_lora 1 \
  --unlearn.enable_difficulty_sampling 1 \
  --unlearn.difficulty_order asc \
  --unlearn.difficulty_score_path "${SCORE_HP}" \
  --dataset.forget_dataset_name HP \
  --dataset.retain_dataset_name HP \
  --dataset.batch_size 2 \
  --dataset.forget_ratio 1.0

############################################################
# 其他模型/数据集：按需参考以上模板替换 model/task/path
############################################################
# - Qwen2.5-3B-Instruct: 替换模型路径为 "${MODEL_ROOT}/Qwen2.5-3B-Instruct"，保存与日志目录改为 Qwen 前缀。
# - TinyLlama_v1.1: 替换模型路径为 "${MODEL_ROOT}/TinyLlama_v1.1"，保存与日志目录改为 TinyLlama 前缀。
# - ToFU/WMDP 数据：将 Fine_tune_hp.py 替换为 Fine_tune_tofu.py / Fine_tune_wmdp.py，
#   并把 dataset.forget_dataset_name / retain_dataset_name 改成对应的 Tofu_forgetXX / Tofu_retainXX 或 WMDPALL。
# - 若需要难度排序，只需在 GA 运行命令中保留 enable_difficulty_sampling=1、difficulty_order=asc，并指向相同的 difficulty_score_path。
