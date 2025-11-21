# Unlearning 实验手册

本手册用中文梳理现有代码框架，并给出“样本遗忘难度”评估与接入思路，便于在 TOFU、WHP/WMDP 数据集上复现并扩展遗忘实验（含 LoRA 微调）。

## 代码总览
- **训练/实验入口：`exec/unlearn_model.py`** — 用 fastargs 定义 CLI，汇总 `overall`/`unlearn`/`dataset`/`logger` 等配置，实例化 `Main` 并触发训练或评估。【F:exec/unlearn_model.py†L16-L196】
- **核心流程：`model/unlearn.py`** — 负责加载因果 LM（可选 LoRA）、组装遗忘/保留/测试数据、根据方法构建训练器、运行训练与评估（PPL、few-shot、毒性、版权、TOFU、WMDP），并处理检查点保存/恢复。【F:model/unlearn.py†L25-L385】
- **数据集路由：`dataset/__init__.py`** — 按 `forget_dataset_name`/`retain_dataset_name`/`dataset_seed` 等配置返回 `UnlearnDataset` 及 collator，覆盖 SafePku、C4、Wikitext、TOFU、WMDP 等分割策略。【F:dataset/__init__.py†L13-L114】
- **遗忘算法注册：`unlearn/__init__.py`** — 将命令行的 `unlearn_method` 映射到具体算法实现（FT、GA、CL、KL、RL、NPO+FT 等），由 `model/unlearn.py` 在构建训练器时调用。【F:unlearn/__init__.py†L1-L23】【F:model/unlearn.py†L121-L178】
- **优化/剪枝/日志**：`optim/`、`pruner/`、`loggers/` 提供优化器封装、掩码剪枝及 JSON/本地日志保存等辅助功能。

## 基线运行链路（复现论文实验）
1. **准备配置**：在 CLI 设定 `model_name`、`unlearn_method`、数据集名称/子集（`forget_dataset_name`/`retain_dataset_name`）、训练超参（epoch、lr、batch、accumulation）及可选 SOPHIA/LoRA 开关。【F:exec/unlearn_model.py†L16-L124】
2. **初始化与播种**：`Main.make_config` 收集参数，`setup_seed` 固定随机种子；随后 `init_model`/`init_logger` 构建 `Unlearn` 与 logger。【F:exec/unlearn_model.py†L141-L193】
3. **模型加载 + 可选 LoRA**：`Unlearn.init_model` 用 `AutoModelForCausalLM` 与 `device_map="auto"` 加载；若 `use_lora=True`，按 Q/V 投影注入可训练 adapter 并调整 tokenizer padding。【F:model/unlearn.py†L59-L95】
4. **数据集组装**：`Unlearn.init_dataset` 调用 `get_dataset` 返回遗忘/保留/测试 split 与 collator，计算 `max_steps` 与 `steps_per_epoch` 供调度与日志使用。【F:model/unlearn.py†L97-L119】【F:dataset/__init__.py†L13-L114】
5. **遗忘器创建**：`init_unlearner` 构造 `TrainingArguments`，再通过 `get_unlearn_method` 实例化对应算法（FT/GA/CL/KL/RL 等）。【F:model/unlearn.py†L121-L178】【F:unlearn/__init__.py†L1-L23】
6. **训练与评测**：`run` 统筹训练、保存（含 LoRA 状态）、资源清理，以及后续评测（PPL、few-shot、毒性、版权、TOFU、WMDP）。resume 模式跳过训练只做评估。【F:model/unlearn.py†L318-L385】

## 样本级遗忘难度评估（计划）
目标：在遗忘数据集上，计算每个样本梯度在“一整个遗忘 epoch 的累积梯度”方向上的投影范数，用作难度分数，后续驱动课程式采样。

### 阶段 A：LoRA 微调以节省显存
- 在原配置中开启 `use_lora=True`，保持其余超参与论文一致，适配 4×3090（减少可训练参数且保持推理精度）。【F:model/unlearn.py†L59-L95】【F:exec/unlearn_model.py†L23-L65】
- 可沿用现有 `batch_size`、`gradient_accumulation_steps` 来平衡显存/吞吐。

### 阶段 B：采集遗忘 epoch 梯度
- 复用 `Unlearn.init_dataset` 生成的遗忘 dataloader，使用“未遗忘但已 LoRA 微调”的模型，在遗忘 split 上训练 1 个 epoch，累加得到全局梯度 \(g_{epoch}\)（必要时仅针对 LoRA 权重存储 fp32 版本）。【F:model/unlearn.py†L97-L119】【F:dataset/__init__.py†L13-L114】
- 将 \(g_{epoch}\) 按参数名或 adapter 层保存，便于后续投影。

### 阶段 C：单样本梯度投影与打分
- 对遗忘集中每个样本（单样本 batch，保持 collator 一致）计算梯度 \(g_i\)，并求其在 \(g_{epoch}\) 上的投影范数 \(\|\text{proj}_{g_{epoch}}(g_i)\|\)。
- 将结果写为 JSON/JSONL（包含数据集名、子集、索引或唯一 ID、随机种子、分数），后续直接复用。

### 阶段 D：在遗忘算法中使用分数
- 在构建训练器前（`init_unlearner` 之前），为遗忘数据集增加“基于分数的排序或加权采样”逻辑：读取阶段 C 的 JSON，按得分升序排列或构建加权 Sampler，然后交给现有算法使用。【F:model/unlearn.py†L121-L178】
- 保持 `TrainingArguments` 与算法本身不变，最小化入侵；如需切换开关，可在 CLI 新增布尔/路径参数传递至 `Unlearn`。

### 阶段 E：对比实验与评测
- 设计“有/无难度采样”两组对照，在 TOFU 与 WMDP 遗忘集上分别跑 FT/GA/CL/KL 等算法；其余配置一致以保证公平。
- 复用 `Unlearn.eval` 已有的 PPL、few-shot、毒性、版权、TOFU、WMDP 评测，观察性能差异。【F:model/unlearn.py†L318-L362】

## 代码落地点与开发建议
- **配置扩展**：在 `exec/unlearn_model.py` 增添难度评估/采样相关的开关与文件路径（如 `difficulty_score_path`、`enable_difficulty_sampling`、`difficulty_order`），放入 `unlearn` 或 `dataset` Section，确保通过 `Unlearn` 构造参数传递。【F:exec/unlearn_model.py†L16-L139】【F:exec/unlearn_model.py†L170-L193】
- **梯度计算模块**：新增 `unlearn/difficulty.py`，复用 `Unlearn` 已加载的模型、tokenizer 和遗忘 dataloader，完成阶段 B/C 的梯度累积与投影计算，保持设备与 padding 行为一致。【F:unlearn/difficulty.py†L1-L113】【F:model/unlearn.py†L146-L207】
- **采样整合**：`Unlearn.init_dataset` 会在给定 `difficulty_score_path` 且 `enable_difficulty_sampling=True` 时按分数排序选取遗忘样本（升/降序可选），之后训练器可直接使用排序后的数据集；若仅需打分，可传入 `compute_difficulty_only=1` 只生成 JSON。【F:model/unlearn.py†L97-L207】
- **检查点与日志**：沿用 `Unlearn.save` 与 logger 的路径管理，将 LoRA checkpoint、\(g_{epoch}\) 与难度 JSON 一并保存，便于 resume 与复现实验。【F:model/unlearn.py†L363-L385】【F:exec/unlearn_model.py†L126-L139】
- **可复现实务提示**：记录使用的 `forget_subset`/`retain_subset` 与 `dataset_seed`，确保难度分数与后续遗忘数据顺序一致；TOFU/WMDP 评测对齐 `task_name` 选择以避免指标缺失。【F:model/unlearn.py†L343-L359】【F:exec/unlearn_model.py†L117-L124】

以上梳理可作为快速上手与扩展“样本遗忘难度”实验的路线图。

## 数据获取与缓存
- **自动下载**：除 `files/data/` 下提供的本地 JSON/CSV 外，其余数据均通过 HuggingFace `load_dataset` 自动下载并缓存在 `--cache_dir`（默认 `.cache`）。首次运行需联网；若离线，请事先在同一路径下缓存数据或传入 `custom_split_path`（WMDP 仅）。【F:dataset/wmdp.py†L19-L74】【F:exec/Fine_tune_wmdp.py†L28-L40】【F:exec/Fine_tune_tofu.py†L17-L37】
- **模型权重**：`model_name` 接受 HuggingFace Hub 名称或本地路径；离线时请将权重放入 `cache_dir` 或直接指定本地目录。
- **显存与 dtype**：LoRA 脚本在检测到 GPU 时使用 `torch.bfloat16`，并关闭 `device_map="auto"` 以兼容 `torchrun` DDP，适合 4×3090（24GB）环境。【F:exec/Fine_tune_hp.py†L56-L82】【F:exec/Fine_tune_tofu.py†L43-L70】【F:exec/Fine_tune_wmdp.py†L43-L71】

## torchrun 示例（LoRA 预训练阶段）
以下命令均需在项目根目录执行，`--nproc_per_node=4` 对应 4×3090 DDP 训练，可按需调整 batch/accumulation 控制显存。

### WHP（Harry Potter QA）
```bash
torchrun --nproc_per_node=4 exec/Fine_tune_hp.py \
  --model_name <基座模型或本地路径> \
  --cache_dir .cache \
  --epochs 3 \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --lr 1e-4 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --save_dir files/models/hp_lora
```

### ToFU
```bash
torchrun --nproc_per_node=4 exec/Fine_tune_tofu.py \
  --model_name <基座模型或本地路径> \
  --cache_dir .cache \
  --subset full \
  --epochs 3 \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --lr 1e-4 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --save_dir files/models/tofu_lora
```

### WMDP（可选 cyber/bio/all 域）
```bash
torchrun --nproc_per_node=4 exec/Fine_tune_wmdp.py \
  --model_name <基座模型或本地路径> \
  --cache_dir .cache \
  --domain cyber \
  --subset forget \
  --epochs 3 \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --lr 1e-4 \
  --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --save_dir files/models/wmdp_lora
```
- 若已有本地切分（如自定义 cyber 忘记集），可添加 `--custom_split_path /path/to/wmdp_cyber_split`，`build_dataset` 会直接 `load_from_disk`。

## LoRA → 遗忘难度 → 遗忘训练 的最小流程
1. **LoRA 适配**：按上文 torchrun 命令获得 LoRA checkpoint。
2. **计算遗忘难度**：
   ```bash
   python exec/unlearn_model.py \
     --overall.model_name files/models/<task>_lora \
     --overall.cache_dir .cache \
     --unlearn.task_name <tofu|wmdp|hp> \
     --unlearn.unlearn_method origin \
     --unlearn.use_lora 1 \
     --unlearn.compute_difficulty_only 1 \
     --unlearn.difficulty_score_path files/logs/<run>/difficulty_scores.json \
     --dataset.forget_dataset_name <对应 forget split> \
     --dataset.retain_dataset_name <对应 retain split> \
     --dataset.batch_size 4
   ```
   仅累积遗忘梯度并输出分数，默认跑 1 个 epoch 后退出。
3. **按分数排序进行遗忘**：
   ```bash
   python exec/unlearn_model.py \
     --overall.model_name files/models/<task>_lora \
     --overall.cache_dir .cache \
     --unlearn.task_name <tofu|wmdp|hp> \
     --unlearn.unlearn_method <FT|GA|CL|...> \
     --unlearn.use_lora 1 \
     --unlearn.enable_difficulty_sampling 1 \
     --unlearn.difficulty_score_path files/logs/<run>/difficulty_scores.json \
     --unlearn.difficulty_order asc \
     --dataset.forget_dataset_name <对应 forget split> \
     --dataset.retain_dataset_name <对应 retain split> \
     --dataset.batch_size 4
   ```
   按分数升序优先遗忘易忘样本，可改 `difficulty_order=desc` 先遗忘难样本。

如需在单机多卡上持续运行，保持相同的 `cache_dir` 与 `save_dir`，即可在 LoRA、难度打分与正式遗忘之间复用权重和数据缓存。
