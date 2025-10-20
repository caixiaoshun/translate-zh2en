## 🇨🇳 中文 README｜translate-zh2en

一个基于 PyTorch Lightning + Hydra 的中英互译（以中译英为主）最小可用示例。包含完整的数据管线、可配置的 Transformer 结构（支持 RoPE）、训练与验证、单句推理、批量推理与 sacreBLEU/chrF/TER 评估。

本项目以 Linux/macOS 的 Bash 为示例命令，亦可在其他平台运行。

---

## 目录

- 功能概述
- 目录结构
- 环境与依赖
- 快速开始
	- 数据准备（JSONL 格式）
	- 下载或准备分词器（Tokenizer）
	- 训练（Hydra 配置）
	- 评估（Hydra）
	- 单句推理
	- 批量推理与自动评测
- 配置说明（Hydra）
- 模型说明（Mini Translate + RoPE）
- 日志、检查点与可视化
- 常见问题（FAQ）
- 致谢与参考

---

## 功能概述

- 使用 Lightning 的模块化训练流程（`LightningModule`/`LightningDataModule`）
- 使用 Hydra 进行配置管理与实验追踪（多种实验/模型/回调配置）
- 支持 Transformer 编解码器结构、Rope 位置编码、权重共享词嵌入/输出头
- 简洁的推理脚本：单句 greedy decode、批量推理到 CSV
- 评测脚本：sacreBLEU、chrF 与 TER

---

## 目录结构

仅列出核心：

- `src/train.py`：训练入口（Hydra）
- `src/eval.py`：评估入口（Hydra，需要传 `ckpt_path`）
- `src/inference.py`：单句推理脚本（独立 argparse）
- `src/batch_inference.py`：批量推理生成 CSV（独立 argparse）
- `src/translate_eval.py`：对 CSV 进行 sacreBLEU/chrF/TER 评测
- `src/data/translate_datamodule.py`：LightningDataModule（组 batch、mask 等）
- `src/data/components/translate_data.py`：数据读取（JSONL：`{"chinese":..., "english":...}`）与分词
- `src/models/components/mini_translate.py`：最小翻译模型（含 RoPE、Encoder/Decoder）
- `src/models/translate_module.py`：LightningModule（loss/metrics/optimizer/schedule）
- `configs/`：Hydra 配置（data/model/trainer/callbacks/experiment 等）

---

## 环境与依赖

建议：

- Python 3.10/3.11（与 PyTorch 版本匹配）
- GPU（CUDA）可用则更快，CPU 也可运行但速度较慢

核心依赖（见 `requirements.txt`）：

- torch 2.9.0，lightning 2.5.5，torchmetrics，transformers，datasets，sacrebleu，sentencepiece
- hydra-core，tensorboard

安装步骤（Bash）：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

若安装 PyTorch 报错，请根据本机 CUDA 版本到官方指引选择命令安装后，再执行其余依赖安装。

---

## 快速开始

### 1) 准备数据（JSONL）

默认配置使用：

- 训练集：`data/translation2019zh_train.json`
- 验证/测试：`data/translation2019zh_valid.json`

文件格式为 JSON Lines（每行一个 JSON 对象），键必须包含：

```json
{"chinese": "今天的天气很好。", "english": "The weather is great today."}
```

如你的路径或文件名不同，可通过 Hydra 覆盖：`data.train_path=... data.val_path=...`

### 2) 准备分词器（Tokenizer）

默认使用目录：`checkpoints/mt5-small`。你可以将 Hugging Face 的 `google/mt5-small` 下载到该目录。例如运行一次性预下载脚本（会把分词器文件缓存到指定目录）：

```bash
python - <<'PY'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('google/mt5-small')
tok.save_pretrained('checkpoints/mt5-small')
print('saved to checkpoints/mt5-small')
PY
```

如果你有自己的词表/分词器，只需保证 `src/data/components/translate_data.py` 与 `src/inference.py` 中的 `tokenizer_path` 指向它，并具有有效的 `pad_token_id` 与 `eos_token_id`。

### 3) 训练

使用默认实验配置（`configs/experiment/translate.yaml`）：

```bash
python src/train.py experiment=translate
```

常见覆盖项（示例）：

- 训练轮数：`trainer.max_epochs=10`
- batch 大小：`data.batch_size=64`
- 学习率：`model.lr=3e-4`
- 使用 CPU：`trainer.accelerator=cpu`

```bash
python src/train.py experiment=translate trainer.max_epochs=10 data.batch_size=64 model.lr=3e-4
```

模型/实验变体：

- 绝对位置编码版本：`experiment=translate-absolute-position-embedding`
- 仅修改模型配置（不换 experiment）：`model=translate`（或其他模型 yaml）

### 4) 评估（Hydra）

`src/eval.py` 需要提供训练得到的 `ckpt_path`。默认 `eval.yaml` 使用的是示例 MNIST，需要覆盖为翻译配置：

```bash
python src/eval.py data=translate model=translate ckpt_path="logs/train/translate/default/<时间戳>/checkpoints/last.ckpt"
```

你也可以指定更细的 trainer 或 logger 覆盖项。

### 5) 单句推理

`src/inference.py` 是独立 argparse 脚本：

```bash
python src/inference.py --config configs/model/translate.yaml --weight_path "logs/train/translate/default/<时间戳>/checkpoints/last.ckpt" --tokenizer_path checkpoints/mt5-small --text "今天天气不错，我们去公园散步吧。" --max_src_len 128 --max_tgt_len 128
```

输出示例：

```
result: Let's take a walk in the park. ...
```

### 6) 批量推理与自动评测

第一步：对验证集 JSONL 进行批量翻译，生成 `CSV`（两列：`pred`、`gt`）。

```bash
python src/batch_inference.py --jsonl data/translation2019zh_valid.json --csv_out outputs/valid_pred.csv --config configs/model/translate.yaml --weight_path "logs/train/translate/default/<时间戳>/checkpoints/last.ckpt" --tokenizer_path checkpoints/mt5-small --max_src_len 128 --max_tgt_len 128 --batch_size 32
python src/translate_eval.py --csv outputs/valid_pred.csv
第二步：使用 sacreBLEU/chrF/TER 评测：

```powershell
python src/translate_eval.py --csv outputs/valid_pred.csv
```

示例输出：

```
====== MT Evaluation (sacreBLEU) ======
Samples: 3000
BLEU: 25.43
chrF: 52.10
TER:  60.12
BLEU signature: BLEU+case.lc+numrefs.1+smooth.exp+tok.13a+version.2.5.1
```

---

## 配置说明（Hydra）

- 训练入口 `src/train.py` 默认加载 `configs/train.yaml`，其中通过 `defaults` 组合：
	- `data: translate` → `configs/data/translate.yaml`
	- `model: translate` → `configs/model/translate.yaml`
	- `callbacks: default`、`trainer: default` 等
	- 你可以用 `experiment=...` 快捷切换一组组合过的超参（如 `configs/experiment/translate.yaml`）。
- 推理脚本 `src/inference.py` 只需要模型子配置：`--config configs/model/translate.yaml`（会从 `net` 节点构建模型）。
- 常用可覆盖项：
	- 数据：`data.train_path`、`data.val_path`、`data.tokenizer_path`、`data.batch_size`、`data.max_src_len`、`data.max_tgt_len`
	- 模型：`model.net.encoder_layers`、`model.net.decoder_layers`、`model.net.embed_dim`、`model.net.num_heads`、dropout 等
	- 训练器：`trainer.max_epochs`、`trainer.accelerator=cpu|gpu`、`trainer.check_val_every_n_epoch` 等

---

## 模型说明（Mini Translate + RoPE）

文件：`src/models/components/mini_translate.py`

- 编码器-解码器 Transformer 结构，支持多头注意力、前馈、LayerNorm（自实现）
- 使用 RoPE（旋转位置编码）作用在 Q/K 上
- 词嵌入与输出层权重共享（`self.head.weight = self.embedding.weight`）
- 解码时采用 greedy 策略（示例脚本可替换为 beam search）

优化器与调度（见 `src/models/translate_module.py`）：

- `AdamW` + 余弦退火（warmup 比例可配 `model.warmup_ratio`，最小 lr 比例 `model.min_lr_rate`）
- loss：`CrossEntropyLoss`（忽略 label 为 `-100` 的 pad 部分）

---

## 日志、检查点与可视化

- Hydra 输出目录模式：`configs/hydra/default.yaml`
	- 训练日志目录：`logs/train/<tags>/.../<时间戳>/`
	- 回调中的检查点保存在：`<output_dir>/checkpoints/`，默认保存 last 与按 `val_loss` 最优
- TensorBoard（当使用 `experiment=translate` 时默认启用 TB Logger）：

```bash
tensorboard --logdir logs
```

---

## 常见问题（FAQ）

1) Torch 安装失败或 CUDA 不匹配？

- 请到 PyTorch 官网选择与你 CUDA 版本匹配的命令安装，然后再 `pip install -r requirements.txt`。

2) 评估脚本 `src/eval.py` 默认是 MNIST 配置？

- 这是模板默认值，需要在命令行覆盖为翻译配置：`data=translate model=translate`。

3) 推理时提示缺少 `pad/eos`？

- 请确认你的分词器目录正确、且包含有效的 `pad_token_id` 和 `eos_token_id`。可用 mT5 分词器（如 `google/mt5-small`）。

4) 自定义数据格式？

- 请确保是 JSONL，每行包含键 `chinese` 与 `english`，并去除空行/空字段。

5) 想在 CPU 上调试？

- 训练：`trainer.accelerator=cpu`；推理脚本会自动检测设备（优先 CUDA）。

6) 想提升速度？

- 可尝试将 `model.compile=true`（默认已打开，需 PyTorch 2.0+），并适当减少 `max_src_len/max_tgt_len`、减小层数/隐藏维度、使用更小 batch。

---

## 参考与致谢

- PyTorch Lightning, Hydra, Hugging Face Transformers
- sacreBLEU, chrF, TER 指标实现

如有问题或建议，欢迎提交 Issue 或 PR！

