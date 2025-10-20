<h1 align="center">Translate-ZH2EN</h1>
<p align="center">轻量可复现的中→英机器翻译项目（PyTorch Lightning + Hydra + Transformers）</p>
<p align="center">
	<a href="https://huggingface.co/caixiaoshun/translate-zh2en" target="_blank">
		<img alt="HF Model" src="https://img.shields.io/badge/HuggingFace-Model-ffcc4d?logo=huggingface" />
	</a>
	<a href="https://huggingface.co/spaces/caixiaoshun/translate-zh2en" target="_blank">
		<img alt="HF Space" src="https://img.shields.io/badge/HuggingFace-Space-ffcc4d?logo=huggingface" />
	</a>
	<img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python" />
	<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch" />
	<img alt="Lightning" src="https://img.shields.io/badge/Lightning-2.x-792ee5?logo=lightning" />
	<img alt="Hydra" src="https://img.shields.io/badge/Hydra-1.3-89b8ff" />
    <img alt="Transformers" src="https://img.shields.io/badge/Transformers-4.x-ff9a00?logo=huggingface" />
    <img alt="sacreBLEU" src="https://img.shields.io/badge/sacreBLEU-2.x-38b2ac" />
    <img alt="SentencePiece" src="https://img.shields.io/badge/SentencePiece-0.2-blue" />
  
</p>
<p align="center">
	<a href="#-总览">总览</a>
	·
	<a href="#-模型说明">模型说明</a>
	·
	<a href="#-快速开始">快速开始</a>
	·
	<a href="#-评估流程">评估流程</a>
	·
  <a href="#-hugging-face-hub">Hugging Face Hub</a>
</p>

## 📖 总览

这是一个基于 **PyTorch Lightning** 和 **Hydra** 的轻量级中英翻译项目。它提供了一个完整的、可复现的训练和评估流程，旨在作为 Transformer 模型在机器翻译任务中的最小可用示例。

**核心特性**:
- **端到端**: 从数据处理、训练、推理到评估的全流程覆盖。
- **模块化设计**: 使用 PyTorch Lightning 实现代码解耦，易于扩展。
- **灵活配置**: 通过 Hydra 管理所有实验参数，支持命令行覆盖。
- **现代模型**: 内置支持 RoPE（旋转位置编码）的 Transformer 结构。
- **资产开放**: 提供预训练权重、在线 Demo 和评估结果，开箱即用。

---

## 📜 目录
- [**总览**](#-总览)
- [**目录**](#-目录)
- [**功能概述**](#-功能概述)
- [**Hugging Face Hub**](#-hugging-face-hub)
- [**流程总览**](#-流程总览)
- [**模型说明**](#-模型说明)
- [**目录结构**](#-目录结构)
- [**环境与依赖**](#-环境与依赖)
- [**快速开始**](#-快速开始)
  - [准备数据](#1-准备数据jsonl)
  - [准备分词器](#2-准备分词器tokenizer)
  - [训练](#3-训练)
  - [单句推理](#4-单句推理)
- [**评估流程**](#-评估流程)
  - [第一步：批量推理](#第一步批量推理--生成-csv)
  - [第二步：计算指标](#第二步计算指标sacrebleu--chrf--ter)
- [**日志、检查点与可视化**](#-日志检查点与可视化)
- [**参考与致谢**](#-参考与致谢)

---

## ✨ 功能概述

- 使用 Lightning 的模块化训练流程
- 使用 Hydra 进行配置管理与实验追踪（多种实验/模型/回调配置）
- 支持 Transformer 编解码器结构、Rope 位置编码、权重共享词嵌入/输出头
- 简洁的推理脚本：单句 greedy decode、批量推理到 CSV
- 评测脚本：sacreBLEU、chrF 与 TER

---

## 🤗 Hugging Face Hub

本项目在 Hugging Face 上开源了以下资源：

| 模型与文件仓库 (Model & Files) | 在线体验 (Space) |
| :---: | :---: |
| [![Hugging Face Model Card](https://img.shields.io/badge/🤗%20Model-caixiaoshun/translate--zh2en-blue)](https://huggingface.co/caixiaoshun/translate-zh2en) | [![Hugging Face Space Card](https://img.shields.io/badge/🚀%20Space-caixiaoshun/translate--zh2en-yellow)](https://huggingface.co/spaces/caixiaoshun/translate-zh2en) |

### 仓库文件结构

```text
translate-zh2en (HF)
├─ weights/                # 预训练权重 (.pt）
│  ├─ default.pt
│  ├─ absolute-position-embedding.pt
│  ├─ no-position-embedding.pt
│  └─ post-norm.pt
├─ result/                 # 预生成的推理结果（CSV）
│  ├─ translate.csv
│  ├─ translate-absolute-position-embedding.csv
│  ├─ translate-no-position-embedding.csv
│  └─ translate-post-norm.csv
├─ data/                   # 完整数据
│  ├─ translation2019zh_train.json
│  └─ translation2019zh_valid.json
└─ README.md
```

> 不同时间点文件可能更新，建议以 Hugging Face 页面“Files”列表为准。

---

## 🧭 流程总览

<p align="center">
	<img src="assets/pipeline.svg" alt="Pipeline Overview" width="820" />
</p>


---

<a id="model"></a>
## 🧠 模型说明

文件：`src/models/components/mini_translate.py`

- 编码器-解码器 Transformer 结构，支持多头注意力、前馈、LayerNorm
- 使用 RoPE（旋转位置编码）作用在 Q/K 上
- 词嵌入与输出层权重共享
- 解码时采用 greedy 策略

预训练权重下载：

- 代码中四种模型对应的权重，已发布在 Hugging Face 的 weights 目录：
	- https://huggingface.co/caixiaoshun/translate-zh2en/tree/main/weights
	- 文件列表：
		- `default.pt`（基础配置，对应 `model=translate` 或 `experiment=translate`）
		- `absolute-position-embedding.pt`（绝对位置编码，对应 `experiment=translate-absolute-position-embedding`）
		- `no-position-embedding.pt`（无位置编码，对应 `experiment=translate-no-position-embedding`）
		- `post-norm.pt`（Post-Norm 结构，对应 `experiment=translate-post-norm`）
	- 使用时将推理或评估命令中的 `--weight_path` 指向下载的 `.pt` 文件即可。

优化器与调度（见 `src/models/translate_module.py`）：

- `AdamW` + 余弦退火
- loss：`CrossEntropyLoss`

### 性能对比

下表展示了不同模型变体在验证集上的 BLEU、chrF 和 TER 分数。所有分数均通过 `translate_eval.py` 脚本计算得出。

| 模型/设置 | BLEU ↑ | chrF ↑ | TER ↓ |
| :--- | :--- | :--- | :--- |
| **Default (RoPE + Pre-LN)** | **15.47** | **43.99** | **79.38** |
| Post-LN | 1.40 | 14.65 | 92.46 |
| No Position Embedding | 6.71 | 35.24 | 94.47 |
| Absolute Position Embedding | 6.45 | 34.20 | 95.21 |

---

## 🗂️ 目录结构

```text
translate-zh2en
├─ assets/
│  └─ pipeline.svg
├─ configs/
│  ├─ data/
│  │  └─ translate.yaml
│  ├─ model/
│  │  └─ translate.yaml
│  ├─ experiment/
│  │  ├─ translate.yaml
│  │  ├─ translate-absolute-position-embedding.yaml
│  │  ├─ translate-no-position-embedding.yaml
│  │  └─ translate-post-norm.yaml
│  ├─ trainer/
│  │  └─ default.yaml
│  ├─ callbacks/
│  │  └─ default.yaml
│  ├─ paths/
│  │  └─ default.yaml
│  └─ hydra/
│     └─ default.yaml
├─ scripts/
│  ├─ train.sh
│  ├─ infer-translate.sh
│  └─ …
├─ src/
│  ├─ train.py
│  ├─ eval.py
│  ├─ inference.py
│  ├─ batch_inference.py
│  ├─ translate_eval.py
│  ├─ data/
│  │  ├─ translate_datamodule.py
│  │  └─ components/
│  │     └─ translate_data.py
│  └─ models/
│     ├─ translate_module.py
│     └─ components/
│        └─ mini_translate.py
├─ tests/
│  ├─ test_train.py
│  ├─ test_eval.py
│  └─ …
├─ requirements.txt
├─ pyproject.toml
├─ setup.py
└─ README.md
```

---

## 📦 环境与依赖

```bash
conda create -n translate-zh2en python=3.11 -y
conda activate translate-zh2en
pip install -U pip
pip install -r requirements.txt
```


---

## 🚀 快速开始
### 1. 准备数据 (JSONL)

默认配置使用 `data/` 目录下的 `translation2019zh_train.json` 和 `translation2019zh_valid.json`。

文件格式为 **JSON Lines**（每行一个 JSON 对象），且必须包含 `chinese` 和 `english` 两个键：

```json
{"chinese": "今天的天气很好。", "english": "The weather is great today."}
```

> [!TIP]
> 如果你的数据路径或文件名不同，可在运行时通过 Hydra 覆盖：
> `data.train_path=path/to/your/train.json data.val_path=path/to/your/val.json`

**数据获取途径**:

- **a) 从开源仓库获取**:
  数据来源于 [brightmart/nlp_chinese_corpus](https://github.com/brightmart/nlp_chinese_corpus)。

- **b) 运行脚本下载**:
  项目提供了下载脚本，可一键下载并解压到 `data/` 目录。
  ```bash
  bash scripts/download.sh
  unzip dataset.zip -d data
  rm dataset.zip
  ```

- **c) 从 Hugging Face 直接下载**:
  我也将整理好的数据上传到了 Hugging Face，支持 `wget` 直接下载：
  ```bash
  # 创建 data 目录
  mkdir -p data
  
  # 下载训练集和验证集
  wget https://huggingface.co/caixiaoshun/translate-zh2en/resolve/main/data/translation2019zh_train.json -O data/translation2019zh_train.json
  wget https://huggingface.co/caixiaoshun/translate-zh2en/resolve/main/data/translation2019zh_valid.json -O data/translation2019zh_valid.json
  ```

### 2. 准备分词器 (Tokenizer)

本项目默认使用 `google/mt5-small` 的分词器，并期望其位于 `checkpoints/mt5-small` 目录。

你可以运行以下一次性 Python 脚本来预下载并保存分词器文件：

```python
from transformers import AutoTokenizer

print("Downloading tokenizer 'google/mt5-small'...")
tok = AutoTokenizer.from_pretrained('google/mt5-small', fast=False)

save_path = 'checkpoints/mt5-small'
tok.save_pretrained(save_path)
print(f"Tokenizer saved to '{save_path}'")
```

### 3. 训练

使用 `src/train.py` 脚本启动训练。通过 `experiment` 参数可以方便地切换不同的实验配置。

**默认训练**:
使用 `configs/experiment/translate.yaml` 中的默认配置。

```bash
python src/train.py experiment=translate
```

**自定义训练**:
Hydra 允许在命令行中轻松覆盖任何配置项。

- **示例：修改训练参数**
  ```bash
  # 训练 10 个 epoch，batch size 设为 64，学习率改为 3e-4
  python src/train.py experiment=translate trainer.max_epochs=10 data.batch_size=64 model.lr=3e-4
  ```

- **示例：在 CPU 上训练**
  ```bash
  python src/train.py experiment=translate trainer.accelerator=cpu
  ```

**切换模型/实验变体**:
项目预置了多种模型结构和实验配置，可通过 `experiment` 参数直接调用。

- **例如，使用绝对位置编码**:
  ```bash
  python src/train.py experiment=translate-absolute-position-embedding
  ```
- **仅修改模型，保持其他配置不变**:
  ```bash
  # 使用 post-norm 结构的模型，但复用默认实验的其他配置
  python src/train.py model=translate-post-norm
  ```

### 4. 单句推理

`src/inference.py` 是一个独立的推理脚本，不依赖 Hydra，使用 `argparse` 解析参数。

```bash
python src/inference.py \
    --config configs/model/translate.yaml \
    --weight_path "path/to/your/weight.pt" \
    --tokenizer_path checkpoints/mt5-small \
    --text "今天天气不错，我们去公园散步吧。"
```

**输出示例**:
```
Input: 今天天气不错，我们去公园散步吧。
Translation: The weather is nice today. Let's go for a walk in the park.
```

## 📊 评估流程

评估模型性能分为两步：首先进行批量推理生成翻译结果，然后使用评测脚本计算标准指标。

### 第一步：批量推理 → 生成 CSV

此步骤使用 `src/batch_inference.py` 脚本，对验证集或测试集（JSONL 格式）进行批量翻译，并将结果（模型预测 `pred` 和真实标签 `gt`）保存到一个 CSV 文件中。

- **输入**:
  - `--jsonl`: 待翻译的 JSONL 文件路径 (默认为 `data/translation2019zh_valid.json`)。
- **关键参数**:
  - `--config`: 模型配置文件路径。
  - `--weight_path`: 已训练好的模型权重路径。
  - `--tokenizer_path`: 分词器路径。
  - `--batch_size`: 推理时的批处理大小。
- **输出**:
  - `--csv_out`: 保存结果的 CSV 文件路径。

**命令示例**:
```bash
python src/batch_inference.py \
	--jsonl data/translation2019zh_valid.json \
	--csv_out outputs/valid_pred.csv \
	--config configs/model/translate.yaml \
	--weight_path "path/to/your/weight.pt" \
	--tokenizer_path checkpoints/mt5-small \
	--batch_size 32
```

### 第二步：计算指标 (sacreBLEU / chrF / TER)

获得包含预测和参考译文的 CSV 文件后，使用 `src/translate_eval.py` 脚本计算各项翻译指标。

- **输入**:
  - `--csv`: 上一步生成的 CSV 文件路径。
- **输出**:
  - 终端打印样本数、sacreBLEU、chrF 和 TER 分数。

**命令示例**:
```bash
python src/translate_eval.py --csv outputs/valid_pred.csv
```

**输出示例**:
```
====== MT Evaluation (sacreBLEU) ======
Samples: 3000
BLEU: 25.43
chrF: 52.10
TER:  60.12
BLEU signature: BLEU+case.lc+numrefs.1+smooth.exp+tok.13a+version.2.5.1
```


### 定性案例（Qualitative Examples）

| 中文输入 | 模型输出 | Google 翻译 |
| --- | --- | --- |
| 根据《中国疼痛医学发展报告（2020）》的数据显示，我国目前有超过3亿人正在经受慢性疼痛的困扰，慢性疼痛已经成为仅次于心脑血管疾病和肿瘤的第三大健康问题。 | According to the China pain-based report on the latest report, more than 300 million people are suffering from chronic pain, which has become the third health problem after cardiovascular disease and tumor. | According to the "China Pain Medicine Development Report (2020)," over 300 million people in my country currently suffer from chronic pain, making it the third most common health concern after cardiovascular and cerebrovascular diseases and cancer. |
| 祝你生日快乐！希望你喜欢这份礼物。 | Wish you a happy birthday! I hope you like this gift. | Happy birthday! I hope you enjoy this gift. |
| 我们将于明天上午九点开会，请准时参加。 | Please attend the meeting at 9am tomorrow. | We will have a meeting at nine o'clock tomorrow morning. Please attend on time. |


---

## 📈 日志、检查点与可视化

- **Hydra 输出**: 默认情况下，每次运行的输出都会保存在 `logs/` 目录下，并按时间戳生成唯一的文件夹。
  - **训练日志**: `logs/train/runs/<时间戳>/`
  - **检查点**: 训练过程中生成的模型检查点（checkpoints）位于每次运行输出目录的 `checkpoints/` 子文件夹下。默认会保存最后一个 epoch 的权重 (`last.ckpt`) 以及验证集损失 (`val_loss`) 最优的权重。

- **TensorBoard 可视化**:
  当使用 `logger=tensorboard` (这是默认配置) 时，训练过程中的各项指标（如 loss、accuracy）会被记录下来。你可以使用 TensorBoard 查看这些指标的变化曲线。
  ```bash
  tensorboard --logdir logs/train/runs
  ```

---

## 🙏 参考与致谢

- 感谢 [ashleve](https://github.com/ashleve) 提供的优秀模板，极大地简化了项目初始化和配置管理的复杂度。

[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=ashleve&repo=lightning-hydra-template)](https://github.com/ashleve/lightning-hydra-template)


- 感谢[brightmart](https://github.com/brightmart/nlp_chinese_corpus)的开源数据

[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=brightmart&repo=nlp_chinese_corpus)](https://github.com/brightmart/nlp_chinese_corpus)


