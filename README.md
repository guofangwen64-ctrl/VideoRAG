# MedHorizon VideoRAG

面向医学长视频 QA 的可扩展研究工程。当前实现第一阶段 VideoRAG baseline：视频切片、视觉特征、向量检索、LLM 回答与评估。Agent 和 GraphRAG 预留在架构边界中，尚未实现。

## 安装

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,video]'
```

## 数据约定

标注为 JSONL，每行：

```json
{"id":"mh_001", "video_id":"case_001", "video_path":"data/videos/case_001.mp4", "question":"何时开始缝合？", "answer":"...", "choices":["..."], "metadata":{}}
```

`video_path` 可以是相对项目根目录的路径。视频切片会写入 `artifacts/chunks.jsonl`，索引写入 `artifacts/index/`。

## 快速运行

```bash
# 1. 生成视频片段清单（需安装 video 依赖）
# baseline 使用不重叠的固定 30 秒窗口，最后不足 30 秒的片段也会保留。
# 先在 configs/baseline.yaml 的 data.video_root 中填写服务器视频根目录；
# 标注中 tmvp/087.mp4 会映射为 ${video_root}/tmvp/087.mp4
medrag chunk --config configs/baseline.yaml --annotations medhorizon_test.jsonl

# 2. 建立特征与向量索引（默认 deterministic，适合管线验证）
medrag index --config configs/baseline.yaml --chunks artifacts/chunks.jsonl

# 3. QA 推理和评估
medrag answer --config configs/baseline.yaml --annotations data/medhorizon/val.jsonl --output artifacts/predictions.jsonl
medrag evaluate --predictions artifacts/predictions.jsonl
```

也可用 `--video-root /path/to/videos` 临时覆盖配置，便于切换数据挂载点。

## 数据集分析

MedHorizon 的原始标注采用“每行一个视频、内嵌 QA 列表”的 JSONL 格式。可生成终端摘要和机器可读报告：

```bash
python experiments/analyze_dataset.py --annotations medhorizon_test.jsonl --output artifacts/dataset_report.json
```

## 时间证据恢复

公开 QA 标注不含独立的证据起止时间字段。恢复脚本仅将题干中直接出现的区间或时间点标为高置信证据；同视频内通过阶段识别题匹配得到的窗口标为 `phase_anchor`（弱锚点），不能作为严格的检索 GT。

```bash
python experiments/recover_temporal_ground_truth.py --annotations medhorizon_test.jsonl
```

默认输出为 `artifacts/recovered_temporal_evidence.jsonl` 和 `artifacts/temporal_recovery_report.json`。

真实实验时，将 `vision.provider` 切换为实现了 `VisualEmbedder` 的模型适配器，并将 `llm.provider` 替换为你的服务适配器。检索与生成的输入/输出均使用领域对象，不依赖具体模型 SDK。

## 目录

```text
src/medhorizon_videorag/
  ingestion/       视频解码、时间切片
  features/        视觉特征接口与实现
  retrieval/       向量索引、检索器
  generation/      LLM 接口与提示词构造
  evaluation/      QA 指标
  pipelines/       阶段编排
  core/            配置、数据模型、端口定义
```
