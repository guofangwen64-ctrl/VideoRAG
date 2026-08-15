# MedHorizon VideoRAG

面向医学长视频 QA 的可扩展研究工程。当前实现第一阶段 VideoRAG baseline：视频切片、视觉特征、向量检索、LLM 回答与评估。Agent 和 GraphRAG 预留在架构边界中，尚未实现。

## 安装

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,video]'
```

要运行真实视觉检索编码器，请安装模型依赖：

```bash
pip install -e '.[models]'
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

切片器对每个视频只解码一次并顺序抽帧，采样帧写入 `artifacts/frames/<video_key>/`，不会修改原始视频。`artifacts/chunks.jsonl` 会在每个视频完成后立即追加；中断后以相同命令重跑会跳过已完成视频。无法打开或解码的视频会记录到 `artifacts/chunk_errors.jsonl`，其余视频继续处理。若明确需要全量重跑，添加 `--restart`。

对已记录的失败或严重缺帧视频，可使用 FFmpeg 后备解码并只替换这些视频的 manifest 记录（默认仅重试缺帧比例至少 10% 的视频）：

```bash
medrag chunk --config configs/baseline.yaml --annotations medhorizon_test.jsonl \
  --retry-errors artifacts/chunk_errors.jsonl \
  --errors artifacts/chunk_errors_retry.jsonl
```

重试期间旧 manifest 保持不变；仅成功重试的视频会在结束时被替换。`--retry-min-incomplete-ratio 0` 可选择所有曾有缺帧告警的视频。

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

## 检索评估

使用题干中恢复出的高置信时间区间和时间点，评估视觉索引的 `Recall@K`、最佳时间 IoU 和 Point Hit@K。区间评估默认采用 `IoU ≥ 0.3`：30 秒 chunk 对 60 秒目标区间的最大 IoU 通常为 0.5。

```bash
python experiments/evaluate_retrieval.py \
  --config configs/baseline.yaml \
  --annotations medhorizon_test.jsonl \
  --index artifacts/index \
  --name openai_clip \
  --output artifacts/retrieval_openai_clip.json \
  --details artifacts/retrieval_openai_clip_details.jsonl
```

真实实验时，将 `vision.provider` 切换为实现了 `VisualEmbedder` 的模型适配器，并将 `llm.provider` 替换为你的服务适配器。检索与生成的输入/输出均使用领域对象，不依赖具体模型 SDK。

## 视觉编码器实验顺序

编码器对每个 chunk 的帧独立编码，先 L2 归一化，再均值池化和再次归一化；题目使用同一模型的文本编码器，因此可进行余弦相似度检索。

1. 默认的 `openai_clip`：OpenAI CLIP ViT-B/32，通用图文检索基线。
2. `biomed_clip`：医学/生物医学图文预训练对照。
3. `siglip2`：更强的通用图文检索对照；NaFlex 变体保留原始宽高比。

切换编码器只需替换 `configs/baseline.yaml` 的 `vision` 块：

```yaml
# Experiment 2
vision:
  provider: biomed_clip
  device: cuda
  batch_size: 32

# Experiment 3
vision:
  provider: siglip2
  model_name: google/siglip2-base-patch16-naflex
  device: cuda
  batch_size: 16
```

编码时显示帧级进度条。视觉向量只为至少有一张实际存在的采样帧的 chunk 建立；零帧 chunk 会跳过并记录到 `<index_path>/skipped_chunks.jsonl`，不会中断全量索引。`batch_size` 是跨 chunk 的实际 GPU 帧批量，A6000 可从 256 开始；若 CUDA out-of-memory，依次降为 128、64。

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
