# MedHorizon VideoRAG Baseline：工程与实验笔记

> 最后更新：2026-08-17。  
> 范围：MedHorizon 长视频 QA 的第一阶段 VideoRAG baseline；不包含 Agent 或 GraphRAG。

## 1. 目标与边界

本项目的第一阶段目标是建立一个可复现、可扩展的医学长视频 QA 基线：

```text
原始视频
  → 固定时间切片与索引帧
  → 视觉编码与向量索引
  → Temporal / Visual Hybrid Retriever
  → Reader 密集取帧与 VLM 多选回答
  → 检索定位与端到端 QA 评估
```

当前工程不实现 Agent、GraphRAG 或跨视频知识图谱推理。这些方法应在本基线冻结后，以相同的数据划分和评估协议进行比较。

## 2. 已实现组件

| 模块 | 当前实现 | 主要位置 |
|---|---|---|
| 数据读取 | MedHorizon 嵌套 JSONL 展平为视频与 QA 对象 | `datasets/medhorizon.py` |
| 视频切片 | 30 秒窗口、步长 30 秒、每 chunk 8 帧；OpenCV 顺序解码与 FFmpeg fallback | `ingestion/chunker.py` |
| 索引 | OpenAI CLIP ViT-B/32，帧级 L2 norm → chunk 均值池化 → L2 norm；NumPy 向量检索 | `features/openai_clip.py`、`retrieval/numpy_index.py` |
| 时间证据 | 从题干、原始题干等字段恢复 direct range / direct point 证据 | `datasets/temporal_ground_truth.py` |
| 混合检索 | 明确时间问题走 TemporalRetriever；其余问题走 CLIP VisualRetriever | `retrieval/temporal.py`、`retrieval/hybrid.py` |
| Reader | 命中证据密集取帧，调用 OpenAI-compatible VLM，多选题 JSON 输出 | `ingestion/fine_frames.py`、`generation/readers.py` |
| QA 评估 | 总体、任务、路由准确率；question-only、配对比较与协议汇总 | `experiments/evaluate_qa.py`、`experiments/compare_qa_predictions.py`、`experiments/evaluate_baseline_suite.py` |
| 鲁棒运行 | HTTP 429 指数退避、逐题 JSONL 落盘、断点续跑、错误日志、运行元数据 | `generation/readers.py`、`cli.py` |

## 3. 数据与时间证据协议

MedHorizon 测试标注共 1,253 道 QA。当前时间解析规则下，评估分区互斥且总和为 1,253：

| 分区 | 数量 | 定义 | 主指标 |
|---|---:|---|---|
| `explicit_time` | 729 | 改写后的部署题干中直接包含可解析时间 | 检索 IoU / Point Hit + QA accuracy |
| `implicit_time` | 66 | 仅在 `question_original` 等历史字段中可恢复直接时间 | 视觉/Hybrid QA；original-question oracle 单独报告 |
| `no_reliable_time` | 458 | unresolved 或 weak phase anchor | 仅端到端 QA accuracy |

直接时间证据总数为 `729 + 66 = 795`，其中本次检索评估包含：

- 654 道 direct range；
- 141 道 direct point。

注意：时间解析规则曾扩展以支持 `interval 1:55-2:55`、`At 49:12-49:42` 等表达。因此，旧检索报告不能与当前分区混用。`evaluate_baseline_suite.py` 会检查检索报告 direct-time 总数是否与当前协议一致。

## 4. 视频、索引与检索配置

核心设置：

```yaml
chunking:
  duration_seconds: 30.0
  stride_seconds: 30.0
  frames_per_chunk: 8

vision:
  provider: openai_clip
  model_name: ViT-B-32
  pretrained: openai
  device: cuda
  batch_size: 256

retrieval:
  router: hybrid
  index_path: artifacts/index_openai
```

当前 OpenAI CLIP 索引包含 **91,252** 个有效 chunks。无可解码帧的 chunk 会被排除并写入索引目录中的 `skipped_chunks.jsonl`。

### Hybrid Retriever 工作方式

```text
题干存在明确 range / point 时间
  → TemporalRetriever：按 chunk 元数据确定性匹配时间重叠
题干无明确时间
  → VisualRetriever：问题文本嵌入，与所属视频的 CLIP chunk 向量检索
```

在 Reader 阶段：

- 索引阶段仍使用每 30 秒 8 帧；
- 对显式时间范围题，`temporal_window` Reader 直接在完整题目时间窗中均匀抽帧，而不是只查看排名第一的 30 秒 chunk；
- 对视觉检索题，Reader 只对命中的 chunk 额外取帧；
- Reader 帧缓存位于 `artifacts/reader_frames/`。

## 5. 当前正式检索结果：Hybrid + OpenAI CLIP（parser v2）

报告文件：`artifacts/retrieval_hybrid_openai_parser_v2.json`

| 指标 | @1 | @4 | @8 |
|---|---:|---:|---:|
| Range Recall（IoU ≥ 0.3） | 91.74% | 92.05% | 92.35% |
| Range mean best IoU | 0.4819 | 0.4834 | 0.4857 |
| Point Hit | 85.11% | 87.23% | 87.94% |

路由统计：

- `temporal`：729；
- `visual`：66。

解释：729 道部署题干直接带时间，因而被时间路由；另 66 道虽然存在可恢复 GT，但时间只存在于历史题干字段，真实部署输入下仍必须走视觉检索。

该结果验证了时间路由、视频内 chunk 元数据和当前 parser 的可用性。@1 到 @8 增益很小，意味着剩余失败主要不是 Top-K 不足，而可能来自时间解析边界、缺帧/未入库 chunk 或改写题干丢失时间。

## 6. Reader 与 QA 实验

### 6.1 Reader 服务

当前远程 Reader 配置：

```yaml
llm:
  provider: openai_compatible
  model: Qwen/Qwen3-VL-8B-Instruct
  api_key_env: MODELSCOPE_ACCESS_TOKEN
  base_url: https://api-inference.modelscope.cn/v1
```

ModelScope Token 仅通过环境变量提供：

```bash
export MODELSCOPE_ACCESS_TOKEN='...'
```

不要将 Token 写入 YAML、JSONL 或提交至 Git。

### 6.2 前 20 题的受控 Reader 对照

比较条件：

1. `question_only`：不加载索引、不运行检索、不发送视频帧，仅发送问题和选项；
2. `temporal_window`：完整时间窗均匀取 16 帧；视觉问题维持 Top-1 证据 chunk。

两种条件都使用 `Qwen/Qwen3-VL-8B-Instruct`、相同前 20 题、相同温度和输出格式。

| 条件 | 总体准确率 | Action Recognition（16） | Phase-Instrument Association（4） |
|---|---:|---:|---:|
| question-only | 25.0% | 18.75% | 50.0% |
| temporal-window 16 frames | 25.0% | 18.75% | 50.0% |

逐题配对结果：

| 结果 | 数量 | 含义 |
|---|---:|---|
| both correct | 3 | 两种条件均正确 |
| question-only only correct | 2 | 加入视频后由对变错 |
| temporal-window only correct | 2 | 视频证据帮助答对 |
| both wrong | 13 | 两种条件均无法解决 |

结论：视频帧会改变部分预测，但在该 20 题小样本中收益与损失相抵，**净视觉证据增益为 0**。特别是 Action Recognition 中也是 2 题受益、2 题受损。当前主要瓶颈是通用 Qwen3-VL-8B 对细粒度医学手术阶段的理解，而不是切片、时间定位或单纯增加帧数。

这不是对整个数据集的最终结论：20 题样本很小，且没有覆盖 implicit-time 分区。后续 Reader 比较必须保留同一题目集合，并与 question-only 配对报告。

## 7. 运行稳定性与恢复机制

`medrag answer` 已具备以下保证：

- 对 `openai.RateLimitError`（HTTP 429）最多重试 8 次；
- 等待序列为 `10, 20, 40, 80, 120, 120, 120, 120` 秒；
- 每完成一道题立即 append 输出 JSONL，并执行 flush + fsync；
- 重复使用相同 `--output` 时，自动读取已有成功 `id` 并跳过；
- 单题其他异常不会终止整个任务，失败写入 `result.errors.jsonl`，重跑时会再次尝试；
- 每次 QA 调用保存 `result.run.json`，包含 config 快照、Git commit、模型、Top-K、帧数、数据范围、包版本、耗时、成功数与失败数。

现有 shell 脚本的调用方式无需改变。

## 8. 常用命令

### 重跑 parser v2 Hybrid 检索评估

```bash
python experiments/evaluate_retrieval.py \
  --config configs/baseline.yaml \
  --annotations medhorizon_test.jsonl \
  --index artifacts/index_openai \
  --name hybrid_openai_parser_v2 \
  --retriever hybrid \
  --scope intra_video \
  --output artifacts/retrieval_hybrid_openai_parser_v2.json \
  --details artifacts/retrieval_hybrid_openai_parser_v2_details.jsonl
```

### ModelScope Qwen3-VL-8B question-only 对照

```bash
./scripts/run_modelscope_qwen3vl8b_question_only_20.sh
```

### ModelScope Qwen3-VL-8B 完整时间窗对照

```bash
./scripts/run_modelscope_qwen3vl8b_temporal_window_20.sh
```

### 逐题配对分析

```bash
python experiments/compare_qa_predictions.py \
  --left artifacts/qa_qwen3vl8b_question_only_20.jsonl --left-name question_only \
  --right artifacts/qa_qwen3vl8b_temporal_window16_top1_20.jsonl --right-name temporal_window \
  --output artifacts/qwen3vl8b_pairwise_20.json \
  --details artifacts/qwen3vl8b_pairwise_20_details.jsonl
```

### 固定协议汇总

```bash
python experiments/evaluate_baseline_suite.py \
  --annotations medhorizon_test.jsonl \
  --retrieval hybrid=artifacts/retrieval_hybrid_openai_parser_v2.json \
  --qa question_only=artifacts/qa_qwen3vl8b_question_only_20.jsonl \
  --qa temporal_window=artifacts/qa_qwen3vl8b_temporal_window16_top1_20.jsonl \
  --output artifacts/baseline_suite_report_v2.json
```

## 9. Baseline 冻结判断与下一步

VideoRAG v1 的架构已经足够冻结：切片、编码、索引、混合检索、Reader、评估与工程鲁棒性均已闭环。

后续优先级：

1. 使用更强或医学/手术领域适配的 Reader，在**相同题目、相同 16 帧时间窗、相同配对协议**下比较；
2. 以 `question_only` 为必要对照，报告配对的收益/损失，而不仅是平均准确率；
3. 在 Reader 能显示稳定视觉增益后，再比较 BiomedCLIP、SigLIP2 等视觉检索编码器；
4. 最后才将 phase、instrument、anatomy、risk 等实体关系扩展到 GraphRAG。

不建议继续仅靠增加抽帧数量或 Top-K 来优化当前 Qwen3-VL-8B Reader；现有受控实验没有显示这会带来净收益。
