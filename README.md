# MedHorizon VideoRAG

面向医学长视频 QA 的可扩展研究工程。当前完整保留 VideoRAG baseline，并新增了与 baseline 隔离的 observation-first 证据图 builder 和确定性 event 图检索器。VLM 验证、多跳回答与 Agent 尚未实现。

## 研究管线边界

仓库现在明确区分两条管线：

| 管线 | 状态 | 目标 |
| --- | --- | --- |
| `baseline` | 已实现、可运行 | 固定 chunk、CLIP/时间检索和 VLM Reader；继续作为可复现对照 |
| `vgent_baseline` | 1 FPS 切片、流式抽帧与结构化 clip 描述已实现 | 复现 VGent 的每 64 个采样帧一组，并对比无上限医学长视频适配与官方采样上限 |
| `medical_graph_rag` | v2.1 observation 图、确定性 event 检索和 v3 pilot 语义假设层已实现 | 在可追溯 observation events 上派生 phase hypothesis、phase boundary 与 instrument type-presence track，再研究医学多跳问答 |

现有 `medrag` CLI 仍然只运行原 baseline。VGent 前期验证使用独立脚本和 `artifacts/vgent_baseline/`；证据图也使用独立实验入口，不改变 baseline CLI。Graph-RAG 的研究假设、阶段计划和工程边界见 [docs/graph_rag_research_plan.md](docs/graph_rag_research_plan.md)，无时间泄漏、多证据区间 QA 的标注与评估协议见 [docs/medical_graph_qa_protocol.md](docs/medical_graph_qa_protocol.md)。

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

## VGent 切片前期验证

当前阶段只复现 VGent 的采样和 clip 分组，不提取实体、不建图、不运行多跳检索。先根据标注时长生成计划清单与统计报告：

```bash
python experiments/validate_vgent_slicing.py \
  --config configs/vgent_baseline.yaml \
  --annotations medhorizon_test.jsonl
```

默认输出为 `artifacts/vgent_baseline/slicing_manifest.jsonl` 和 `artifacts/vgent_baseline/slicing_report.json`。默认 `medical_streaming` 模式保持 1 FPS 且没有全视频 7,200 帧上限；将配置切换为 `official_cap` 时，报告会显示官方上限造成的有效 FPS 下降。两种模式都会记录尾部 partial clip 和官方 `64 × 20` 最小采样帧条件。

真实视频抽帧使用独立的顺序解码命令。首次只处理一个视频，核对时长、帧数、磁盘占用和首/中/尾 clip 后再扩大范围：

```bash
python experiments/extract_vgent_streaming.py \
  --config configs/vgent_baseline.yaml \
  --annotations medhorizon_test.jsonl \
  --video-root /path/to/MedHorizon \
  --limit 1
```

该命令不会把整段视频加载到内存或 GPU；它顺序解码一次并按 1 FPS 写入 `artifacts/vgent_baseline/frames/`。每个视频完成后原子保存独立 manifest，重跑会跳过已完成视频。OpenCV 严重缺帧时会使用 FFmpeg 做一次固定 FPS 后备解码。聚合清单与报告分别写入 `streaming_manifest.jsonl` 和 `streaming_report.json`。由于全量约有数百万张采样帧，不应在未检查单视频空间占用前直接去掉 `--limit`。

Qwen2.5-VL 的 64 帧 clip 描述使用独立服务脚本 `scripts/serve_qwen25vl_vgent64.sh`，不会改变原 16 帧 QA baseline。描述实验只选择帧缓存完整的 clip，并一次输入全部 64 张 1 FPS 帧；每帧在内存中等比例缩小后发送，原始缓存不会被改写：

```bash
# 在 .venv-vllm 环境中启动 64-image 服务，默认端口 8001。
CUDA_VISIBLE_DEVICES=0 ./scripts/serve_qwen25vl_vgent64.sh

# 在项目 .venv 中运行；manifest 是单视频抽帧产生的 JSON。
OPENAI_API_KEY=EMPTY python experiments/describe_vgent_clips.py \
  --config configs/vgent_baseline.yaml \
  --manifest artifacts/vgent_baseline/video_manifests/<video>.json \
  --output artifacts/vgent_baseline/descriptions/<video>.jsonl \
  --errors artifacts/vgent_baseline/descriptions/<video>.errors.jsonl
```

使用 `--all-clips` 可覆盖单视频全部 clip。若尾片段不足 64 帧，脚本只对已完整抽取的合法 partial clip 重复最后一帧补足请求长度，并在输出中记录源帧数与 padding 数量。

若使用 ModelScope 的 Qwen3.5-27B 做限额对比，使用独立配置和脚本，不改变本地 Qwen2.5-VL baseline。该脚本固定选择 20 个完整 clip，关闭 SDK 自动重试和摘要二次改写，并在首个失败后立即停止，使每个 clip 最多发起一次模型请求：

```bash
export MODELSCOPE_ACCESS_TOKEN='你的_ModelScope_Token'
./scripts/run_modelscope_qwen35_27b_vgent20.sh \
  artifacts/vgent_baseline/<cache>/video_manifests/<video>.json
```

也可以直接给 `experiments/describe_vgent_clips.py` 传入 `--clip-indices 0,5,9` 复现实验子集。Token 只从环境变量读取，不得写入配置、脚本、日志或 Git。

对筛选出的 `079`、`047` 和 `grasp_CASE003`，可使用独立批处理脚本顺序完成 1 FPS medical-streaming 抽帧和 Qwen3-VL-235B 全 clip 描述。每个父 clip 保持 64 秒、64 帧和原 clip ID，但按连续前后半段拆为两次 32 帧视觉请求；输出保留带时间边界的 `segment_descriptions`，并提供兼容旧构图代码的确定性事实合并 `description`。partial tail 在每个半段内复用该段最后一帧补齐；视频级和 clip 级均显示进度条，重跑时会跳过完整缓存、已成功父 clip 和已成功子段，并清除已经恢复成功的旧错误记录。

```bash
export AGICTO_API_KEY='运行时提供，不要写入 Git'
./scripts/run_agicto_qwen3vl235b_selected3.sh /path/to/MedHorizon
```

默认抽帧缓存位于 `artifacts/vgent_baseline/streaming_cache_selected3_qwen3vl235b/`，描述输出位于 `artifacts/vgent_baseline/agicto_qwen3vl235b_selected3_observation_first_v10_2x32/<video>/`。也可通过脚本第二、第三个参数覆盖这两个目录。对 429、常见 5xx、超时和连接错误采用较长的指数退避；中断后执行同一命令会跳过已成功 clip。脚本不执行构图或 QA。

生成完成后，可用 `experiments/compare_vgent_descriptions.py` 将该子集与已有完整描述 JSONL 配对，输出规则违规、医学推断、uncertainty、耗时以及逐 clip 摘要。

## v2.1 可追溯证据图

证据图 builder 只使用 `observed_facts` 建图；`medical_inferences` 会保留在原始 clip metadata 中，但不会成为图事实。v2 将 action 映射到有限观察词表，将复合实体描述拆为基础 entity concept 与 mention attributes，并通过显式动作转移表支持 `pass_through -> pull -> tighten` 等连续事件。v2.1 额外为 event 计算可解释的结构支持分数，并选择最多 3 个代表 clip。原始 clip 节点不会被 temporal merge 覆盖或删除。实体跨 clip 只标记为低置信 `possible_continuation`，不会直接断言是同一物理实体。

```bash
python experiments/build_evidence_graph.py \
  --descriptions artifacts/vgent_baseline/<run>/descriptions.jsonl \
  --manifest artifacts/vgent_baseline/<cache>/video_manifests/<video>.json \
  --output-dir artifacts/graph_rag/<video>/evidence_graph_v2_1
```

输出包括：

- `normalized_observations.jsonl`：保留 surface form 的规范化 mention 与 action；
- `temporal_events.jsonl`：相邻 clip 的派生事件分组，包含逐次合并分数、动作关系、共享信息实体、event structural support score 和 representative evidence；
- `evidence_graph.json`：包含 clip、mention、concept、action event、temporal event 及其证据边；
- `graph_report.json`：节点、边、合并率、缺失帧路径和高频概念统计。

每个非 clip 节点通过 `observed_in`、`contains` 或 `part_of` 回溯到 clip；只有 clip 节点保存完整帧路径，避免在图 JSON 中重复数千次相同路径。该版本不包含高层医学推理。

`structural_support_score` 只表示图内连续性、实体/角色一致性与 observation specificity，不是医学事实正确率。singleton 和 merged event 使用不同 `support_mode`。representative evidence 只保存 clip ID、时间区间、覆盖动作/实体和选择理由，帧仍从 clip 节点追溯。可用 `--max-representative-clips` 覆盖默认上限 3。

## 确定性 event 图检索 v1

第一版检索器不调用 LLM 或 embedding。它复用建图时的有限 action/entity vocabulary，将问题规范化后检索 temporal events，并使用动作/实体覆盖、词项稀有度、可见属性、clip 文本、event structural support 和代表证据覆盖率排序。`after` / `before` 问题可沿 `temporal_before` 边扩展一个或多个相邻 event；跨 event 扩展只在补充查询词或满足明确方向上下文时发生。

```bash
python experiments/retrieve_evidence_graph.py \
  --graph artifacts/graph_rag/<video>/evidence_graph_v2_1/evidence_graph.json \
  --question "What happens after the needle-like instrument passes through tissue?" \
  --question-id demo-001 \
  --top-k 5
```

也可通过 `--questions-jsonl` 批量输入包含 `id`、`question` 和可选 `video_id` 的问题，并用 `--output` 写入新的 JSONL。输出包含规范化查询、event group 排名、分数组成、显式 reasoning path、代表 clip 时间区间和原始帧路径。当前分数是确定性检索分数，不是医学正确率；该入口与原 `medrag retrieve` baseline 保持隔离。

小规模 Graph-QA 诊断可让同一个 OpenAI-compatible VLM 先对 observation events 做文本重排，再读取 event onset/representative frames 回答多选题。对原本含明确时间区间的问题，脚本会从模型看到的题干中删除时间文本，但仍把该区间作为视觉取帧依据；无时间题不会使用人工时间窗口。

```bash
OPENAI_API_KEY=EMPTY python experiments/evaluate_graph_qa_qwen25vl.py \
  --annotations medhorizon_test.jsonl \
  --graph artifacts/graph_rag/<video>/evidence_graph_v2_1/evidence_graph.json \
  --video-key <video> \
  --qa-uids 0,1,2,3,4,5 \
  --video-root /path/to/MedHorizon \
  --output-dir artifacts/graph_rag/<video>/<run>
```

该实验会保存清理前后题干、route、Qwen event 排名、证据 clip、帧路径、预测和理由，便于把时间窗口 Reader 与无时间图检索分别诊断。它是研究实验入口，不改变原 baseline QA 协议。

## v3 pilot：阶段边界与器械轨迹语义层

v3 pilot 不修改 v2.1 observation facts，而是消费独立的语义假设 JSONL，派生三类节点：`phase_hypothesis`、`phase_boundary` 和 `instrument_track`。每个节点保留来源模型、置信度和 supporting event；`instrument_track` 仅表示同类器械在相邻事件中的持续出现，不表示已确认的同一物理实例。

可先用 OpenAI-compatible VLM 对每个 temporal event 生成候选受限的语义假设：

```bash
OPENAI_API_KEY=EMPTY python experiments/infer_graph_semantic_hypotheses.py \
  --graph artifacts/graph_rag/<video>/evidence_graph_v2_1/evidence_graph.json \
  --annotations medhorizon_test.jsonl \
  --video-key <video> \
  --output artifacts/graph_rag/<video>/<run>/semantic_hypotheses.jsonl
```

当前 ontology 诊断模式只读取该视频 Phase-Instrument 题干中的目标 phase、器械候选项，以及 Action Recognition 题中公开的手术阶段候选项，不读取答案；因此结果必须标记为 `candidate_aware_diagnostic`，不能作为 question-independent 正式 benchmark。正式实验应改用训练集或外部领域 ontology。

随后增广图并评测阶段起点到器械轨迹的可追溯路径：

```bash
python experiments/augment_semantic_evidence_graph.py \
  --graph artifacts/graph_rag/<video>/evidence_graph_v2_1/evidence_graph.json \
  --hypotheses artifacts/graph_rag/<video>/<run>/semantic_hypotheses.jsonl \
  --output-dir artifacts/graph_rag/<video>/<run>/semantic_graph

python experiments/evaluate_phase_instrument_graph.py \
  --annotations medhorizon_test.jsonl \
  --graph artifacts/graph_rag/<video>/<run>/semantic_graph/semantic_evidence_graph.json \
  --video-key <video> --qa-uids 2,3,4,5 \
  --output-dir artifacts/graph_rag/<video>/<run>/phase_instrument_eval
```

检索路径为 `phase_hypothesis -> phase_boundary -> onset event <- instrument_track`，并默认附带一个紧邻 event 作为粗粒度 64 秒切片的边界上下文。语义节点均标记为医学假设，不会被序列化成直接观察事实。

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

默认 `--scope intra_video`：每道 MedHorizon QA 已给定源视频，因此只在该视频的 chunks 中检索，这是主实验设置。`--scope global` 可作为跨视频检索的困难对照，但不能与视频内指标混合比较。

要验证完整时间/视觉路由而非纯 CLIP 检索，添加 `--retriever hybrid`。显式时间题会走确定性的 `TemporalRetriever`；没有时间表达时才实例化视觉编码器。

```bash
python experiments/evaluate_retrieval.py --config configs/baseline.yaml \
  --index artifacts/index_openai --name hybrid_openai --retriever hybrid \
  --output artifacts/retrieval_hybrid_openai.json \
  --details artifacts/retrieval_hybrid_openai_details.jsonl
```

对于“改写后题干没有时间、但 `question_original` 等旧字段包含时间”的题，可运行单独的
**original-question oracle**。它只替换检索输入为产生 GT 的原始字段；时间 GT、索引和指标不变，
因此只能作为时间路由的上限诊断，不能和真实部署结果混合报告。

```bash
python experiments/evaluate_retrieval.py --config configs/baseline.yaml \
  --annotations medhorizon_test.jsonl --index artifacts/index_openai \
  --name hybrid_original_question_oracle --retriever hybrid \
  --query-source evidence_source --subset rewritten_time_missing \
  --output artifacts/retrieval_hybrid_oracle_127.json \
  --details artifacts/retrieval_hybrid_oracle_127_details.jsonl
```

报告中的 `query_source: evidence_source` 和 `subset: rewritten_time_missing` 用于标识该 oracle 条件。

## 混合检索路由

`HybridRetriever` 会优先解析题干中的明确时间范围或时间点，并按 chunk 元数据确定性返回重叠片段；没有时间表达的问题才加载视觉编码器并在所属视频内检索。可单独验证路由，无需启动 CLIP 来处理时间题：

```bash
medrag retrieve --config configs/baseline.yaml --video-id multibypass_SBP12 \
  --question "What happened from 1:09:18 to 1:10:18?" --top-k 4
```

## 检索后回答（Reader）

`medrag answer` 会读取 MedHorizon 的嵌套 QA 标注，先走 HybridRetriever；再仅对命中的 Top-K
chunk 从原视频密集抽帧（默认每个 chunk 16 帧）并缓存到 `artifacts/reader_frames/`，最后交给
可替换的多选题 Reader。索引阶段的 30 秒 8 帧不会被修改，也不需要重建索引。

先用离线 `mock` Reader 验证数据路径、FFmpeg 和输出格式：

```bash
medrag answer --config configs/baseline.yaml --annotations medhorizon_test.jsonl \
  --limit 10 --output artifacts/qa_mock_10.jsonl

python experiments/evaluate_qa.py --predictions artifacts/qa_mock_10.jsonl \
  --output artifacts/qa_mock_10_report.json
```

`mock` 固定选择第一个选项，因此它只用于冒烟测试，不能作为模型结果。仓库已提供 A6000 的
首个真实模型配置 [reader_qwen25vl.yaml](configs/reader_qwen25vl.yaml) 及两个脚本。首次在服务器执行：

```bash
# 一次性：创建与项目 .venv 隔离的 vLLM 环境
python3 -m venv .venv-vllm
source .venv-vllm/bin/activate
./scripts/setup_vllm_cuda12.sh

# 保持此终端运行；首次会下载 Qwen2.5-VL-7B
./scripts/serve_qwen25vl.sh
```

另开一个终端，进入项目环境并运行第一轮 20 题实验：

```bash
source .venv/bin/activate
./scripts/run_qwen25vl_smoke.sh
```

使用其他 OpenAI-compatible VLM 时，在另一个配置文件中设置：

```yaml
llm:
  provider: openai_compatible
  model: <your-vlm-model>
  api_key_env: OPENAI_API_KEY
  # base_url: http://your-vllm-server/v1  # 本地 vLLM 等兼容服务时设置
  frames_per_chunk: 16
```

### ModelScope 远程 API

无需启动本地 vLLM。仓库提供
[reader_modelscope_qwen3vl8b.yaml](configs/reader_modelscope_qwen3vl8b.yaml)，使用
`Qwen/Qwen3-VL-8B-Instruct` 和 ModelScope 的 OpenAI-compatible API。先在 ModelScope 创建
Token，并在服务器项目环境中设置它：

```bash
export MODELSCOPE_ACCESS_TOKEN='你的_ModelScope_Token'
./scripts/run_modelscope_qwen3vl8b_check.sh
```

单题检查成功后再运行：

```bash
./scripts/run_modelscope_qwen3vl8b_20.sh
```

该远程基线固定为 Top-1、每 chunk 8 帧，以控制上传图像数量与 API 消耗。

### 完整时间窗 Reader 对照

对于显式时间范围题，`temporal_window` Reader 不再只把排名第一的 30 秒 chunk 交给 VLM，
而是直接在题目指定的完整时间范围均匀提取 16 帧；视觉路由题仍按检索 chunk 取帧。它是针对
`Top-1 × 8 帧` 基线的受控改进实验：

```bash
./scripts/run_modelscope_qwen3vl8b_temporal_window_20.sh
```

输出为 `artifacts/qa_qwen3vl8b_temporal_window16_top1_20_report.json`。与旧报告比较时，必须使用
同一前 20 题；新预测 JSONL 的时间题证据 `source` 会标为 `temporal_window`，并保存完整窗口对应的
缓存帧路径。

### Question-only 对照

严格 question-only 模式不会加载索引、不会运行检索，也不会把任何视频帧发送到 API；它只测量
问题与选项带来的语言先验。它应始终使用与视觉实验相同的题目、模型和输出长度：

```bash
./scripts/run_modelscope_qwen3vl8b_question_only_20.sh
```

输出为 `artifacts/qa_qwen3vl8b_question_only_20_report.json`，其中所有预测的路由均为
`question_only`，证据数组为空。

### 远程 API 的重试与断点续跑

所有 `medrag answer --output <结果>.jsonl` 调用均会逐题追加保存。重新执行相同命令时，程序会读取
现有输出并跳过已有成功预测的 `id`；因此无需改变任何 shell script 即可从中断处继续。对
OpenAI-compatible API 的 HTTP 429，会按 `10, 20, 40, 80, 120, 120, 120, 120` 秒退避，最多重试
8 次。其他单题异常不会中止整个任务，失败会记录为与输出相邻的错误日志（例如
`result.jsonl` 对应 `result.errors.jsonl`），之后重跑时该题会自动再次尝试。

每次 QA 运行还会写出 `result.run.json`，记录实际 config 快照、Git commit、模型与帧数、Top-K、
题目范围、依赖版本、起止时间和成功/失败数量。

### 逐题配对与固定评估协议

比较 question-only 与视频条件时，使用逐题配对报告，而不仅是比较两个均值：

```bash
python experiments/compare_qa_predictions.py \
  --left artifacts/qa_qwen3vl8b_question_only_20.jsonl --left-name question_only \
  --right artifacts/qa_qwen3vl8b_temporal_window16_top1_20.jsonl --right-name temporal_window \
  --output artifacts/qwen3vl8b_pairwise_20.json \
  --details artifacts/qwen3vl8b_pairwise_20_details.jsonl
```

`right_only_correct` 表示视频证据帮助答对，`left_only_correct` 表示视频条件反而答错。

固定协议汇总工具将数据划分为 explicit time、implicit time、no reliable time 三组；前两组可接入
检索报告，最后一组只报告端到端 QA：

```bash
python experiments/evaluate_baseline_suite.py --annotations medhorizon_test.jsonl \
  --retrieval hybrid=artifacts/retrieval_hybrid_openai.json \
  --qa question_only=artifacts/qa_qwen3vl8b_question_only_20.jsonl \
  --qa temporal_window=artifacts/qa_qwen3vl8b_temporal_window16_top1_20.jsonl \
  --output artifacts/baseline_suite_report.json
```

汇总会检查传入检索报告中的直接时间题数量是否与当前解析规则一致；若显示
`matches_current_protocol: false`，必须先重新运行检索评估，不能混合比较旧报告与新分区。

然后执行相同的 `medrag answer` 命令。预测 JSONL 会保存每题的答案标签、检索路由、时间证据、
Reader 帧路径和简短理由；`evaluate_qa.py` 输出总体、任务类别和 temporal/visual 路由分组的
多选准确率。

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
