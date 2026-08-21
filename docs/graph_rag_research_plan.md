# 医学长视频 Graph-RAG 研究线

## 定位

本项目包含两条明确分离的研究管线：

1. `baseline`：当前可运行、可复现的固定 chunk VideoRAG，包含 CLIP、时间路由和 VLM Reader。
2. `medical_graph_rag`：面向无显式时间问题的证据建图、多跳检索和医学问答研究线。

Graph-RAG 不替换或改写现有 baseline。只有在小规模实验中验证有效后，才接入正式 CLI。

## 研究问题

核心假设是：与独立 chunk 检索相比，带原视频证据指针的医学事件图能更完整地检索多个相关区间，尤其适用于跨阶段、器械—动作—解剖结构关联、因果和时序顺序问题。

显式时间题不作为 Graph-RAG 的主要成绩，但保留为 Reader oracle 和管线诊断条件。

## 图结构

每个视频单独建图，首阶段不做跨视频知识图谱。节点类型为：

- `segment`：底层视频区间；
- `phase`：较长的手术或检查阶段；
- `action`：细粒度操作或事件；
- `instrument`：器械；
- `anatomy`：解剖结构；
- `finding`：病灶、组织表现、并发情况或检查发现。

关系类型为：

- `temporal_before`、`contains`；
- `uses`、`acts_on`、`causes`；
- `same_entity`、`co_occurs`。

所有节点与关系必须能够追溯到 `video_id + start_seconds + end_seconds`。纯模型生成且没有视频证据的描述不得进入正式图索引。

## 目标管线

1. 多尺度分段；
2. 从短区间提取医学实体、动作、阶段与发现；
3. 合并同一实体并建立时序、层级和医学关系；
4. 将问题分解为可检索的子目标；
5. 初始节点检索与图上多跳扩展；
6. 使用原始视频 clip 验证候选证据；
7. 合并、重排并输出关键证据区间集合；
8. 医学 VLM Reader 基于证据回答。

## 实验阶段

### 阶段 0：冻结现有 baseline

保留当前 CLIP/Temporal/Hybrid 报告、question-only、时间窗口 Reader 和逐题配对协议，不改变已有输出语义。

### 阶段 1：VGent 切片基线验证（已完成单视频前期验证）

默认采用 `medical_streaming`：顺序解码并始终保持 1 FPS，每 64 个采样帧组成约 64 秒 clip，不设置全视频 7,200 帧上限。`official_cap` 仅作为严格复现对照。先基于 annotation duration 统计 clip 数量和 partial clip，再用真实视频核对有效 FPS、时长与解码完整性，不做实体抽取或建图。

随后在小规模真实视频上核对 FFprobe 时长、实际抽帧时间戳和首/中/尾 clip 视觉内容，确认采样协议后再进入建图实验。

### 阶段 1.5：Observation 证据图（v2.1 已实现）

builder 只消费 observation-first `observed_facts`，构建 clip、entity mention、canonical concept、action event 和 temporal event。v2 将动作限制在有限词表内，将颜色、外观、形状、尺寸和材质从基础实体 concept 中拆分到 mention attributes，并支持 `pass_through -> pull -> tighten` 等可审计动作转移。v2.1 为每个 event 新增带分量的 structural support score，并确定性选择最多 3 个代表 clip，用于后续检索排序与 Reader 证据预算。原始 observation 与帧指针保存在 clip 节点；医学推断不进入图事实。跨 clip 实体仅允许 `possible_continuation` 弱关系，避免在缺少视觉跟踪证据时错误合并物理实体。

### 阶段 2：数据协议

构建一批不泄漏时间信息、人工核验且有一个或多个证据区间的问题。优先覆盖多跳、因果、时序顺序、跨阶段和器械—动作关联。

### 阶段 3：通用 Graph-RAG 对照

在小规模视频子集上适配 VGent 风格的实体图、图检索和结构化验证，作为通用 Graph-RAG baseline。

### 阶段 4：医学事件图

加入多尺度节点、医学类型、关系约束、证据验证和可审计多跳路径。不要在没有小规模增益的情况下对全部视频离线建图。

### 阶段 5：全量实验

至少比较：

- 当前 Naive VideoRAG；
- VGent 风格通用 Graph-RAG；
- 医学图检索但不做验证；
- 医学图检索 + 结构化验证；
- oracle evidence + 相同 Reader。

## 工程边界

- 现有 `medrag chunk/index/retrieve/answer` 在 Graph-RAG 接入前保持 baseline 语义。
- VGent 兼容代码放在 `medhorizon_videorag.vgent_baseline`，不得复用 `baseline` 名称或覆盖其 artifacts。
- 当前 VGent 阶段只允许生成切片计划与验证报告，不生成伪图结果。
- Graph-RAG 代码只放在 `medhorizon_videorag.graph_rag` 命名空间。
- Graph-RAG artifacts 使用 `artifacts/graph_rag/`，不得覆盖 baseline artifacts。
- 配置、日志和图中不得保存 token。
- 视频、模型、图索引和实验产物不提交到 Git。
- 当前 `configs/graph_rag_research.yaml` 是研究契约，不是可执行配置。
