# 医学长视频证据图：当前实现与实验状态

> 状态日期：2026-08-27
>
> 功能代码基线：`c1da6e4`（本文档提交前）
>
> 当前主试验视频：`087`
>
> 说明：本文记录已经实现的建图、阶段推断、器械外观轨迹和图证据 Reader。所有路径均相对于仓库根目录；实验 artifacts、视频、模型和访问凭据不进入 Git。

## 1. 当前定位

项目保留两条彼此隔离的管线：

- 原 VideoRAG baseline：30 秒固定 chunk、CLIP/Temporal/Hybrid 检索和 VLM Reader，继续作为稳定对照。
- Medical Graph-RAG：面向没有明确时间提示的医学长视频问题，研究结构化证据表示、阶段定位、图上候选轨迹检索和多区间推理。

当前 Graph-RAG 仍是**单视频证据图原型**，没有实现 Agent、跨视频知识图谱或完整的通用多跳推理。现阶段重点是验证以下闭环是否成立：

```text
1 FPS / 64 秒 clip
        ↓
Qwen3-VL-235B observation-first 描述
        ↓
实体、属性、动作规范化
        ↓
时间事件合并与代表证据选择
        ↓
阶段假设 + 阶段边界 + 器械外观轨迹
        ↓
图检索候选轨迹 + 代表帧 + QA 选项
        ↓
VLM Reader 重排并回答
```

## 2. 长视频 observation 输入层

### 2.1 Medical streaming 切片

为避免 VGent 类输入长度限制无法覆盖数小时手术视频，已采用 `medical_streaming`：

- 视频按 1 FPS 顺序采样；
- 每 64 个采样帧组成约 64 秒 clip；
- 长视频被流式拆成任意数量的 clip，不限制完整视频必须小于 7,200 秒；
- 末尾不足 64 帧时保留真实 source frame 数，并只在模型输入侧重复末帧补齐；
- clip、时间范围和原始帧路径均写入 manifest，供后续图节点追溯。

视频 `087` 共得到 88 个 clip，其中 87 个完整 clip、1 个 partial clip，覆盖 5,628 秒真实视频内容。

### 2.2 Observation-first 描述

当前建图输入来自 Qwen3-VL-235B，使用 observation-first JSON 协议。核心约束是：

- `summary` 只能包含直接可见内容；
- `observed_facts` 只保存视觉可验证的实体、动作、状态变化和证据；
- 医学解释只能进入 `medical_inferences`，不进入图事实；
- 优先使用 `tissue`、`thread-like material`、`needle-like instrument`、`red fluid` 等保守外观词；
- 不利用视频标题、数据集元数据、相邻 clip 或预期术式补充判断；
- 证据不足时保留 `uncertainties`，不强行识别解剖结构、器械或手术阶段。

视频 `087` 的 88 个 clip 均已生成符合 schema 的描述。当前图使用的描述目录为：

`artifacts/vgent_baseline/agicto_qwen3vl_235b_instruct_vgent88_observation_first_v10_087/`

该层提供的是模型观察结果，不等同于人工视觉真值。

## 3. Observation evidence graph v2.1

核心实现：

- `src/medhorizon_videorag/graph_rag/evidence_builder.py`
- `src/medhorizon_videorag/graph_rag/schemas.py`
- `experiments/build_evidence_graph.py`

正式产物：

`artifacts/graph_rag/087/evidence_graph_v2_1_qwen3vl235b_observation_v10/`

### 3.1 节点与关系

| 节点类型 | 含义 | `087` 数量 |
|---|---|---:|
| `segment` | 64 秒底层 clip，保存时间与帧指针 | 88 |
| `entity_mention` | 某个 clip 中直接观察到的实体提及 | 820 |
| `concept` | 规范化后的基础视觉实体概念 | 48 |
| `action_event` | 单条规范化 subject-action-target 观察 | 296 |
| `temporal_event` | 跨相邻 clip 合并的连续事件 | 28 |

主要边包括：

- `observed_in`：mention/action 指向来源 clip；
- `instance_of`：mention 指向规范 concept；
- `has_subject`、`acts_on`：动作角色；
- `part_of`、`contains`：action、clip 与 temporal event 的层级关系；
- `possible_continuation`：保守表示相邻观察可能连续，不断言同一物理实例；
- `temporal_before`：clip、action 或 event 的时间先后关系。

`087` 的 v2.1 图共有 1,280 个节点和 3,580 条边。

### 3.2 Action normalization

自由文本动作已映射到有限动作词表，例如：

- `inserts`、`insert_into` → `insert`；
- `loops_around_tissue`、`forms_loops_around_tissue` → `loop_around`；
- 相近的拉动、握持、穿过、压迫等表达分别归入固定 canonical action。

当前词表实际覆盖 `apply`、`attach`、`contact`、`cut`、`deliver`、`emit`、`grasp`、`guide`、`hold`、`insert`、`loop_around`、`manipulate`、`move`、`pass_through`、`pierce`、`position`、`press`、`pull`、`push`、`remove`、`tighten` 等动作；`087` 中 `other_action` 为 0。

### 3.3 Concept = entity + attributes

图不再把 `yellowish_fatty_looking_material` 一类复合短语直接作为 concept。当前表示方式为：

```text
base concept: generic_material
attributes:
  color: yellow
  appearance: fatty-looking
```

颜色、形状、材质、尺寸、表面外观和标记与基础实体分离，既降低 concept 碎片化，也保留后续检索所需的视觉区别。

### 3.4 Temporal event merging

已支持两类合并：

- 相同动作、角色和实体连续出现时的 exact-action merge；
- 主体/目标实体连续且动作满足允许转移规则时的 action-transition merge。

因此可以把 `pass_through → pull → tighten` 这类动作不同但实体持续一致的序列合并成同一 temporal event。`087` 中产生 38 次 exact-action merge 和 22 次 action-transition merge；28 个 temporal event 中有 20 个跨多个 clip，最长为 5 个 clip。

这仍是保守的结构合并，不声明跨 clip 的器械或组织必然是同一物理对象。

### 3.5 Event-level support score

每个 temporal event 都包含 `structural_support_score`，由以下可审计分量组成：

- action transition 支持；
- entity continuity；
- subject/target role consistency；
- observation specificity；
- action-argument specificity。

它用于候选排序，不是医学正确率或校准概率。`087` 上 28 个 event 的均值为 0.5798；20 个 merged event 的均值为 0.6154。

### 3.6 Event-level representative evidence

每个 temporal event 确定性选择最多 3 个代表 clip，覆盖 primary、transition、terminal 或 supporting 角色，并保存：

- clip ID 与时间范围；
- 代表动作和实体；
- 选择分数与选择原因；
- 原始帧路径。

`087` 上平均每个 event 选择 2.25 个代表 clip，平均 action coverage 为 0.9714，平均 entity coverage 为 0.9896。代表证据是 query-independent 的压缩层，具体问题仍需二次重排。

## 4. 确定性事件图检索

已实现首版不依赖 LLM/embedding 的事件检索器：

- 复用 action/entity normalization 解析查询；
- 以 IDF 降低高频通用实体的权重；
- 综合 action、entity、attribute、clip lexical match、event support 和 representative coverage；
- 明确 `before` / `after` 关系时沿 `temporal_before` 做定向扩展；
- 输出 event、clip、时间、帧路径、分数分解和 reasoning path。

实现入口：

- `src/medhorizon_videorag/graph_rag/retrieval.py`
- `experiments/retrieve_evidence_graph.py`

该检索器证明了 event → clip → frame 的可追溯链路，但尚不能视为完整医学多跳推理器。

## 5. 序列级阶段推断

单个 64 秒 clip 难以可靠判断手术阶段，因此在完整 observation 序列上增加了两阶段推断：

1. **开放活动分段**：只依据按时间排列的观察描述，把全视频切成活动连续的长区间，不预先要求输出某个医学阶段名。
2. **严格阶段映射**：再把活动区间与候选 phase ontology 对齐；泛化的穿线、拉动或组织操作不足以支持具体阶段时，必须输出 `unknown`。

这一阶段不重新生成 clip 描述，也不重新输入全部 64 帧；它消费现有 observation 序列。实现入口：

- `src/medhorizon_videorag/graph_rag/sequence_phase.py`
- `experiments/infer_two_stage_sequence_phases.py`

`087` 的结果为：

| 项目 | 结果 |
|---|---:|
| 开放活动区间 | 22 |
| 接受的命名阶段映射 | 2/22 |
| `unknown` 活动区间 | 20/22 |
| `unknown` clips | 79/88 |
| Pericardial Hemostasis Suturing clips | 7/88 |
| Aortic Clamping clips | 2/88 |
| 投影后 `unknown` temporal events | 23/28 |
| 投影后命名阶段 temporal events | 5/28 |

产物目录：

`artifacts/graph_rag/087/sequence_phase_two_stage_qwen3vl235b_candidate_aware_20260825/`

其中包含 `open_activity_segments.json`、`sequence_phase_segments.json`、`event_phase_hypotheses.jsonl`、两阶段原始响应和运行 metadata。

当前 phase ontology 来自该视频测试问题中的候选阶段，但没有读取答案，因此结果只能称为 **candidate-aware diagnostic**。正式 benchmark 必须冻结来自训练集或外部医学资源的 ontology。

## 6. Semantic evidence graph v3 pilot

序列级阶段结果已与 v2.1 observation graph 合并。合并采用 overlay 方式，不修改底层 observation facts。

新增节点：

- `phase_hypothesis`：具有来源、置信度和 supporting events 的医学阶段假设；
- `phase_boundary`：阶段 onset/end 边界；
- `instrument_track`：由相邻器械外观 mention 归并的观察级轨迹。

新增关系：

- `has_boundary`、`grounded_by`：阶段与边界/事件的证据联系；
- `derived_from`：appearance track 指向来源 mention；
- `visible_during`：appearance track 与 event 的共现；
- `co_occurs`：track 与 phase hypothesis 的共现。

### 6.1 不补人工器械标签的 appearance tracks

为避免在不知道真实器械名称时伪造标签，轨迹从 `visible_instruments` 的外观 mention 构建，保存：

- `canonical_instrument = unknown`；
- `surface_forms`；
- `appearance_signature`：颜色、形状、材质、外观、尺寸、标记；
- `action_roles`；
- supporting event/mention IDs；
- `physical_identity_confirmed = false`；
- `fact_status = derived_observation_track`。

因此这里的 track 表示“某类外观器械在一段时间内持续出现”，而不是“确认追踪到同一把持针器”。医学器械名称只允许在最终 Reader 结合画面和 QA 选项时判断。

`087` 合并图统计：

| 新语义节点 | 数量 |
|---|---:|
| `phase_hypothesis` | 2 |
| `phase_boundary` | 4 |
| appearance `instrument_track` | 67 |

67 条器械轨迹均保持未知医学身份，且均未声明物理实例一致性。

合并图：

`artifacts/graph_rag/087/sequence_phase_two_stage_qwen3vl235b_candidate_aware_20260825/combined_semantic_graph/semantic_evidence_graph.json`

## 7. Phase-Instrument 图检索与 Reader

### 7.1 基本路径

已经实现以下候选生成与重排路径：

```text
phase hypothesis
    → onset boundary
    → onset/context temporal events
    ← appearance instrument tracks
    → representative clips / frames
    → QA-option-aware VLM Reader
```

图检索阶段先依据 track confidence、onset 支持、外观类别特异性、动作角色、外观描述和 event support 进行确定性排序；随后去重 evidence clips，默认最多保留 6 个 tracks、4 个 clips，每个 clip 8 帧，即最多 32 帧交给 Reader。

QA 选项只在最后的 Reader 重排阶段出现，检索不读取参考答案，也不需要人工器械标签。

核心实现：

- `src/medhorizon_videorag/graph_rag/semantic_layer.py`
- `src/medhorizon_videorag/graph_rag/phase_instrument_reader.py`
- `src/medhorizon_videorag/graph_rag/qa_experiment.py`
- `experiments/augment_semantic_evidence_graph.py`
- `experiments/evaluate_phase_instrument_reader.py`

### 7.2 初版覆盖问题

仅使用永久写入图中的 2 个 phase hypotheses 时，`087` 的 4 道 Phase-Instrument 问题只能解析 Aortic Clamping 1 题，其余 3 题因图中没有对应阶段而 unresolved。这个结果表明主要瓶颈首先是阶段覆盖，而不是器械轨迹数量。

### 7.3 Query-conditioned phase fallback

为提高阶段覆盖率，同时不把低置信度阶段永久写入图中，已加入查询时 fallback：

1. 优先匹配永久 `phase_hypothesis`；
2. 若不存在，加载 22 个开放活动区间；
3. 使用固定的 `phase-action-cues-v1` 对活动区间做确定性 Top-3 召回；
4. VLM 只查看候选活动摘要与代表帧，验证哪个区间最符合问题中的阶段；
5. 不把 QA 选项或答案交给阶段验证器；
6. 选中区间只生成临时 onset/context event 路径，不修改持久图；
7. 再沿同一 appearance-track Reader 路径完成器械选择。

默认阶段验证预算为 Top-3 activity segments × 每段 2 clips × 每 clip 4 frames，最多 24 帧。输出显式记录：

- `phase_route = query_conditioned_activity_fallback`；
- `persistent_graph_mutated = false`；
- 检索候选、视觉验证依据和临时边界；
- 最终使用的 appearance track 与证据帧。

器械角色提示额外约束：画面中的 needle-like object 不能直接当作 needle holder；当选项为 holder/forceps 时，应优先检查夹持、抓取该细长物体的器械，并根据钳口、杆体和尺寸线索区分候选。

## 8. `087` Phase-Instrument 开发诊断

最终诊断使用单卡 A6000 上的 Qwen2.5-VL-7B Reader，4 道题全部获得候选证据并完成回答：

| QA uid | 目标阶段 | 阶段路由 | 选中活动区间 | 预测 | 参考 | 结果 |
|---:|---|---|---|---:|---:|---|
| 2 | Left Atrium Suturing | query fallback | `open_activity:00001` | A | A | correct |
| 3 | Pericardial Suspension | query fallback | `open_activity:00010` | B | B | correct |
| 4 | Perfusion Needle Spacer Suturing | query fallback | `open_activity:00020` | A | A | correct |
| 5 | Aortic Clamping | persistent phase | 图中命名阶段 | D | D | correct |

汇总：

- graph/query-time coverage：4/4；
- completed：4/4；
- correct：4/4；
- query-conditioned route：3；
- persistent phase route：1；
- 人工器械身份标签：未使用；
- 答案参与检索或 Reader 输入构造：否。

结果目录：

`artifacts/graph_rag/087/phase_instrument_reader_qwen25vl7b_fallback_20260827/results_final/`

其中保留 `predictions.jsonl`、`report.json` 和 `run_metadata.json`，最终运行无错误记录。

### 8.1 必须保留的解释边界

这 4 道题及其输出在开发过程中被反复检查，并用于调整 cue 权重和 Reader 角色提示，因此 **4/4 只能作为开发集诊断，不能作为无偏 benchmark 成绩**。虽然运行时没有把参考答案输入检索器或 Reader，但规则调试已经间接使用了这些题目的反馈。

## 9. 已完成的工程验证

最新阶段/器械相关回归测试共 17 项通过，Ruff lint 与 format 检查通过。测试覆盖：

- evidence graph schema 与 normalization；
- event temporal merging、support score 和 representative evidence；
- 两阶段 sequence phase 输出；
- semantic overlay 与 appearance tracks；
- persistent phase 路径和 query-conditioned fallback；
- QA 选项仅进入 Reader、参考答案不参与检索；
- 持久图在查询 fallback 中不被修改。

当前本地与 A6000 功能代码均位于 `main` 分支的 `c1da6e4`；实验服务进程已经停止，无遗留 Reader/vLLM 后台任务。

## 10. 当前已知局限

1. **持久阶段覆盖仍低。** `087` 只有 2/22 个开放活动区间被映射为命名阶段，fallback 提高的是查询时可用性，不等同于完整阶段分段已经解决。
2. **候选阶段 ontology 存在 benchmark 泄漏风险。** 当前 ontology 来自测试问题候选阶段，只适合 candidate-aware diagnostic。
3. **规则在 `087` 上调过。** `phase-action-cues-v1` 和器械角色提示必须先冻结，再到 `047`、`079`、`grasp_CASE003` 等未参与调试的视频验证。
4. **appearance track 不是器械真值。** 它表示外观类型的连续出现，不能声称确定了医学器械名称或同一物理实例。
5. **缺少人工阶段边界与器械标注。** 当前不能直接测量阶段边界准确率、器械实体识别准确率或真实 track ID accuracy。
6. **Reader 仍可能过度解释。** 最终答案可能正确，但 rationale 中出现超出 observation graph 支持范围的医学判断；正式评价应把答案正确与证据依据分开。
7. **尚未完成通用多跳闭环。** 当前重点路径是 phase → boundary → instrument，尚未系统覆盖因果、阶段间比较、动作链问答和跨多个远距离事件的组合检索。

## 11. 推荐的下一阶段

按优先级建议：

1. **冻结 `087` 上的 schema、cue 权重和 prompt。** 不再针对这 4 道题继续调规则。
2. **在未参与调试的视频上做 held-out 验证。** 优先复用已生成描述的 `047`、`079`、`grasp_CASE003`，分别构建 v2.1 图、开放活动分段和 appearance semantic overlay。
3. **先评估阶段覆盖与证据路径，再评最终 QA。** 至少报告 persistent coverage、fallback coverage、Top-K 活动召回、unresolved rate、候选 evidence 区间和最终 QA accuracy。
4. **把 phase fallback 从手写 cue 表升级为可训练/可复现检索器。** 可比较文本 embedding、cross-encoder 或强模型只做候选验证，但保持检索、视觉验证和答案选择三阶段隔离。
5. **扩展真正的多跳问题。** 在 phase-boundary-instrument 闭环稳定后，增加 action sequence、phase transition、instrument-action-target 和前后状态变化路径。
6. **最后才考虑全量建图。** 多视频 schema 和阶段召回尚未稳定前，不建议对全部医学长视频进行昂贵的序列推断。

## 12. 关键代码与产物索引

### 代码

- `src/medhorizon_videorag/graph_rag/evidence_builder.py`
- `src/medhorizon_videorag/graph_rag/retrieval.py`
- `src/medhorizon_videorag/graph_rag/sequence_phase.py`
- `src/medhorizon_videorag/graph_rag/semantic_layer.py`
- `src/medhorizon_videorag/graph_rag/phase_instrument_reader.py`
- `src/medhorizon_videorag/graph_rag/qa_experiment.py`
- `experiments/build_evidence_graph.py`
- `experiments/retrieve_evidence_graph.py`
- `experiments/infer_two_stage_sequence_phases.py`
- `experiments/augment_semantic_evidence_graph.py`
- `experiments/evaluate_phase_instrument_reader.py`
- `configs/graph_rag_research.yaml`

### 文档

- `docs/graph_rag_research_plan.md`：总体研究路线与工程边界；
- `docs/medical_graph_qa_protocol.md`：无时间泄漏的多跳 QA 标注协议；
- 本文：已经实现的图结构、实验状态、结论和局限。

### `087` 主要 artifacts

- Observation descriptions：`artifacts/vgent_baseline/agicto_qwen3vl_235b_instruct_vgent88_observation_first_v10_087/`
- Evidence graph v2.1：`artifacts/graph_rag/087/evidence_graph_v2_1_qwen3vl235b_observation_v10/`
- Two-stage phase inference：`artifacts/graph_rag/087/sequence_phase_two_stage_qwen3vl235b_candidate_aware_20260825/`
- Combined semantic graph：`artifacts/graph_rag/087/sequence_phase_two_stage_qwen3vl235b_candidate_aware_20260825/combined_semantic_graph/`
- Final Phase-Instrument diagnostic：`artifacts/graph_rag/087/phase_instrument_reader_qwen25vl7b_fallback_20260827/results_final/`

## 13. 一句话状态

目前已经完成从保守 clip observation 到可追溯 temporal evidence graph，再到阶段边界、未知身份器械外观轨迹和 Phase-Instrument Reader 的第一版闭环；下一步不应继续在 `087` 上调参，而应冻结协议并在未参与开发的视频上验证阶段覆盖和证据路径泛化。
