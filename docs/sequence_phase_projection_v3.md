# Sequence phase 与 temporal event 的 v3 相交投影

旧流程给整个 temporal event 投票选一个阶段，再用事件起止生成阶段边界。
短阶段可能在两侧事件都输掉投票而消失；unknown 区间也可能被命名标签覆盖。

v3 的 `sequence-event-intersection-v3` 保留每个来源 sequence segment，
一个事件可以支持多个阶段，只把真实相交的区间作为支持证据。
该版本延续 observation graph v3，不修改第一层事实、阶段映射标签或接受阈值。

## 区间与证据

- 采用半开区间 `[start, end)`；仅接触端点不产生支持边。
- 对事件的逐 clip evidence 分别求交，不用事件包围区间填补空隙。
- 每条 phase→event 的 `derived_from` 边记录交集 evidence、重叠秒数、占事件时长比例和占阶段时长比例；时长按区间并集计，避免重复证据重复计时。
- 阶段节点保留完整 `source_segment`，包括原标签、候选、状态、basis 和源区间。同名的多个来源段不合并；unknown 也保留。
- onset/offset 时间直接来自来源阶段，grounded_by 只连接该边界所在的交集证据。若来源段没有事件支持，保留来源假设并显式记录空的 supporting_event_ids，不虚构事件边。
- `source_clip_duration_seconds` 记录输入 clip 粒度，`boundary_accuracy_seconds=null` 表示真实边界精度未知。64 秒采样粒度不等于 64 秒内准确。
- 对被部分截取的 clip，不携带无法逐帧核验时间的帧路径；Reader 仅取完整落在所选交集内的 clip。原始图和帧文件不改动。

器械 co_occurs、阶段边界检索和 persistent phase Reader 同步使用交集范围。
上下文事件仍按原有数量选取，但阶段外部分不作为直接支持；阶段外 clip 的器械动作角色也不进入本次 Reader 重排。

## 兼容与离线重投影

原推断 CLI 参数不变，`project_sequence_phases_to_events` 仍返回每个事件一行。
新增 `phase_overlaps` 及投影版本；完整来源目录 `sequence_phase_segments` 只在第一行保存一次。
标量 `phase_hypothesis` 仅在一个来源段完整覆盖事件时保留该标签，混合/不完整覆盖为 unknown；
它不能替代相交记录。旧事件级 JSONL 仍走原有 semantic augmentation 路径，
不会被静默重解释。旧的 Top1 投票指标不能与新标量字段直接比较。

旧图中 event IDs 与 v3 不一定一致，必须从保存的 sequence 文件重新投影；
augment 会重算并验证相交记录，拒绝旧 event 集合、缺行或被修改的投影证据。

```bash
python experiments/reproject_sequence_phases.py \
  --graph <observation-v3-evidence-graph.json> \
  --sequence-phases <saved-sequence-phase-segments.json> \
  --output-dir <new-output-directory>
```

默认重建 appearance instrument tracks，无 API、GPU、QA 答案或新模型推断。
目录必须不存在。输出 semantic_evidence_graph.json、semantic_hypotheses.jsonl、
semantic_graph_report.json、原 sequence 文件的内容副本、projection_audit.json、
projection_details.jsonl 和来源哈希 run_metadata.json。

验收检查：来源段与边界保留、实际事件交集一致、原观察图不变、支持边及边界检索不越界、schema 重载通过。
重点回归包括 079 的 Left Atrium Dissection 768–960 秒，在两个相邻事件上分别得到 64 秒和 128 秒支持。

候选集合随来源阶段保存，但本项不构造额外候选阶段节点、不改变候选排序或自动运行 Reader/QA；
query-conditioned activity 路线仍是原有独立协议，不是本次 persistent phase 投影验收对象。
