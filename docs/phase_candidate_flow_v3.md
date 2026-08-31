# 阶段候选传递与统一指标 v3

候选名字出现在生成文件中，不等于它是正向证据，也不等于检索器或 Reader 实际使用了它。
本协议分别记录候选的来源、证据极性、图中记录、检索、验证和 Reader 去留。

## 显式启用，旧默认行为不变

原 `evaluate_phase_instrument_reader.py` 不加新参数时仍使用原有 persistent phase / activity fallback 流程。
新增路线必须显式指定 `--phase-candidates`，不能与 `--open-activity-segments` 混合，避免隐式跨协议回退。
图需先按 sequence 相交投影协议保留完整 `source_segment`，对应 `sequence_phase_intersection_v3_20260831` 等新产物。

```bash
python experiments/evaluate_phase_instrument_reader.py \
  --annotations <annotations.jsonl> \
  --graph <semantic_evidence_graph.json> \
  --video-key <video> --qa-uids <ids> \
  --phase-candidates <phase_candidate_hypotheses.jsonl> \
  --candidate-top-k 3 --candidate-min-confidence medium \
  --candidate-dry-run \
  --output-dir <new-output-directory>
```

候选输入也可为包含显式 `phase_candidates` 的 sequence JSON，绝不把无候选列表的主标签伪装成候选。
087 的旧文件没有 top-k 列表，应单列为空目录诊断，不能混入 held-out top-k 实验分母。

`--candidate-dry-run` 不实例化模型，只准备验证输入和候选 Reader 包；状态为 prepared / prepared_not_sent，
verified_candidate_id、reader_candidate_id 和答案保持空。去掉该参数才会使用既有 model/base-url/api-key-env 配置调用模型；本协议不启动服务、不读取或保存密钥值。
输出目录必须不存在。原 `--option-aware-tracks`、`--option-verifier` 在新路线中仍可使用。

## 稳定 ID、极性和流转

- candidate ID 由 video、source segment ID、规范化 label 和原区间生成，排名、分数和文件顺序变化不会改变 ID。
- 拒绝重复 ID、视频不符、来源段缺失或区间不一致；不复用旧 event IDs 猜测投影。
- `contradicted` 永远是 counter_evidence；即使其 accepted=true 或有 positive_cues，也不能进入正向 Reader 路线。
- supported/tentative 且有正向 cue 的源记录标记 positive；insufficient 保持 uncertain。非反证不等于正向，也不等于验证通过。
- 检索按问题中的 phase name 筛选非反证候选，再按源 score、rank、时间和 ID 排序。这个 Top1 的名字命中是条件化路由检查，不是独立医学阶段识别成绩。
- 验证器接收候选 ID、decision、区间、正向 cue、负向 cue、缺失证据和实际帧；不接收 QA 答案、选项或隐藏时间标注。逐候选记录 positive_evidence / counter_evidence / missing_evidence。
- 只有具有可读帧的非反证候选被验证为 supported、有 positive_evidence 且置信度达到门槛，才可送入 Reader。
- Reader 输入与真实提示包含选中 candidate ID、源 decision、区间、验证结论和独立反证记录；复用相交区间的器械/clip 范围约束。

每题 `candidate_traces.jsonl` 保留全部文件候选，包括因名字不符、反证、top-k、缺少图支持或帧而没有继续的记录。
各候选均有 load / graph / retrieval / verification / reader 状态；分别记录 retrieval_top1_candidate_id、verified_candidate_id、reader_candidate_id，不能混用。
模型失败只记录阶段与异常类型，不把可能含凭据的 provider 异常文本写入日志。

## 统一指标：phase-candidate-metrics-v3

| 指标 | 定义与分母 |
|---|---|
| candidate_name_coverage | 文件中任意同名候选，包含反证；分母为请求题数 |
| candidate_non_counter_coverage | 同名且 decision 非 contradicted；insufficient 仍未确认 |
| candidate_positive_coverage | 同名且源证据角色为 positive |
| source_candidate_rank1_name_coverage | 来源段内 rank1 名字覆盖；不是检索 Top1 |
| graph_primary_phase_name_survival | 实际主阶段节点中是否有同名标签，独立于候选文件是否为空 |
| graph_candidate_record_survival | 同名候选在 source_segment 的候选元数据中存活；不是额外 candidate node |
| runtime_candidate_interval_grounding | 非反证候选是否映射到当前图实际相交事件 |
| retrieval_top1_name_match | 实际检索首选的名字匹配；受显式名字过滤条件化 |
| verified_selection_name_match | 真正执行验证后选中的候选；未运行模型为 null |
| phase_time_recall_gold / IoU | 仅使用独立审核的 gold phase windows；无标注为 null |
| weak_phase_anchor_recall / IoU / any_overlap | 仅使用 recover_evidence 的 weak phase_anchor；与 gold 分开，不以弱锚点宣称真值正确 |
| retrieval_top1_weak_anchor_recall | 首选候选对弱锚点的时长召回，不能用名字匹配替代 |
| answer_correct_all_requested / completed | 模型运行时分别按请求题/完成题计；未运行模型全部 null |
| evidence_interval_correct | 真正完成 Reader 后，其所选 track 对应显示 clip 对每个审核 gold evidence window 的召回达到门槛；只是时间覆盖，不是临床判定 |
| answer_and_evidence_interval_correct | 同时满足答案与上述证据区间标准；任一不可评估则 null |

时间召回按区间并集交集时长/标注并集时长计算，同时输出 union IoU 与 best-candidate IoU；不把“有交集”当作完整召回。
每项报告 value、value_sum、denominator、unavailable。没有模型输出或适用标注时不偷偷算 0。
参考答案、weak anchors、gold windows 仅在推理结束后传入共享指标函数，并存于 `evaluation_only`，不进入检索函数接口。

可选 `--gold-evidence` 为独立人工审核 JSONL，每行包含 id、video_key、reviewed=true、phase_windows 和/或 evidence_windows；区分阶段时间与回答所需器械证据。
`--evidence-recall-threshold` 默认 0.5，显式写入逐题指标；未提供审核标注时不报告证据正确率。

## 诊断与汇总

产物包括 candidate_traces.jsonl、question_metrics.jsonl、report.json、run_metadata.json。
event_catalog.jsonl 与可直接阅读的 event_details.md 保留全部 clip observations、atomic actions、entity mentions、完整 predicates/concepts；不截取 event label 或数组前几项充当事件语义。

```bash
python experiments/analyze_phase_candidate_flow.py \
  --run-dir <run-a> --run-dir <run-b> \
  --output-dir <new-summary-directory>
```

汇总使用相同指标实现，拒绝重复 video/QA 或混合指标版本。旧分析脚本/历史报告保留，不用旧 top1 字段与这里的检索 Top1 横比。
本项不修改阶段 mapper、不增加医学别名、不运行完整 Agent，也不宣称 QA 效果已经提高。
