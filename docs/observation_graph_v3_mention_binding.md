# Observation graph v3：mention 绑定

此次 v3 指第一层 observation evidence graph，区别于已有的 semantic v3 pilot。
它不修改 observation 描述、动作词表或全局 entity normalization，也不修改 phase mapper。

原实现按 `(category, canonical)` 取第一条 mention，会把同类不同器械的动作角色接错。
v3 使用以下顺序，在当前 clip 原始 visible mentions 中解析 subject/target：

1. 对大小写、空白和标点做归一化后的唯一精确匹配。
2. 找到短语中心词；只做有限词形/同义表达归并。`instrument holding thread` 的中心词是 instrument，绑定时不让其所持的 thread 覆盖主体。
3. 要求引用中保留的限定词和外观属性均得到候选支持，只接受唯一兼容候选。列表顺序不作平局裁决。
4. 多个兼容或重复精确候选标为 `ambiguous`，无充分支持标为 `unmatched`；为该原始动作论元创建独立 mention。它不与候选建立确定身份关系，也不会作为后续动作的候选池成员。

这是文本指代消歧，不是器械医学身份或物理实例确认。`resolved` 也不意味着跨 clip 追踪到同一把器械。

## 元数据和兼容性

- builder：`observation-evidence-graph-v3`。
- schema：`medical-video-evidence-graph-v3`；既有节点/边类型和 loader 保持兼容。
- binding：`observation-mention-binding-v3`。
- action metadata 增加 `subject_binding`、`target_binding`；角色边保存相同绑定记录。
- 记录 `surface`、`head`、`qualifiers`、`status`、`method`、候选 IDs、兼容候选 IDs、选中 ID 和 `physical_identity_confirmed=false`。
- 独立论元节点仍使用 `entity_mention`，来源字段为 `action_subject` / `action_target`，并保留 `argument_binding`。
- 不确定绑定不贡献跨 clip 角色一致性或论元特异性支持；相关独立节点和未消歧动作不生成可能连续关系。其他观察仍可支持活动连续，不能将这理解为禁止所有事件合并。

原 `build_evidence_graph.py` 参数不变，默认产出 v3。旧图继续可读且不覆盖。
事件数量及 event IDs 可能改变，因此旧第二层事件投影不可直接套用到新图；本次不自动重建第二层或重跑 QA。

## 四视频离线重建与验收

可以复用基线图的帧引用与合并参数，无需重抽帧或调用模型：

```bash
python experiments/evaluate_mention_binding.py \
  --baseline-graph <v2.1-graph.json> \
  --descriptions <observation-descriptions.jsonl> \
  --output-dir <new-v3-output-directory>
```

输出标准 graph 文件，以及 `mention_binding_audit/report.json` 和逐角色 `details.jsonl`。
输出目录必须不存在。已有 v3 图也可用 `--graph` 代替 `--descriptions` 单独验收。

验收除了“精确替代误连数为 0”还检查：

- 所有有唯一精确 visible mention 的论元都实际绑定到该节点，不能靠全部弃权得到 0。
- v2.1 已确认误连有多少被精确修复、有多少转成独立论元，分开报告。
- resolved 只选唯一兼容原始 visible mention；不确定项保留原文、独立节点和候选 IDs。
- clip 原始 observations 和 atomic actions 不变。
- 补充查看不同绑定方法的实际实例和画面；文本规则通过不等于视觉或临床准确率。

全局 concept 规则仍可能把复合器械短语归入材料概念；本次仅避免该问题决定指代绑定，没有顺带修改另一项实体归一化任务。
规则对未知中心词、复杂限定关系和未表达属性会保守弃权。新增独立 mentions 及事件变化必须在验收报告中展示，不能只报告错误减少。
