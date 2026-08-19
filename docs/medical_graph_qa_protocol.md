# 无时间泄漏的医学长视频 QA 协议

## 目标

问题必须依靠视频内容定位证据，不能直接给出待检索区间。时间戳只作为隐藏的评估标注，不进入检索输入或 Reader 提示。

不能通过机械删除时间表达生成题目。如果删除时间后同一视频存在多个合理答案，该题必须重写或舍弃。

## 每题必需字段

```json
{
  "id": "case_001_q_001",
  "video_id": "case_001",
  "question": "在首次暴露目标结构后，术者下一步使用了什么器械？",
  "answer": "分离钳",
  "choices": ["分离钳", "持针器", "吸引器", "剪刀"],
  "reasoning_type": "multi_hop",
  "required_evidence_count": 2,
  "evidence_relation": "phase_then_instrument",
  "evidence": [
    {"video_id": "case_001", "start_seconds": 320.0, "end_seconds": 346.0},
    {"video_id": "case_001", "start_seconds": 351.0, "end_seconds": 372.0}
  ],
  "hard_negative_evidence": []
}
```

机器可校验格式见 `schemas/medical_graph_qa.schema.json`，Python 侧契约见 `MedicalGraphQAExample`。

## 标注规则

- `evidence` 是回答问题所需的最小充分区间集合，而不是所有相关画面。
- `required_evidence_count` 表示答案至少依赖多少个独立区间。
- `multi_hop` 必须至少包含两个证据区间，并说明 `evidence_relation`。
- 重复出现的器械、动作和阶段应标注为 hard negatives，检验检索器能否排除语义相似但逻辑错误的区间。
- 每道题必须由医学或手术知识合格的标注者检查可回答性、唯一性与证据充分性。
- 数据按视频或病例划分，禁止同一视频的相邻片段跨训练集和测试集。

## 问题类型

- `single_hop`：单个区间即可回答；
- `multi_hop`：需要组合多个区间；
- `comparison`：比较不同阶段或发现；
- `causal`：联系前置操作与后续结果；
- `temporal_order`：恢复多个事件的先后顺序。

## 指标

检索与问答分开报告：

- 区间 Recall@K 和最佳 IoU；
- Evidence Set Recall：是否找齐回答所需的全部区间；
- hard-negative precision；
- 多跳路径正确率；
- 最终 QA accuracy；
- question-only 与视频条件的逐题配对结果；
- oracle evidence 下的 Reader 上限。

只有最终答案正确且所用区间覆盖标注证据时，才能计为 evidence-grounded correct。
