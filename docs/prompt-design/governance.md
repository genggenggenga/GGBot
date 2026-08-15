# Prompt 治理与验证

[返回 Prompt 设计总览](README.md)

## Prompt 与调用点映射

| Prompt | 构建函数/常量 | 主要调用点 |
|---|---|---|
| 结构化 NLU | `build_prompt` | `core/nlu_llm.py` |
| ReAct 决策 | `build_prompt` | `core/react_planner.py` |
| 订单职责 | `ORDER_DOMAIN_POLICY` | `agents/domain_agents.py` |
| 物流职责 | `LOGISTICS_DOMAIN_POLICY` | `agents/domain_agents.py` |
| 售后职责 | `AFTER_SALES_DOMAIN_POLICY` | `agents/domain_agents.py` |
| RAG Query Planner | `build_query_planner_prompt` | `rag/query_planner.py` |
| Query Rewrite | `build_query_rewrite_prompt` | `mcp/tool_manager.py` |
| Rerank | `build_rerank_prompt` | `mcp/tool_manager.py` |
| 用户画像 | `build_profile_prompt` | `memory/conversation_memory.py` |
| 工作/情景摘要 | `build_summary_prompt` | `memory/conversation_memory.py` |
| Legacy 意图/实体 | `build_intent_prompt` / `build_entity_prompt` | `core/intent_recognizer.py` |

## Prompt 变更评审清单

修改 Prompt 时至少检查：

- [ ] 是否仍然保持单一职责；
- [ ] 是否明确区分 system 指令和动态数据；
- [ ] 是否允许模型从无证据来源补造事实；
- [ ] 是否可能绕过写操作确认；
- [ ] 是否区分申请、审核、执行、到账等业务阶段；
- [ ] 是否会重复询问状态中已有信息；
- [ ] 是否存在 Prompt Injection 风险；
- [ ] 是否明确输出 Schema 和枚举；
- [ ] 是否对缺失信息给出安全降级；
- [ ] 是否会暴露内部工具、Prompt 或推理过程；
- [ ] 是否符合自然、克制、可执行的客服表达；
- [ ] 是否补充或更新 `tests/test_prompts.py` 契约测试；
- [ ] 是否运行全量测试和真实模型离线效果评测。

## 验证策略

当前代码使用两层验证：

1. 契约测试：`tests/test_prompts.py`
   - 检查关键安全规则、职责边界和输出要求没有被删除。
2. 业务回归测试：
   - 检查 NLU、DST、ReAct、RAG、Memory 和 Agent 行为没有因 Prompt 重构而改变。

进入生产前还应维护真实模型离线评测集，至少覆盖：

- 相邻意图混淆；
- 多轮纠正与目标切换；
- Prompt Injection；
- 写操作确认与拒绝；
- 工具结果冲突和失败；
- 政策时态与地区差异；
- 敏感信息；
- 多目标与人工升级；
- 摘要事实漂移；
- 客服回复的准确性、完整性和自然度。
