# Legacy NLU Prompt

旧 AgentOrchestrator 已删除。本页仅保留仍由 `IntentRecognizer` 使用的
legacy 意图分类和实体提取 Prompt 说明。

## 适用范围

- `build_intent_prompt()`：在兼容 NLU 路径中进行单标签意图分类。
- `build_entity_prompt()`：提取消息中逐字出现的订单号、商品、日期、金额和错误码。

两类 Prompt 都将用户消息和历史视为不可信数据，禁止其覆盖任务边界；输出必须符合
固定 JSON Schema。它们不执行客服回复、不加载业务 Skill，也不调用任何业务工具。

源码：[core/prompts/legacy.py](../../core/prompts/legacy.py)。
