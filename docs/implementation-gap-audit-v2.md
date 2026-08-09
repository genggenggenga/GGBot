# GGBot 当前实现缺口审计 v2

> 审计时间：2026-08-09  
> 审计基线：当前工作区实现  
> 原审计文档保持不变；本文只描述当前仍成立的结论。  
> 范围：`/chat` 主链路、领域 Agent、内部 RPC、RAG、记忆、并发、监控、评测与部署。

## 1. 结论摘要

本轮升级已完成主链路的三项结构性收敛：

1. 主应用不再启动 MCP stdio 子进程，业务工具改为显式 ToolSpec 和内部 RPC Protocol。
2. NLU、ReAct、QueryPlanner、RAG Answer、ResponsePolisher 和画像提取改用原生 Tool Calling 与 Pydantic。
3. 同一会话增加 Redis 租约锁，写操作增加 Redis 共享 action_id 幂等仓库。

全量回归结果：

| 检查项 | 结果 |
|---|---|
| 全量 pytest | `383 passed, 1 warning` |
| 售后 Planner 降级定向测试 | 退货、取消订单均不调用退款工具 |
| 文档链接与图片资源 | v2 文档引用独立 v2 图 |
| 真实模型效果 | 未验证，不得由确定性测试代替 |

当前没有发现会在正常已覆盖路径中直接绕过 PendingAction 的写操作。仍需优先处理的风险集中在真实业务接入、跨系统最终幂等、认证授权、真实模型效果和运维可靠性。

## 2. 当前主链路

```text
POST /chat
  -> RedisConversationLockManager(user_id, conv_id)
  -> MemoryManager.get_context()
  -> Fast Track / StructuredLLMClient
  -> DialogueStateTracker
  -> TurnEngine
  -> Deterministic Router
  -> KnowledgeAgent or bounded ServiceAgent
  -> ToolRegistry
  -> Internal RPC Client / RAG
  -> state + messages + episodic event commit
  -> release conversation lock
```

关键安全边界：

- Router 仍是确定性映射，不由模型自由选择 Agent。
- ReAct 只在领域 Agent 内选择白名单工具。
- WRITE 决策只能创建 PendingAction。
- 用户确认后，ToolRegistry 才放行冻结的动作参数。
- RedisActionExecutionRepository 以 `tool_name + action_id` 协调跨实例重放。
- 售后 Planner 失败时，仅退款允许使用退款专用确定性 fallback。

## 3. 本轮已关闭的旧缺口

| 旧缺口 | 当前状态 | 当前实现 |
|---|---|---|
| MCP 子进程与动态发现漂移 | 已关闭 | `register_internal_rpc_tools()` 显式注册 ToolSpec |
| 写结果只在进程内幂等 | 已关闭到应用层 | Redis Lua 原子抢占、参数指纹和结果重放 |
| 同会话并发覆盖状态 | 已关闭 | Redis token 租约锁、续租和 compare-and-delete 释放 |
| 宽松截取模型 JSON | 主链路已关闭 | 强制 Tool Calling，Pydantic 禁止额外字段 |
| RAG 只返回 Top-1 片段 | 已关闭 | Top-N 证据预算、Grounded Answer 和 Citation 校验 |
| Skill 全局拼入上下文 | 已关闭 | 只向 AfterSalesAgent 注入匹配 intent 的软策略 |
| Monitor 动态改变路由 | 已关闭 | Monitor 只观测和告警 |
| 退货/取消 Planner 异常误走退款 | 已关闭 | 非退款售后禁止进入退款确定性 fallback |

这些关闭项只代表当前应用层实现成立，不代表下游真实业务服务已经具备相同保证。

## 4. P0：对外发布阻断项

### P0-1 身份认证与资源授权缺失

当前 API 没有认证，CORS 仍允许广泛访问，内部 Mock RPC 也不校验订单归属。知道订单号的调用者可能查询订单或发起售后动作。

发布前必须补齐：

- 用户或服务身份认证；
- `user_id` 不可由未认证请求任意声明；
- 订单、物流和售后资源归属校验；
- 管理接口和普通客服接口分权；
- `/skills/reload`、知识发布/撤销、评测和 Trace 查询的权限控制；
- 审计日志与敏感字段分类。

### P0-2 真实业务系统最终幂等缺失

RedisActionExecutionRepository 可以协调 GGBot 实例，但不能替代真实退款、退货或取消系统的最终唯一约束。

风险场景：

- RPC 已成功但应用在写 Redis 结果前崩溃；
- Redis 数据过期后旧 action_id 被重放；
- 下游超时但实际已产生副作用；
- 跨系统补偿导致业务状态与幂等记录不一致。

真实下游必须以 action_id 或业务幂等键建立数据库唯一约束，并提供可查询的最终动作状态。

### P0-3 真实模型和 no-answer 效果未达发布证据标准

当前 50 条样本主要用于确定性回归。FakeDense 与 FakeReranker 不能证明真实 BGE、BM25 融合、Cross-Encoder 或回答生成效果。

发布前至少需要：

- 真实 Embedding / Reranker 离线集；
- no-answer、冲突知识、历史版本和跨地区口径样本；
- AnswerGenerator 引用支持率与幻觉率；
- Planner 工具选择、参数准确率和降级率；
- Prompt Injection 与恶意 Observation 样本；
- 按业务风险分层的人工评审。

## 5. P1：高优先级工程缺口

### P1-1 ConfirmationGate 仍是进程内瞬时状态

PendingAction 本身持久化在 DialogueState，写执行结果持久化在 Redis；但 ToolRegistry 的 confirmed 集合仍在进程内。当前确认轮会先从 DialogueState 恢复并重新调用 `confirm_action()`，主路径可工作，但该状态分散在两个组件中。

建议将“动作已确认”的权威判断收敛为持久化 ActionExecution 状态机，避免未来新增调用入口绕开恢复逻辑。

### P1-2 会话锁缺少失锁后的强制中止

租约锁支持续租，但续租失败时 `_renew_lease()` 直接退出，持锁请求不会立即获知自己已经失去所有权。极端 Redis 抖动或长时间阻塞后，两个请求可能先后进入提交阶段。

建议：

- 暴露 lock-lost 事件；
- 在状态与消息提交前检查所有权；
- 为长模型调用设置低于租约的统一超时；
- 增加 Redis 故障与租约丢失集成测试。

### P1-3 售后 Skill 仍是非结构化软策略

Skill 已按 `AfterSalesAgent + intent` 定向注入，不再污染 KnowledgeAgent；但它仍是自然语言 Prompt，不能作为工具权限、资格或合规规则的权威来源。

建议把可执行策略拆为结构化字段：

```json
{
  "intent": "return_request",
  "required_read_tools": ["query_order", "evaluate_after_sales_options"],
  "allowed_write_tools": ["create_return"],
  "confirmation_required": true
}
```

自然语言 Skill 只保留处理顺序和客服表达，权限继续由代码控制。

### P1-4 售后写结果缺少统一强类型模型

退款和工单有显式 Pydantic 输出模型；退货和取消主要依赖 ReAct 对 Observation 的解释。应为所有写工具建立统一的 created/cancelled、request_id、reason_code 和 finality 契约，避免模型自由解释缺字段结果。

### P1-5 情景记忆提交增加请求尾延迟

任务完成和转人工事件现在在会话锁内同步写入，解决了跨轮污染，但 Chroma 或摘要模型延迟会延长锁持有时间。

后续可使用 outbox：

1. 锁内原子写入事件事实；
2. 锁外异步生成摘要和写 Chroma；
3. 通过 event_id 保证幂等；
4. 失败可重试且不读取下一轮消息。

### P1-6 健康检查只验证对象存在

`/health` 能确认组件已装配，但没有主动探测 Redis、Chroma、模型端点和内部 RPC。建议拆分：

- `/live`：进程存活；
- `/ready`：关键依赖轻量探测；
- `/health/details`：受权限保护的详细状态。

## 6. P2：持续演进项

| 方向 | 当前边界 | 建议 |
|---|---|---|
| Trace | 有界内存，重启丢失 | 接入持久 Trace backend 和保留策略 |
| Mock RPC | 进程内固定数据 | 替换为真实 HTTP/Thrift/gRPC Client |
| Tool schema | output_schema 多为宽泛 object | 为每个 RPC 维护强类型输入输出契约 |
| 评测 | 确定性 fixture 为主 | 增加真实模型、真实知识与并发回归 |
| 部署 | 多个脚本与环境变量来源 | 收敛单一启动入口和配置校验 |
| 兼容代码 | 仍保留 MCP/Legacy 模块 | 明确下线窗口，禁止主链路重新依赖 |
| 配置安全 | 示例中包含演示密码 | 生产从 Secret Manager 注入 |

## 7. Mock 与演示边界

| 模块 | 当前简化 | 可接受范围 |
|---|---|---|
| Internal RPC | 订单、物流、售后使用固定字典 | 本地 Demo 和确定性测试 |
| 写动作编号 | 根据 action_id 稳定生成 | 仅验证编排，不代表真实受理 |
| 默认知识 | 空库自动写入演示知识 | 本地体验 |
| Dense/Reranker 评测 | Fake 模型 | 代码回归 |
| TraceStore | 内存最多 1000 条 | 单实例调试 |
| ConfirmationGate | 进程内集合 | 仅主 `/chat` 恢复路径 |

任何演示结果都不能用于声明生产成功率、模型效果或真实退款能力。

## 8. 推荐实施批次

### Batch 1：发布安全

1. 接入身份、订单归属与管理接口授权。
2. 与真实下游约定 action_id 唯一约束和状态查询。
3. 建立审计日志和敏感字段分类。

### Batch 2：并发与副作用可靠性

1. 增加 lock-lost 检测和提交前所有权校验。
2. 收敛 ConfirmationGate 与 ActionExecution 状态。
3. 覆盖超时未知结果、崩溃恢复和 Redis 故障测试。

### Batch 3：策略与工具契约

1. 将售后 Skill 拆成结构化策略和自然语言 SOP。
2. 为 refund/return/cancel/ticket 建立统一强类型结果。
3. 让 completion condition 只依赖经过校验的业务结果。

### Batch 4：RAG 与模型效果

1. 建立真实模型和 no-answer 数据集。
2. 校准分模式阈值、证据预算和生成拒答。
3. 记录结构化输出失败率、Planner 降级率和引用校验失败率。

### Batch 5：生产运维

1. 引入 outbox、持久 Trace 和可靠重试。
2. 拆分 liveness/readiness。
3. 收敛部署脚本、配置校验和 Secret 管理。

## 9. 关键代码入口

- `api/main.py`
- `core/conversation_lock.py`
- `core/structured_llm.py`
- `core/idempotency.py`
- `core/internal_rpc.py`
- `core/rpc_tools.py`
- `core/customer_agent_runtime.py`
- `agents/domain_agents.py`
- `core/tool_registry.py`
- `rag/answer_generator.py`
- `memory/conversation_memory.py`
