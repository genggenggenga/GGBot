# GGBot 当前实现缺口与修改优先级

> 审计基线：`feat-v1` / `c80e5fb`  
> P0 / P1 / P2 代码级修复状态：已完成，等待提交  
> 审计范围：主 `/chat` 运行时、领域 Agent、MCP、RAG、记忆、监控、评测和部署脚本。

## 1. 结论摘要

当前版本已经打通退款 Demo 的基本链路，但“声明支持的能力”和“真实可执行能力”之间仍有明显差距。问题不只是存在 Mock，更关键的是部分 Mock、路由和评测逻辑掩盖了真实错误。

建议修改顺序：

1. **P0：先恢复可信测试基线，并阻止错误业务动作。**
2. **P1：补齐主运行时已经声明、但没有真正接线的能力。**
3. **P2：在外部部署前替换 Mock，并补足持久化、安全和运维能力。**

本次实测结果：

| 检查项 | 结果 |
|---|---|
| 修复前默认环境执行全量测试 | `214 passed, 24 failed` |
| 修复后默认环境执行全量测试 | `277 passed, 1 warning` |
| 定向 P0 测试 | `164 passed, 1 warning` |
| Docker Compose 静态校验 | 当前机器无 Docker，未执行 |
| Shell 语法与 Git diff | 通过 |

P0 修复结果：

| P0 | 状态 | 修复结果 |
|---|---|---|
| P0-1 错误售后工作流 | 已修复 | 退货、取消、投诉和转人工在能力未实现时 fail closed，不再调用退款工具 |
| P0-2 业务失败误报成功 | 已修复 | 订单、物流和退款创建结果使用强类型业务输出并检查 `found/created` |
| P0-3 评测绕过主链路 | 已修复 | Tool/E2E 用例改走 CustomerAgentRuntime 与 TurnEngine，预测工具来自真实调用记录 |
| P0-4 默认测试基线损坏 | 已修复 | Fake Chroma 注入时不再隐式创建远端 Embedding；评测 fixture 纳入版本管理 |
| P0-5 Fast-track 歧义 | 已修复 | confirm/reject 只在待确认状态生效，correction 优先，配送 FAQ 进入知识查询 |
| P0-6 固定业务日期 | 已修复 | 退款资格改用可注入 business clock，并增加截止日前后测试 |

P1 修复结果：

| 范围 | 状态 | 修复结果 |
|---|---|---|
| Handoff 工作流 | 已修复 | 投诉/转人工支持确认后创建工单 |
| Skills 与记忆 | 已修复 | 受控上下文进入 RAG query，消息写入后触发稳定偏好画像更新 |
| 复合意图与完成契约 | 已修复 | NLU 输出 intents，Runtime 去重执行 Agent 并评估 completion condition |
| 状态恢复 | 已修复 | 本轮入口由持久化业务状态推导，非法迁移转结构化 FAILED |
| 主链路监控 | 已修复 | CustomerAgentRuntime 与 MCPToolAdapter 统计接入 Monitor |
| RAG 阈值 | 已修复 | Dense / RRF / Reranker 使用独立阈值 |
| Canonical chunk | 已修复 | KnowledgeBase 保存统一 chunk_id 与完整 citation metadata |
| 异步阻塞 | 已修复 | 同步 Redis / Chroma / Retriever 调用移出事件循环 |
| 健康检查 | 已修复 | `/health` 检查主 Runtime、ToolRegistry、Memory、MCP 和 RAG |
| 双运行时 | 已收敛 | `/search`、默认评测和 CLI 使用主 Runtime；Legacy evaluator 改为显式开关 |

P2 代码级修复结果：

| 范围 | 状态 | 结果 |
|---|---|---|
| ConfirmationGate 清理 | 已修复 | WRITE 成功后释放 pending / confirmed action_id |
| 幂等 payload 冲突 | 已修复 | 同 action_id 不同 payload 返回 `idempotency_conflict` |
| 后台任务 | 已修复 | 情景记忆和 Webhook 任务被跟踪，关停时 drain |
| Trace 数据分类 | 已修复 | 顶层事件和 Observation preview 改为 allowlist |
| 告警去重 | 已修复 | 相同未恢复阈值只保留一条 active alert |
| 无效声明 | 已清理 | 移除 `query_payment` 和未接线安全/功能开关 |
| Mock 替换、持久化幂等、鉴权、持久 Trace、部署脚本 | 暂缓 | 需要真实后端、数据库、认证或容器环境 |

## 2. Mock 与演示实现清单

Mock 本身不是错误，但必须明确边界，不能作为真实效果或业务完整性的证据。

| 模块 | 当前 Mock / 简化 | 当前是否可接受 | 后续要求 |
|---|---|---|---|
| MCP 业务服务 | 订单、物流、退款、工单均为进程内字典 | Demo 可接受 | 外部接入前替换为 Repository / Gateway，并加入身份与订单归属校验 |
| MCP 写入幂等 | `_REFUNDS_BY_ACTION`、`_TICKETS_BY_ACTION` 为进程内字典 | Demo 可接受 | 改为数据库唯一键或 Redis 原子写；校验相同 action_id 的 payload 一致性 |
| 默认知识库 | 空库自动写入 6 篇演示知识 | 本地体验可接受 | 生产配置必须允许关闭，且明确数据来源与版本 |
| 本地评测 Dense | token overlap，不是真实 Embedding | 仅组件回归可接受 | 不能用于证明 BGE 效果；增加真实模型离线评测 |
| 本地评测 Reranker | 对 `query + chunk_id` 做哈希随机排序 | 仅管线回归可接受 | 不能用于证明重排效果；增加真实 Cross-Encoder 评测 |
| 本地评测 MCP | 另一套独立 Mock handler | 风险较高 | 改为调用标准 MCP Server 或共享 contract fixture，避免两套规则漂移 |
| Legacy Agent | 主要依赖 LLM 文本生成，不执行真实业务动作 | 兼容路径可暂留 | 明确下线时间，避免继续承载监控和默认评测 |
| TraceStore | 进程内最多 1000 个 trace | Demo 可接受 | 多实例或重启场景改为持久化 Trace backend |

关键证据：

- `mcp_server/customer_service_server.py:10-72`
- `mcp/knowledge_base.py:86-89,193-266`
- `evaluation/local_eval_runner.py:47-203`
- `core/trace_store.py:119-138`

## 3. P0：已修复

### P0-1 退货和取消订单会进入退款流程

**现象**

`return_request`、`cancel_order`、`complaint` 和 `escalation` 都被路由到 AfterSalesAgent，但该 Agent 只有退款规划：

`query_order → check_refund_eligibility → create_refund`

实测：

- “我要退货 ORD-1001”返回“订单符合退款条件，请确认提交退款申请”。
- “取消订单 ORD-1001”返回相同退款确认文案。

如果用户继续确认，系统会调用 `create_refund`，而不是退货或取消订单工具。

**证据**

- `core/agent_models.py:182-225`
- `agents/domain_agents.py:35-49`
- `agents/domain_agents.py:220-310`

**修改**

先建立明确的 `IntentCapability`：

- `refund_request` → RefundWorkflow
- `return_request` → ReturnWorkflow
- `cancel_order` → CancelOrderWorkflow
- `complaint / escalation` → HandoffWorkflow

未实现的能力必须 fail closed，返回“当前尚未支持该动作”，不能复用退款链路。

**验收**

- 四类意图各有独立工具序列测试。
- 退货和取消订单流程中永远不会出现 `create_refund`。
- 未注册能力在执行任何 WRITE 工具前终止。

### P0-2 工具传输成功被错误当作业务成功

**现象**

MCP 返回 `found=false` 或 `created=false` 时，MCPToolAdapter 仍生成 `ToolResult(success=True)`。领域 Agent 只检查 `ToolResult.success`，没有统一检查业务结果。

已确认：

- OrderAgent 对不存在订单返回“订单 ORD-X 当前状态：None”。
- LogisticsAgent 在物流不存在时可能返回“当前物流状态：None”。
- `create_refund` 返回 `created=false` 时，AfterSalesAgent 仍会返回“退款申请已提交，申请编号：None”。

**证据**

- `core/mcp_adapter.py:97-130`
- `agents/domain_agents.py:100-129`
- `agents/domain_agents.py:175-216`
- `agents/domain_agents.py:277-286`
- `mcp_server/customer_service_server.py:75-101,145-152`

**修改**

为每个工具定义强类型输出模型和业务状态，例如：

- `OrderQueryResult(found, order, error_code)`
- `RefundCreateResult(created, refund_id, error_code)`

Adapter 只负责协议成功，Agent 必须根据 typed result 判断业务完成、业务拒绝和系统失败。

**验收**

- `found=false`、`eligible=false`、`created=false` 分别有独立断言。
- 用户响应不得出现 `None`。
- 业务拒绝不进入 `FAILED`，系统异常才进入 `FAILED`。

### P0-3 当前“端到端评测”没有执行真实主链路

**现象**

本地评测不能发现 P0-1 和 P0-2：

- Tool Selection 由 `_AGENT_DEFAULT_TOOL` 和字符串判断直接构造，不执行 Agent。
- DST 的 expected slots 由被测的同一个 `fast_track_extract()` 生成，形成循环验证。
- E2E 绕过 CustomerAgentRuntime 和 TurnEngine，手工调用 Agent。
- 当用例期望 `failed` 时，代码直接把 predicted status 改为 `completed`。
- Citation Precision 和 Faithfulness 固定写死为 `0.0`。
- NLU act 错误不进入 intent 指标，因此“不要办了”识别失败不会影响 1.0 的 Intent Accuracy。

**证据**

- `evaluation/local_eval_runner.py:248-288`
- `evaluation/local_eval_runner.py:291-329`
- `evaluation/local_eval_runner.py:356-432`
- `evaluation/local_eval_runner.py:494-548`

**修改**

将评测分成两层：

1. deterministic unit regression：保留 Fake，但重命名，禁止用于效果结论。
2. runtime integration eval：通过 CustomerAgentRuntime、TurnEngine、标准 MCP Server 执行，并使用人工标注的 expected intent、slots、tools、status 和 citations。

删除任何根据 `expected_*` 修改预测结果的代码。

**验收**

- P0-1、P0-2 的回归测试在修复前必须失败。
- E2E case 能输出真实 state path 和真实 tool observations。
- expected 数据只来自 fixture，不调用被测实现生成。

### P0-4 默认测试基线已损坏

**现象**

MemoryManager 即使注入 FakeChroma，也会根据默认 `RAG_EMBEDDING_PROVIDER=api` 构造 SiliconFlowEmbeddingFunction。未提供 Key 时，24 个记忆测试在初始化阶段失败。

这意味着测试是否通过依赖开发机环境变量，不再是 hermetic test。

**证据**

- `memory/conversation_memory.py:40-55`
- `memory/conversation_memory.py:232-242`
- `tests/test_memory.py:202-222`

**修改**

- Embedding function 改为显式依赖注入。
- 注入 `chroma_client` 时允许同时注入 `embedding_function=None`。
- 测试 fixture 固定 provider，不读取开发机环境。
- 明确配置优先级：函数参数应高于环境变量。

**验收**

- 不设置任何外部 Key 时全量测试通过。
- `api / local / off` 三种 provider 分别有启动测试。
- 更新技术文档中的测试数字。

### P0-5 Fast-track 把歧义表达当作高置信度确定结果

**现象**

- “不对，是 ORD-1002”同时命中 correction 和 reject；如果已有 PendingAction，会被当作拒绝并取消动作。
- “不办了”没有命中 reject，但评测仍显示整体 Intent Accuracy 1.0。
- “配送一般几天”被高置信度识别为 `logistics_query`，随后强制索取订单号，而不是走配送 FAQ。
- `greeting / other` 默认进入 KnowledgeAgent；配合阈值 0，可能返回任意知识片段。

**证据**

- `core/nlu_fast_track.py:27-56`
- `core/nlu_fast_track.py:78-96`
- `core/nlu_fast_track.py:169-201`
- `core/intent_recognizer.py:198-210`
- `agents/domain_agents.py:43-49`

**修改**

- 用户行为识别改为上下文相关：仅在 `confirmation_status=PENDING` 时识别 confirm/reject。
- correction 优先级高于 reject。
- 区分“查询具体订单物流”和“咨询配送政策”。
- greeting、feedback 和 unsupported other 使用确定性响应或 clarification，不直接 RAG。

**验收**

- 建立歧义语料表，覆盖 correction/reject/switch 的组合。
- act accuracy 单独计分。
- FAQ 不要求 order_id，订单物流查询必须要求 order_id。

### P0-6 退款资格时间固定为 `2026-08-02`

**现象**

`check_refund_eligibility()` 默认使用固定日期；`create_refund()` 也没有传入真实当前时间。随着时间变化，过期订单仍可能被判定为可退款。

**证据**

- `mcp_server/customer_service_server.py:104-130`
- `mcp_server/customer_service_server.py:134-152`

**修改**

通过 Clock 依赖注入当前日期，测试中使用 FrozenClock；生产实现使用服务端 UTC 日期。不要让调用方任意传 `as_of` 绕过规则。

**验收**

- 退款截止日前、截止日当天、截止日后均有测试。
- create 与 eligibility 使用同一业务时钟。

## 4. P1：已修复

### P1-1 投诉、转人工和工单仅声明，未实现

`complaint` 和 `escalation` 的 completion condition 是 `handoff_created`，MCP 也暴露了 `create_ticket`，但 AfterSalesAgent 从不规划 `create_ticket`。直接执行这两个意图时会用 `order_id=None` 调 `query_order` 并失败。

建议实现独立 HandoffAgent，生成结构化 HandoffPackage，并在用户确认或策略允许后创建工单。

证据：

- `core/agent_models.py:182-185`
- `agents/domain_agents.py:220-310`
- `mcp_server/customer_service_server.py:167-190`

### P1-2 Skills 和长期记忆被组装，但不影响主 Agent 行为

API 将 Skills、DialogueState、摘要、情景记忆和画像组装成 `agent_context`，ServiceAgent 又把它作为 tool context 传入；但 MCPToolAdapter 直接丢弃 context，RAG handler 也不使用 context，领域规划完全是确定性代码。

此外，`update_profile()` 在生产代码中没有调用点，因此用户画像不会自动产生。

建议：

- Skills 进入明确的 PolicyResolver，而不是只拼接 Prompt。
- 长期记忆只提供经过 schema 校验的偏好字段。
- 在消息写入后按稳定偏好门控触发 `update_profile()`。
- 记录哪些上下文实际参与了决策。

证据：

- `api/main.py:360-385`
- `core/customer_agent_runtime.py:84-89,194-199`
- `core/mcp_adapter.py:97-105`
- `memory/conversation_memory.py:277-343`

### P1-3 复合意图和 completion condition 是“声明能力”

`UnderstandingResult` 支持 intents，Router 支持 `route_tasks()`，IntentSchema 也声明 completion condition，但 CustomerAgentRuntime 只保存 primary intent，执行时没有传 intents；completion condition 只是字符串，从未被求值。

建议建立可执行 GoalContract，明确每个 goal 的完成谓词，并让 Runtime 以 task list 驱动多个 Agent。

### P1-4 TurnEngine 恢复状态被主运行时覆盖

`load_context()` 会推导 resume state，但 CustomerAgentRuntime 随后重新创建 `TurnContext(execution_state=UNDERSTANDING)`。当前退款确认依赖 DST 再次计算后仍能工作，但恢复算法并没有真正决定入口状态。

另外 InvalidTransitionError 被直接重新抛出，会形成 API 500，而不是结构化 FAILED/Handoff。

证据：

- `core/customer_agent_runtime.py:67-83`
- `core/turn_engine.py:144-172`
- `core/turn_engine.py:199-210`

### P1-5 监控采集的是兼容运行时，不是主 `/chat` 运行时

PerformanceMonitor 读取 AgentOrchestrator 和旧 MCPToolManager；主链路使用 CustomerAgentRuntime 和 ToolRegistry。MCPToolAdapter 又没有 stats，因此 `/monitor` 无法反映主链路的 Agent / MCP 成功率。

Prometheus Histogram 还在周期任务中重复 observe“累计平均延迟”，不是逐请求延迟；`requests_total` 定义后没有递增。

证据：

- `api/main.py:231-240`
- `monitor/performance_monitor.py:183-241`
- `core/mcp_adapter.py:90-130`
- `core/tool_registry.py:566-581`

### P1-6 RAG 拒答阈值默认无效

默认 `RAG_RELEVANCE_THRESHOLD=0`。只要 Dense 返回任何候选，`selected[0].score < 0` 几乎不会成立，因此无关问题也可能被当作有证据回答。

同一阈值还同时比较 Dense score、RRF score 和 Reranker score，三者量纲不同。

建议按检索模式分别校准阈值，并增加 no-answer 标注集。

证据：

- `.env.example:32-39`
- `rag/retriever.py:98-169`

### P1-7 RAG 存在双重索引、重复 Embedding 和元数据丢失

KnowledgeBase collection 已经执行一次 Embedding，KnowledgeRuntime 又创建 `knowledge_dense_bge` 并重新 Embedding。每次启动还会对已有 chunks 再次 upsert。

导入时 KnowledgeRuntime 把 page 等结构信息写入独立 Dense chunk，但 KnowledgeBase.add_documents 不保存这些元数据；重启后从 KnowledgeBase 重建 Dense 时，PDF page 和 section 可能丢失。

建议只保留一个 canonical chunk store，Dense/BM25 都使用相同 chunk_id 和 metadata。

证据：

- `rag/runtime.py:71-111`
- `rag/runtime.py:140-159`
- `mcp/knowledge_base.py:92-117`

### P1-8 同步存储和模型调用阻塞异步请求

主 API 是 async，但 Redis 使用同步客户端，Chroma query、SiliconFlow rerank 的同步 httpx Client、本地 CrossEncoder predict 都在事件循环中直接执行。

建议使用 async Redis，将阻塞 Chroma / 模型调用放入线程池或独立服务，并设置统一并发限制。

### P1-9 健康检查不能代表主运行时健康

`/health` 只检查旧 `_orchestrator` 是否存在，并返回 legacy stats。它不检查 `_customer_runtime`、Redis、Chroma、MCP session 或主 ToolRegistry。

建议拆分：

- `/live`：进程存活
- `/ready`：主运行时及依赖可用
- `/health/details`：各组件状态

证据：`api/main.py:323-327`

### P1-10 双运行时继续造成配置、监控和评测漂移

主 `/chat` 使用新运行时；CLI、默认 evaluator、`/search`、monitor 仍使用旧 Orchestrator / MCPToolManager。相同概念有两套 ToolResult、CircuitBreaker、RAG 和 Agent 统计。

建议停止向 legacy 链路增加功能，逐项迁移后删除：

1. monitor
2. evaluator
3. `/search`
4. CLI
5. legacy ToolManager / Orchestrator

## 5. P2：代码级已修复，部署项暂缓

### P2-1 替换业务 Mock，但保留标准 MCP contract

将 MCP Server 的字典替换为 OrderGateway、LogisticsGateway、RefundGateway 和 TicketGateway。先定义 interface，Demo 和真实实现都实现同一 contract。

### P2-2 幂等与确认状态需要持久化

ConfirmationGate 和 MCP 幂等表均在内存中：

- 重启后状态丢失。
- confirmed action_id 永不清理。
- 相同 action_id 携带不同 payload 时直接返回旧结果，没有冲突检测。

建议使用数据库唯一索引，并绑定 `action_id + user_id + tool_name + payload_hash`。

### P2-3 外部暴露前必须增加身份与资源授权

当前 API 无认证、CORS 为 `*`，MCP query/refund 也不校验订单归属。任何知道 order_id 的调用者都可查询或退款。

在当前“本地技术原型”定位下列为 P2；一旦对外部署，该项立即升级为发布阻断级 P0。

证据：

- `api/main.py:293-305`
- `mcp_server/customer_service_server.py:75-85,134-164`

### P2-4 Trace 和后台记忆任务需要可靠执行

Trace 只存内存；情景记忆通过未跟踪的 `asyncio.create_task()` 写入。进程退出时任务可能丢失，异常也不会进入请求结果或可靠重试队列。

建议引入持久 Trace、后台任务队列和 shutdown drain。

### P2-5 部署脚本与配置需要收敛

- `.env.example` 暴露多个未使用开关：`ENABLE_*`、`SECRET_KEY`、`JWT_SECRET_KEY`。
- `run-image.sh` 映射宿主 9090 到容器 9090，但示例配置的应用 Prometheus 端口是 9091。
- `docker-deploy.sh` 对启用密码的 Redis 执行无密码 `redis-cli ping`。
- 同时维护 `ggbot.sh`、`docker-deploy.sh`、`run-image.sh`，行为已经漂移。

建议只保留一个受测试的启动入口。

### P2-6 Trace 脱敏需要正式数据分类

当前仅按 key 名过滤并截断 preview，仍可能把订单金额、用户标识等业务敏感数据写入 trace。需要按字段 schema 做 allowlist，而不是 blacklist。

### P2-7 清理无效声明和死代码

- OrderAgent 白名单包含不存在的 `query_payment`。
- `.env.example` 的功能开关没有代码读取。
- `ANOMALY_DETECTION_THRESHOLD` 没有接入 AnomalyDetector。
- legacy 与新运行时存在重复模型和重复统计类型。

## 6. 推荐实施批次

### Batch 1：可信基线

1. 修复 Embedding 配置注入，恢复无外部 Key 的测试。
2. 重写 runtime integration eval，删除循环 oracle 和 expected-status 篡改。
3. 为本审计中的 P0 行为添加失败回归用例。

完成标准：默认命令全绿，测试结果不依赖开发机环境变量。

### Batch 2：业务动作安全

1. 建立 IntentCapability / GoalContract。
2. 拆分 Refund、Return、Cancel、Handoff workflow。
3. 引入 typed tool output 和业务错误映射。
4. 修复业务时钟。

完成标准：每个 WRITE 工具有唯一合法意图、确认策略和完成谓词。

### Batch 3：主运行时闭环

1. 让 Skills 进入 PolicyResolver。
2. 接入画像更新与受控记忆消费。
3. 接入复合任务和真实 resume state。
4. 把 monitor / eval / CLI 迁移到 CustomerAgentRuntime。

完成标准：`/chat`、监控、评测和 CLI 共享同一套 runtime 与 ToolRegistry。

### Batch 4：RAG 可信度

1. 收敛 canonical chunk store。
2. 保留 page / section / source 元数据。
3. 校准不同模式的拒答阈值。
4. 使用真实 Embedding / Reranker 数据集做离线消融。

完成标准：Citation Precision 和 Faithfulness 不再是硬编码值，并有 no-answer 指标。

### Batch 5：可部署性

1. 替换业务 Mock。
2. 加入认证、订单归属、持久幂等和审计。
3. 持久化 Trace，可靠执行后台任务。
4. 收敛部署脚本和配置。

完成标准：通过真实依赖的 smoke test、权限测试、重启恢复测试和并发压测。

## 7. 优先级判断原则

| 优先级 | 判断标准 |
|---|---|
| P0 | 会执行错误业务动作、返回明确错误事实、让测试结果失真，或导致默认测试无法运行 |
| P1 | 文档和接口已声明能力，但主链路没有真正消费；影响可解释性、效果或运行稳定性 |
| P2 | 当前本地 Demo 可暂时接受，但外部部署、并发、多实例或真实业务接入前必须完成 |
