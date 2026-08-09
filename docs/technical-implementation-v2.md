# GGBot 当前版本技术实现 v2

> Turn-level Agent Runtime · Internal RPC Tools · Hybrid RAG · Persistent Memory
>
> 本文是 2026-08-09 工作区实现的独立 v2 副本；原技术文档与历史图保持不变。

| 项目     | 内容                      | 项目     | 内容                              |
| -------- | ------------------------- | -------- | --------------------------------- |
| 当前版本 | `feat-v1 / working tree v2` | 核心场景 | 退款、退货、取消与人工工单确认闭环 |
| 验证结果 | 383 passed · 1 warning      | 技术主线 | 显式状态机 + 受约束 ReAct + RPC + RAG |

> **项目定位**
>
> 这是一个可运行、可解释、面向技术展示的智能客服原型。重点是打通 Agent 业务闭环并展示关键机制，不等同于已经完成鉴权、审计和真实业务接入的生产平台。

## 1. 一页看懂当前版本

| 能力        | 当前实现                                               | 状态           |
| ----------- | ------------------------------------------------------ | -------------- |
| 对话编排    | 自研 TurnEngine，显式状态迁移、暂停与跨轮恢复          | 主链路已落地   |
| 对话理解    | 规则 fast-track + 单次结构化 LLM + 安全降级            | 主链路已落地   |
| Multi-Agent | Knowledge、Order、Logistics、AfterSales 四类领域 Agent | 确定性路由     |
| 工具协议    | 显式 ToolSpec + 可替换内部 RPC Client                  | 主链路已落地   |
| 知识检索    | Chroma Dense + BM25 + RRF + Cross-Encoder              | 效果待继续验证 |
| 记忆系统    | Redis 工作记忆与状态，Chroma 情景记忆与画像            | 分层存储       |
| 质量体系    | Trace、Prometheus、383 个测试、50 条离线评测           | 主链路监控已接入 |

### 本次实现更新摘要

| 变化方向 | 当前实现 | 文档调整 |
| --- | --- | --- |
| 业务动作安全 | 写操作统一进入 PendingAction；售后 Planner 失败时仅退款允许确定性降级，退货/取消 fail closed | 防止售后目标错配 |
| 售后闭环 | 投诉与转人工通过 `create_ticket` 进入确认、恢复和幂等链路 | Multi-Agent 图增加退款/工单双分支 |
| 复合意图 | NLU 输出 `intents`，Runtime 按 Agent 分组 goal，并执行 completion condition | 更新 NLU、Router 和结果合并说明 |
| RPC 与幂等 | RPC Adapter 接入参数校验、超时、熔断和统计；Redis 共享 action_id 幂等 | 移除进程内写结果缓存 |
| RAG 一致性 | Chroma collection 成为 canonical chunk store，保留完整引用元数据；Dense/RRF/Reranker 使用独立阈值 | 重绘索引与证据门控图 |
| 结构化输出 | NLU、ReAct、QueryPlanner、RAG Answer、ResponsePolisher 和画像提取使用 Anthropic Tool Calling + Pydantic | 移除宽松 JSON 截取 |
| 并发与可观测性 | Redis 会话租约锁覆盖整轮提交；Chroma/Retriever 调用移出事件循环 | 补充会话事务边界和主链路监控 |

### 系统架构

![GGBot 系统架构](../diagrams/2026-08-09T120001/diagram.png)

**主运行时。** `POST /chat` 已切换到 `CustomerAgentRuntime → DialogueStateTracker → TurnEngine → DomainAgentRuntime → ToolRegistry`。FAQ 使用固定 RAG 路径；订单、物流和退款使用有界 ServiceAgent；写操作在用户确认前不会执行。

**兼容运行时。** 仓库仍保留早期 `AgentOrchestrator` 与 `MCPToolManager` 供显式 legacy 评测使用，但 `/chat`、`/search`、默认评测、CLI 和在线监控均已迁移到 CustomerAgentRuntime / ToolRegistry。Legacy evaluator 默认不初始化，仅在 `ENABLE_LEGACY_EVAL=true` 时启用。

### 启动装配与模块边界

应用通过 FastAPI lifespan 统一完成运行时装配。启动阶段先读取模型配置和售后 Skill，再建立异步 Redis、会话租约锁与 ChromaDB 连接；随后构建 StructuredLLMClient、KnowledgeRuntime、ToolRegistry、内部 RPC Client、Redis 幂等仓库和领域 Agent，最后把 Router、TurnEngine、TraceStore 组装成 CustomerAgentRuntime。关键依赖初始化失败时，应用不会进入可服务状态。

**API 层只负责协议与生命周期。** `api/main.py` 处理请求模型、组件初始化、上下文读取和响应序列化，不承载退款判断等业务决策。核心编排集中在 `core/customer_agent_runtime.py`，状态迁移集中在 `core/turn_engine.py`，领域动作集中在 `agents/domain_agents.py`，工具治理集中在 `core/tool_registry.py`。

**业务状态与执行状态分离。** DialogueState 保存 active_intent、slots、missing_slots、pending_action 和 confirmation_status，描述“业务进行到哪里”；TurnContext 保存 execution_state、state_history、observations、step_count 和 response，描述“本轮执行到哪一步”。这种拆分使业务状态可以跨轮持久化，而本轮执行轨迹可以独立限制和观测。

| 模块   | 核心对象                                     | 职责边界                                               |
| ------ | -------------------------------------------- | ------------------------------------------------------ |
| 接入层 | `FastAPI /chat`<br>`RedisConversationLockManager` | 请求校验、同会话串行化、上下文装配与结果序列化 |
| 理解层 | `IntentRecognizer`<br>`DialogueStateTracker` | 把自然语言变成结构化理解，并归并为可持久化业务状态     |
| 编排层 | `CustomerAgentRuntime`<br>`TurnEngine`       | 注册状态 handler、执行有界状态循环、处理暂停与失败     |
| 领域层 | `Router`<br>`DomainAgentRuntime`<br>`ReActPlanner` | 确定性路由，领域内受约束规划，执行并合并多个目标 |
| 能力层 | `ToolRegistry`<br>`LocalToolAdapter`<br>`InternalRPCClients` | 工具注册、白名单、参数校验、确认门禁和 RPC 调用 |
| 数据层 | Redis / ChromaDB                             | 保存 DialogueState、工作记忆、情景记忆、画像和知识索引 |

## 2. 一次请求如何执行

1. **获取会话锁并装配上下文。** API 以 `user_id + conv_id` 获取 Redis 租约锁，再读取最近消息、摘要、相关情景记忆、画像和 DialogueState；锁等待超时返回 HTTP 409。
2. **结构化理解。** 正则 fast-track 优先识别意图、订单号、物流号、确认、拒绝和槽位纠正；信号不足时通过强制 Tool Calling 调用一次 LLM，并用 Pydantic 校验工具输入。
3. **更新业务状态。** DialogueStateTracker 以纯 Reducer 方式合并旧状态与本轮 UnderstandingResult，计算 required_slots 和 missing_slots。
4. **驱动执行状态。** TurnEngine 校验状态边、执行 handler，并在每一步后把 DialogueState 持久化到 Redis。
5. **路由领域 Agent。** Router 根据 active_intent 与本轮 intents 生成 goal → Agent 映射，按 Agent 去重，不增加额外 LLM 调用。
6. **调用工具并原子提交本轮。** ToolRegistry 检查工具存在性、Agent 白名单和写操作确认状态；API 在释放会话锁前保存消息、画像和情景事件，再返回状态、引用和 trace_id。

### TurnEngine 状态路径

`UNDERSTANDING → CLARIFYING / ROUTING → RETRIEVING / ACTING → AWAITING_CONFIRMATION / RESPONDING → COMPLETED`

**暂停点：** `CLARIFYING` 与 `AWAITING_CONFIRMATION` 会保存状态并等待下一轮输入。**失败点：**异常、非法状态、缺失 handler 或超过 6 步时进入 `FAILED`，生成结构化 HandoffPackage。

### 状态机执行循环

CustomerAgentRuntime 每个请求都会通过 `turn_engine.fork()` 创建独立 handler registry，但复用同一个 StateStore。这样不同请求不会共享临时 handler，同时仍然可以从 Redis 恢复同一会话的 DialogueState。恢复位置不是直接保存 TurnContext，而是由 `resume_state()` 根据 missing_slots、pending_action 和 confirmation_status 推导，减少持久化模型与运行时代码的耦合。

TurnEngine 在进入循环前判断终态；每一步查找当前状态对应的 handler，支持同步或异步返回 Transition。Transition 在应用前必须通过允许边校验，然后统一更新 DialogueState、Observation、状态历史、step_count 和 response。状态更新完成后立即写入 StateStore，因此即使后续步骤失败，前一步已经确认的业务状态仍然可恢复。

```python
while state not in terminal_states:
    if state in paused_states and not first_step:
        break
    if step_count >= max_steps:
        return fail_with_handoff("max_steps_exceeded")
    transition = await handler(context)
    context = apply_transition(context, transition)
    await state_store.save(context.dialogue_state)
```

| 状态                    | 主要处理                      | 可能去向                                          |
| ----------------------- | ----------------------------- | ------------------------------------------------- |
| `UNDERSTANDING`         | 检查必填槽位                  | 缺槽进入 CLARIFYING，否则进入 ROUTING             |
| `ROUTING`               | 根据 DialogueState 选择 Agent | Knowledge 进入 RETRIEVING，业务 Agent 进入 ACTING |
| `RETRIEVING / ACTING`   | 执行领域 Agent 与工具         | 完成后 RESPONDING；写操作待确认时暂停             |
| `AWAITING_CONFIRMATION` | 等待下一轮确认或拒绝          | 确认后 ACTING；拒绝后 RESPONDING                  |
| `FAILED`                | 生成 HandoffPackage           | 终止自动执行，保留意图、槽位和 Observation        |

## 3. 退款申请：最小完整业务闭环

| 用户输入          | 状态                    | 系统行为                                                               |
| ----------------- | ----------------------- | ---------------------------------------------------------------------- |
| “我要退款”        | `CLARIFYING`            | 识别 `refund_request`，发现缺少 `order_id`，追问订单号                 |
| “订单号 ORD-1001” | `AWAITING_CONFIRMATION` | 继承退款意图，查询订单并核验资格，生成包含 action_id 的 PendingAction  |
| “确认”            | `COMPLETED`             | 恢复 PendingAction，通过确认门禁调用 `create_refund`，返回退款申请编号 |

### 写操作确认、恢复与幂等

**第一轮只建立目标，不执行工具。** “我要退款”被 fast-track 识别为 `refund_request`。DST 根据 INTENT_SCHEMAS 得到必填槽位 `order_id`，发现缺失后写入 missing_slots。UNDERSTANDING handler 返回 CLARIFYING Transition，状态保存后本轮暂停。

**第二轮先读后写。** 用户补充订单号后，DST 继承已有 active_intent，并把 order_id 合并进 slots。AfterSalesAgent 先调用 `query_order`，再调用 `check_refund_eligibility`。只有订单存在且符合资格时才创建 PendingAction；PendingAction 包含唯一 action_id、目标工具和参数，但此时不会调用 `create_refund`。

**第三轮恢复待执行动作。** 用户说“确认”时，NLU 识别 UserAct.CONFIRM，DST 把 confirmation_status 从 PENDING 更新为 CONFIRMED。TurnEngine 由 `resume_state()` 推导从 ACTING 恢复。AfterSalesAgent 使用原 PendingAction 的 action_id 调用 `confirm_action()`，随后 ToolRegistry 才放行 WRITE 工具。

**幂等键贯穿状态与工具。** action_id 同时存在于 PendingAction、ToolRegistry 确认门禁、内部 RPC 请求和 Redis ActionExecutionRepository 中。相同 action_id 只有 payload 指纹一致时才返回持久化结果；payload 不同返回 `idempotency_conflict`。真实业务 RPC 仍应使用数据库唯一键提供最终幂等保证。

**拒绝和失败都有显式语义。** 拒绝会清空 pending_action 并返回取消文案，不会触达写工具；订单不存在或超出退款窗口属于业务完成结果；工具超时、参数错误或 handler 异常才进入 FAILED，并生成可交给人工的 HandoffPackage。

> **关键保证**
>
> 拒绝路径不会执行写工具；重复请求以 action_id 幂等；订单不存在、不符合资格或工具失败都有明确分支，不依赖模型自由发挥。

## 4. 核心技术实现

### 4.1 NLU 与 Dialogue State Tracking

**图解阅读方式。** 沿主箭头查看自然语言如何先经过规则或 LLM 形成强类型理解结果，再由 DST 按冲突规则合并为可持久化 DialogueState；图中的琥珀色节点对应下文重点解释的校验与状态治理规则。

![NLU 与 Dialogue State Tracking](../diagrams/2026-08-09T120002/diagram.png)

| 机制              | 实现方式                                                                    |
| ----------------- | --------------------------------------------------------------------------- |
| 确定性 fast-track | 正则提取订单号、物流号；关键词识别退款、退货、取消订单、物流和订单意图      |
| 用户行为识别      | 识别 confirm、reject、switch、inform，并在意图关键词存在时避免误判确认      |
| 结构化 LLM        | fast-track 不足时强制调用 `submit_understanding` Tool，Pydantic 校验失败后降级 |
| 槽位治理          | 支持跨轮继承、显式纠正、静默覆盖抑制，以及跨意图复用 order_id / tracking_no |

**理解结果使用强类型契约。** `UnderstandingResult` 包含 intents、primary_intent、confidence、extracted_slots、corrected_slots、user_act 和 route_to。Pydantic 校验 primary_intent 必须出现在 intents 中，置信度必须位于 0 到 1，非法字段被禁止，从数据入口阻止模型输出污染状态机。

**fast-track 不是简单兜底，而是优先通道。** 订单号、运单号、确认和拒绝等高确定性信息使用正则提取；命中意图与槽位时置信度达到 0.9 以上，直接跳过远程 LLM。若当前已经存在 active_intent，而本轮只补充槽位或纠正槽位，识别器会保留原目标，避免“订单号 ORD-1001”被误切换成普通订单查询。

**用户行为识别依赖确认上下文。** confirm / reject 只在 `confirmation_status=PENDING` 时生效；显式 correction 的优先级高于 reject，避免“不对，是 ORD-1002”被误判为取消动作。配送时效、费用等通用问题进入知识查询，只有具体订单物流查询才要求 `order_id`。

**LLM 只负责规则无法覆盖的语义。** Prompt 中包含可用意图、必填槽位、当前 DialogueState 和少量示例。`StructuredLLMClient` 使用 Anthropic `tools + tool_choice` 强制模型调用 `submit_understanding`，其 input 直接进入 `_NLUOutput` 校验；缺失工具调用、未知意图、非法字段或调用异常都会回退到 `make_fallback_understanding()`，不再从自由文本中截取 JSON。

**DST 负责冲突处理。** 普通新值不能静默覆盖已有槽位，只有 corrected_slots 明确标记后才允许改写。发生 UserAct.SWITCH 时，状态机会清空 pending_action，并仅保留 order_id、tracking_no 等允许跨意图复用的槽位。工具 Observation 可以补齐缺失字段，但不会覆盖用户明确提供的值。

**IntentSchema 是业务契约。** 每个意图声明 required_slots、allowed_agents 和 completion_condition。DST 用它计算 missing_slots，Router 用它确定允许的 Agent，评测也可据此判断目标是否完成，避免意图、槽位和路由规则散落在 Prompt 中。

**复合意图从理解结果进入执行计划。** 规则与 LLM 两条路径都可以输出去重后的 `intents`。如果本轮不是显式 SWITCH，Runtime 会保护已经在处理的 active_intent；否则把多个目标交给领域 Runtime 分组执行，避免同一 Agent 因多个 goal 被重复调用。

### 4.2 Multi-Agent 与有界执行

**图解阅读方式。** 先看 Router 如何把业务意图交给领域 Agent，再看 ServiceAgent 的有界 Observation/Action 循环；KnowledgeAgent 的固定检索路径和 AfterSalesAgent 的确认分支在图中被单独展开。

![Multi-Agent 与有界执行](../diagrams/2026-08-09T120003/diagram.png)

| Agent           | 职责                 | 核心工具                                                 |
| --------------- | -------------------- | -------------------------------------------------------- |
| KnowledgeAgent  | 政策与 FAQ           | `rag_search`，固定一次检索                               |
| OrderAgent      | 订单事实             | `query_order`                                            |
| LogisticsAgent  | 订单与物流轨迹       | `query_order → track_package`                            |
| AfterSalesAgent | 退款、退货、取消与人工工单 | `create_refund` / `create_return` / `cancel_order` / `create_ticket` |

Order、Logistics 和 AfterSales 复用同一个 ServiceAgent 执行骨架，默认最多 4 步。KnowledgeAgent 不进入 ReAct 循环，避免 FAQ 场景产生无界工具规划。

**ServiceAgent 默认使用受约束的“观察—动作”循环。** ReActPlanner 每轮只能返回 tool、finish、clarify 或 handoff；工具必须来自 Agent 白名单，参数必须通过 ToolSpec 校验，相同工具与参数不可重复。循环最大 4 步，超过限制返回 `max_steps_exceeded`。`REACT_ENABLED=false` 时使用确定性 `next_action()` 路径。

**规划结果结构化且执行层保持最终控制。** ReActPlanner 通过强制 `submit_agent_decision` Tool Calling 返回 AgentDecision，不输出 Chain-of-Thought。执行层独立检查白名单、重复调用、READ/WRITE 类型与确认状态；模型选择 WRITE 工具时只创建冻结参数的 PendingAction，不直接执行副作用。

**KnowledgeAgent 单独设计。** 知识问答不需要多步业务动作，因此只调用一次 `rag_search(mode="rerank")`。返回无证据或 answered=false 时明确拒答；有证据时将重排后的 Top-N chunk 去重并纳入统一 Token 预算，再由受约束 AnswerGenerator 综合生成带引用回答。生成超时、输出非法或本地校验失败时降级为首个 chunk 的抽取式响应。它不进入 ServiceAgent 循环，从架构上限制 FAQ 工具调用成本。

**售后动作按目标显式分流。** ReAct 正常可用时，退款、退货和取消订单分别选择匹配的写工具；`complaint` / `escalation` 由确定性工单链路处理。Planner 异常时只有退款允许进入退款专用确定性 fallback，退货、取消订单和通用售后直接停止自动执行并建议人工处理，避免目标错配。

**DomainAgentRuntime 已接入复合任务。** Router 将每个 goal 映射到 Agent，再按 Agent 分组去重；多个 Agent 按确定性顺序执行。`GoalCompletionEvaluator` 根据 IntentSchema 的 completion condition 检查引用、工具 Observation 或工单结果，只有满足完成谓词的 goal 才进入 `completed_goals`，最后由 ResponseComposer 合并响应。

### 4.3 内部 RPC 与工具安全边界

**图解阅读方式。** 从 Agent 请求开始，依次核对 ToolRegistry 的工具白名单、读写分类、确认门禁，以及 Redis 对写 RPC 的幂等保护；红色支路表示调用在产生副作用前被拒绝。

![工具安全边界](../diagrams/2026-08-09T120004/diagram.png)

主应用不启动工具子进程。`register_internal_rpc_tools()` 显式注册领域 ToolSpec，LocalToolAdapter 把工具参数转换为 CommerceRPC、FulfillmentRPC 和 AfterSalesRPC 调用。Mock Client 与未来真实 HTTP/Thrift/gRPC Client 实现同一 Protocol。

| 执行约束     | 行为                                                       |
| ------------ | ---------------------------------------------------------- |
| Agent 白名单 | 每个 Agent 只能调用明确授权的工具                          |
| 读写分类     | refund / return / cancel / ticket 创建工具均标记为 WRITE   |
| 确认门禁     | WRITE 工具必须携带已确认的 action_id，否则直接拒绝执行     |
| 幂等保护     | Redis 按 tool_name + action_id 保存参数指纹、状态和结果   |
| 韧性策略     | Local Adapter 提供校验、超时、缓存、熔断、fallback 和统计 |

**工具契约由 ToolSpec 描述。** 每个工具包含 name、description、input_schema、output_schema、tool_type、timeout_s、cache_ttl 和 supports_rerank。调用前使用轻量 JSON Schema 规则检查 required、基础类型和 enum；失败时返回结构化 ToolResult，而不是直接抛出到 Agent。

**RPC 工具在启动时显式注册。** 工具名称、JSON Schema 和 READ/WRITE 类型都由代码审查控制，不依赖远端发现结果推断写权限。`/tools` 暴露当前 ToolRegistry 契约；`/mcp/tools` 仅保留为兼容别名。

**ToolRegistry 是统一治理入口。** Agent 调用工具时依次检查：工具是否注册、工具是否在 Agent 白名单、WRITE 工具是否携带 action_id、action_id 是否已确认。只有四项全部通过才会委托 Adapter 执行。白名单在 Agent 初始化时注册，因此越权调用即使工具本身存在也会被拒绝。

**Adapter 共享基础韧性。** LocalToolAdapter 提供 TTL 缓存、fallback、参数校验、asyncio 超时、熔断与 ToolStats。主监控直接读取 ToolRegistry 聚合结果，但不参与 Router 决策。

**确认状态在成功写入后释放。** ToolRegistry 委托 WRITE 工具执行成功后调用 `ConfirmationGate.complete(action_id)`，同时清理 pending 与 confirmed 集合，避免确认令牌在进程内无限累积。业务层仍以强类型输出判断 `created/found`，协议成功不等同于业务成功。

**Mock RPC 保证演示可重复。** Mock Client 不保存幂等状态，写结果编号由 action_id 稳定生成。跨实例幂等由 Redis 原子抢占和结果存储提供；替换真实业务 RPC 时仍需下游数据库唯一约束。

### 4.4 Hybrid RAG

**图解阅读方式。** 上方泳道展示文档如何保留结构并同步建立 Dense/BM25 索引，下方泳道展示在线查询如何双路召回、按排名融合、重排，并在阈值判断后生成引用或拒答。

![Hybrid RAG](../diagrams/2026-08-09T120005/diagram.png)

`Loader → Version Timeline → QueryPlanner → Multi-Query Dense + BM25 → RRF → Cross-Encoder → Evidence Assembly → Grounded Generation → Citation Validation`

| 阶段       | 实现                                                                    |
| ---------- | ----------------------------------------------------------------------- |
| 文档解析   | 支持 TXT、Markdown、PDF 和 JSON；保留 Markdown 标题路径与 PDF 页码      |
| 切片       | 模型无关 token 预算，默认 chunk_size=500、overlap=80；参数可通过环境变量调整，并使用内容哈希生成 chunk_id 与 parent_id |
| 查询规划   | 一次结构化 LLM 调用完成指代消解、独立问题生成和最多 3 条 Multi-Query；失败时回退原问题 |
| 版本时间   | knowledge_id/version/status/effective_at/expires_at 构成不重叠发布区间，支持当前与 as_of 历史检索 |
| 双路召回   | 每条 Query 分别执行 ChromaDB + BGE Dense 与进程内 BM25 检索             |
| 融合与重排 | 每条 Query 分别召回，跨 Query RRF 去重融合后只执行一次 `BAAI/bge-reranker-v2-m3` |
| 证据组装   | Top-N 按 chunk_id 和规范化正文去重，默认最多 5 个 chunk、1800 个模型无关 token |
| 生成与校验 | 只允许基于编号证据生成；每个事实段必须引用；本地校验引用及数字、期限、金额、ID、错误码 |
| 引用与拒答 | 返回 source、title、section、page、chunk_id、knowledge_id 和 version；检索或生成证据不足时拒绝回答 |

**导入阶段保持结构信息。** `load_document()` 按文件类型分发：TXT 作为单节；Markdown 按标题层级构造 section path；PDF 逐页提取并记录 page。`chunk_sections()` 使用段落、换行、中文标点和空格递归切分，再按模型无关 token 单元执行硬预算与 overlap，避免简单定长截断破坏全部语义边界。`RAG_CHUNK_SIZE_TOKENS` 与 `RAG_CHUNK_OVERLAP_TOKENS` 只影响新导入或重新索引的文档。

**Canonical chunk store 消除双重索引漂移。** KnowledgeBase 的 Chroma collection 保存统一 chunk_id、正文以及 source / title / section / page / metadata。KnowledgeRuntime 直接复用这个 collection 做 Dense 检索，并从同一批 DocumentChunk 构建 BM25；导入时优先调用 `add_chunks()`，不再创建第二套 Dense collection 或在启动时重复 Embedding。

**QueryPlanner 合并指代消解与 Multi-Query。** KnowledgeAgent 将最近 5 条消息、active_intent 和结构化 slots 交给 QueryPlanner，一次调用返回 standalone_query、候选查询和 resolved_references。系统校验订单号、错误码和数字等硬实体没有被删除；模型失败、输出非法或低于置信度阈值时保留原问题。原问题始终参与召回，避免改写偏移造成零召回。

**版本时间线保留历史但隔离召回。** 每个逻辑知识使用稳定 knowledge_id，每个版本使用唯一 version_id；effective_at 与 expires_at 采用 UTC epoch 和左闭右开区间。同一知识的已发布版本按生效时间自动闭合前一版本，draft/revoked 或查询时间点无效的 Chunk 在 Dense 与 BM25 两路检索前被过滤。旧版本不物理删除，可通过 `/search?as_of=...` 重放历史口径。

**Dense 负责语义相似，BM25 负责精确词项。** Dense 使用 Chroma cosine distance 转换为相似度，适合语义改写；BM25 使用英文 token 与中文单字 token，适合订单规则名、错误码和关键词命中。两路各自取 candidate_k 后进入 RRF，不直接比较两种不可同量纲的原始分数。

**RRF 按排名而不是原始分数融合。** 每个 chunk 的融合分数为 `Σ 1 / (rrf_k + rank)`，默认 rrf_k=60。相同 chunk 在两路同时出现时分数累加，并保留 dense_score、bm25_score 和 rrf_score，便于调试召回来源。

**Cross-Encoder 只处理融合候选。** 启用 reranker 时，将 query 与候选 chunk 组成 pair 批量打分，用 rerank_score 覆盖最终排序分数，再截取 top_k。这样把计算量限制在候选集合，而不是对整个知识库做交叉编码。

**AnswerGenerator 在受控证据窗口内综合回答。** `RAGAnswerGenerator` 保持重排顺序，按 chunk_id 和规范化正文去重，并在 `max_chunks` 与 `max_context_tokens` 双重预算下组装 evidence。生成模型严格返回 answer、used_citations 和 sufficient_evidence；证据冲突或不足时必须拒答。输出还会经过引用集合、逐段引用以及新增数字、期限、金额、ID、错误码检查，校验失败不会直接面向用户。

**Citation 来自检索元数据。** 最终结果为每个命中生成 Citation，包含 citation_id、chunk_id、source、title、section 和 page。AnswerGenerator 只能使用实际进入证据窗口的引用，KnowledgeAgent 最终只返回回答实际使用的 Citation。Dense-only、RRF 和 Reranker 分别使用 `dense_threshold`、`rrf_threshold` 和 `rerank_threshold`，不再用同一个数值比较不同量纲的分数。

**当前效果结论保持克制。** 三类阈值已有非零默认值，但仍需要真实 no-answer 数据校准；本地确定性评测中的 Dense 和 Hybrid 指标相同，FakeReranker 还降低了 MRR。因此代码链路与拒答机制已经实现，但 BM25 和重排是否带来效果增益仍需真实模型与更有区分度的数据集验证。

### 4.5 记忆与持久化

**图解阅读方式。** 以 MemoryContext 装配为中心，向外查看结构化状态、工作记忆、情景记忆和用户画像四条读写链路；不同颜色同时表示存储层级和写入门控。

![记忆与持久化](../diagrams/2026-08-09T120006/diagram.png)

| 数据          | 存储       | 策略                                           |
| ------------- | ---------- | ---------------------------------------------- |
| DialogueState | Redis      | 每个状态机步骤保存，TTL 24 小时                |
| 工作记忆      | Redis List | 15 条触发覆盖式摘要，保留最近 5 条             |
| 情景记忆      | ChromaDB   | 仅任务完成或转人工时写入，支持跨会话检索       |
| 用户画像      | ChromaDB   | 稳定偏好信号门控；过滤订单号、物流号等时效事实 |

**DialogueState 与自然语言记忆分开保存。** RedisStateStore 使用 `dst:{user_id}:{conv_id}` 作为键，保存 Pydantic JSON，默认 TTL 24 小时。TurnEngine 只读写这个结构化状态；MemoryManager 负责消息、摘要、情景记忆和画像，两者共享 Redis 连接但不共享 key，避免摘要文本成为业务状态真源。

**工作记忆使用有界压缩。** 每轮消息写入 Redis List，达到 15 条时触发压缩。系统把旧摘要与待压缩消息一起交给 LLM，生成一份全新的覆盖式摘要，最大 600 字，然后只保留最近 5 条原始消息。覆盖而不是追加可以阻止摘要随轮次无限膨胀。

**情景记忆采用事件触发写入。** 普通消息和摘要压缩不会写入 Chroma episodic collection；只有任务完成或转人工时，API 才在会话锁释放前调用 `record_episodic_event()`。这确保事件只包含当前轮已提交的消息，不会混入同一 conv_id 的下一轮请求。

**用户画像有稳定偏好门控。** 只有近期文本命中“我喜欢”“以后都”“请用中文”等稳定偏好信号时才调用 LLM 提炼画像；结果必须通过字段白名单，并过滤订单号、运单号、退款状态等时效事实。画像更新采用读取旧值、合并、删除旧文档、写入新文档的覆盖方式。

**上下文装配顺序固定。** MemoryContext 将 DialogueState、最近消息、会话摘要、相关历史、用户画像和 Observation 分区输出。售后 Skill 不在 API 层全局注入，而是在路由确定 `AfterSalesAgent + intent` 后单独解析并传给 ReActPlanner。

**同步存储调用不阻塞事件循环。** `_backend_call()` 识别异步客户端；对于同步 Redis / Chroma 方法统一使用 `asyncio.to_thread()`。RAG 工具和 `/search` 也把同步 Retriever 查询移出事件循环，避免单次向量检索阻塞并发请求。

**会话事务覆盖记忆提交。** Redis 分布式锁覆盖上下文读取、Agent 执行、DialogueState 保存、消息写入和情景记忆事件。相同 user_id + conv_id 串行，不同会话仍可并发。

## 5. 可观测性与质量验证

| Agent Trace                                                                   | 在线监控                                                          |
| ----------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| 记录理解摘要、Agent 结果、工具名、状态路径和延迟                              | 采集 Agent / Tool 成功率和延迟，使用滑动窗口 Z-score 检测异常     |
| 顶层事件与 Observation preview 都使用字段 allowlist，不保存用户原文、Prompt 或隐藏推理 | CustomerAgentRuntime 与 ToolRegistry 统计已接入 Prometheus、Webhook 和异常检测 |

### Trace 与在线监控实现

**Trace 记录公开执行事实，不记录隐藏推理。** CustomerAgentRuntime 在 understanding、agent_result、rag_retrieval 和 turn_end 四类节点写入事件。事件包含意图、槽位名、Agent、工具名、成功状态、状态路径和延迟，但不保存用户原文、完整 Prompt、检索 query 或 Chain-of-Thought。

**Observation 会先做摘要和字段级 allowlist。** `summarize_observations()` 最多保留 8 条 Observation，仅输出 source、name、success，以及状态、结果标志和引用元数据。订单号、用户标识、金额、原文、Prompt 和隐藏推理默认不进入 Trace；失败错误统一降级为 operation_failed。

**TraceStore 当前是有界内存实现。** 默认最多保存 1000 个 trace_id，超过后淘汰最早记录；`GET /traces/{trace_id}` 用于演示和调试。它验证了 Trace 数据模型与脱敏策略，但服务重启后会丢失，不属于生产级持久化追踪系统。

**PerformanceMonitor 周期拉取运行统计。** 监控任务按 interval 读取 CustomerAgentRuntime 与 ToolRegistry 的成功率、平均延迟、连续失败数和熔断状态。滑动窗口 Z-score 用于识别突变，固定阈值用于告警；可选 Webhook 异步发送告警，Prometheus 暴露 Gauge 和 Counter。

**主链路统计已经统一。** CustomerAgentRuntime 按领域 Agent 记录请求数、成功率和延迟；LocalToolAdapter 提供工具统计与熔断状态；`/monitor` 和 `/health` 直接读取这些数据。Monitor 只负责观测和告警，不计算或写回动态路由权重。

### 本地确定性评测

| 指标                       | 结果          | 判断                  |
| -------------------------- | ------------- | --------------------- |
| Intent Accuracy / Macro-F1 | 1.000 / 1.000 | 固定样本全部命中      |
| User Act Accuracy          | 1.000         | 确认、拒绝和纠正独立计分 |
| Slot F1 / DST JGA          | 1.000 / 1.000 | 状态样本全部命中      |
| Recall@5 / MRR             | 0.900 / 0.850 | Dense 与 Hybrid 相同  |
| Hybrid + Reranker MRR      | 0.525         | 当前测试排序下降      |
| Tool Selection / Parameter | 1.000 / 1.000 | 工具路径与参数全部命中 |
| Task Completion Rate       | 1.000         | 10 条 E2E 状态均符合预期 |
| Citation / Faithfulness    | 0.409 / 0.900 | 确定性证据覆盖口径    |

> **评测口径限制**
>
> 这些结果只代表 50 条本地确定性样本。现有消融没有证明 BM25 或 Cross-Encoder 优于 Dense-only；synthetic baseline 也不能用于宣称线上效果提升。

### 评测如何执行

确定性评测由 `evaluation/local_eval_runner.py` 驱动，固定读取 50 条样本。Tool 与 E2E 用例执行真实的 CustomerAgentRuntime、TurnEngine、领域 Agent 和 ToolRegistry；RAG 消融为保证离线可重复，仍使用 FakeDenseIndex、真实 BM25 和确定性 FakeReranker，不访问远程 LLM，也不下载本地模型。

样本分为 DST、Tool、RAG、E2E 和 NLU 五组。指标层分别计算 Intent Accuracy / Macro-F1、Slot Precision / Recall / F1、DST Joint Goal Accuracy、Recall@5、MRR、工具选择与参数准确率、任务完成率、Citation Precision 和 Faithfulness。

消融会在 dense、hybrid、rerank 三种模式下重复执行同一批样本。当前 FakeDenseIndex 依赖 token overlap，FakeReranker 依赖稳定哈希，因此结果适合做代码回归，不等价于 BGE 与 Cross-Encoder 的真实离线效果。另一个 legacy EndToEndEvaluator 支持 LLM-as-Judge，但受模型波动和调用成本影响，不作为当前确定性报告的依据。

Chunking 另有独立黄金评测集 `data/eval/rag_chunking_cases.json`。`evaluation/chunking_eval.py` 实际执行 `load_document → chunk_sections → BM25Index`，检查 Retrieval Hit Rate、MRR、证据完整率、Section 准确率和 token 预算违规；该评测不使用 FakeDense 或 FakeReranker，也不代表真实向量模型效果。

全量 pytest 覆盖状态转移、会话并发锁、内部 RPC、Redis 共享幂等、原生 Tool Calling、售后 Skill、Planner 降级隔离、确认门禁、RAG Loader/索引/融合、知识版本、记忆、监控、Trace 和 API 回归。当前验证结果为 383 passed；这说明实现行为可回归，但不代表真实业务数据上的模型效果已经达标。

## 6. API 与部署形态

| 领域 | 接口                                                 | 用途                            |
| ---- | ---------------------------------------------------- | ------------------------------- |
| 对话 | `POST /chat`<br>`POST /search`<br>`GET /traces/{trace_id}` | 执行主链路、统一检索并查询公开 Trace |
| 知识 | `POST /knowledge/add`<br>`POST /knowledge/upload`<br>`GET /knowledge/{knowledge_id}/versions`<br>`POST /knowledge/{knowledge_id}/versions/{version}/publish`<br>`POST /knowledge/{knowledge_id}/versions/{version}/revoke` | 导入知识并管理版本发布、撤销和时间线 |
| 工具 | `GET /tools`<br>`GET /mcp/tools`（兼容）             | 查看当前内部 RPC ToolSpec       |
| 运营 | `GET /skills`<br>`POST /skills/reload`               | 查看和热加载业务 Skill          |
| 质量 | `POST /eval/run`<br>`GET /health`<br>`GET /monitor`<br>`GET /metrics` | 就绪检查、评测、监控与指标 |

**部署栈。** Python 3.12 + FastAPI/Uvicorn + Redis 7 + ChromaDB 0.5.23 + Prometheus + Nginx。Dockerfile 使用多阶段构建和非 root 用户；Docker Compose 配置健康检查、持久卷和服务依赖。

### 应用启动顺序与配置

lifespan 启动时首先校验 `ANTHROPIC_API_KEY`，读取模型、Skills 目录和 Redis/Chroma 地址。SkillManager 只向 AfterSalesAgent 提供匹配当前售后意图的软策略；OrderAgent、LogisticsAgent 和 KnowledgeAgent 不注入 Skill。权限、确认、资格和幂等始终由代码与 RPC 保证。

Redis 同时服务 DialogueState 与工作记忆，但使用不同 key 空间；ChromaDB 优先连接独立服务，失败后回退本地 PersistentClient。KnowledgeBase 为空时写入演示知识，KnowledgeRuntime 再基于 collection 构建 Dense/BM25/Reranker 链路。若启用本地 BGE 模型，首次启动需要准备模型缓存。

内部 RPC 默认装配可重复的 Mock Client，未来可替换为真实 HTTP、Thrift 或 gRPC Client。应用退出时按 best-effort 顺序停止 Monitor 并关闭 MemoryManager。

| 关键配置                     | 作用                                                     |
| ---------------------------- | -------------------------------------------------------- |
| `ANTHROPIC_MODEL / ANTHROPIC_BASE_URL` | 选择结构化 NLU、摘要和 legacy Agent 使用的模型与兼容端点 |
| `REDIS_URL`                  | DialogueState、工作记忆和会话摘要连接地址                |
| `ACTION_IDEMPOTENCY_TTL_S`   | 写 RPC 幂等状态与结果的 Redis 保留时间                   |
| `CONVERSATION_LOCK_LEASE_S`  | 同一会话分布式锁租约时间                                 |
| `CONVERSATION_LOCK_WAIT_TIMEOUT_S` | 获取同一会话锁的最长等待时间                       |
| `CONVERSATION_LOCK_RETRY_INTERVAL_S` | 会话锁竞争重试间隔                                |
| `CHROMA_HOST / CHROMA_PORT`  | 知识库、情景记忆和用户画像的 ChromaDB 服务地址           |
| `RAG_EMBEDDING_PROVIDER`     | 选择 `api / local / off` 检索模型策略                     |
| `RAG_DENSE_THRESHOLD`        | Dense-only 模式的证据阈值                                |
| `RAG_RRF_THRESHOLD`          | Hybrid RRF 模式的证据阈值                                |
| `RAG_RERANK_THRESHOLD`       | Cross-Encoder 重排后的证据阈值                           |
| `RAG_CHUNK_SIZE_TOKENS`      | 新导入文档的最大 chunk token 预算                        |
| `RAG_CHUNK_OVERLAP_TOKENS`   | 新导入文档的相邻 chunk 重叠 token 预算                   |
| `RAG_MULTI_QUERY_ENABLED`     | 是否启用指代消解与 Multi-Query QueryPlanner              |
| `RAG_MULTI_QUERY_MAX_QUERIES` | 单次知识检索允许的最大查询数量，默认 3                   |
| `RAG_QUERY_REWRITE_MIN_CONFIDENCE` | 接受 LLM 问题改写的最低置信度                       |
| `RAG_GENERATION_ENABLED`      | 是否启用 Top-N 证据约束生成，默认开启                    |
| `RAG_ANSWER_MAX_CHUNKS`       | 单次回答最多组装的重排证据数，默认 5                     |
| `RAG_ANSWER_MAX_CONTEXT_TOKENS` | 单次回答的证据 Token 预算，默认 1800                   |
| `RAG_ANSWER_TIMEOUT_S`        | 证据生成超时，超时后降级为抽取式回答                    |
| `REACT_ENABLED`               | 是否启用领域受约束 ReAct；关闭后使用确定性 fallback     |
| `RESPONSE_POLISH_*`           | 回答润色开关、长度门槛与超时                            |
| `GGBOT_SKILLS_DIR`           | 业务 Skill 的扫描目录，支持运行时 reload                 |
| `ENABLE_LEGACY_EVAL`         | 仅在显式需要时初始化旧 LLM evaluator                     |

Docker Compose 将 Redis、ChromaDB、Prometheus、GGBot 和 Nginx 放在同一网络内，通过服务健康检查控制启动依赖。Redis 开启 AOF，Chroma 和 Prometheus 使用持久卷；应用容器以非 root 用户运行，并把知识数据、评测报告、Skills 和日志映射到宿主机目录。

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
docker compose up -d redis chromadb
export ANTHROPIC_API_KEY=your_key
.venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
.venv/bin/python -m pytest -q
.venv/bin/python -m evaluation.local_eval_runner
.venv/bin/python -m evaluation.chunking_eval
```

## 7. 当前限制与演进优先级

| 优先级 | 方向              | 目标                                                               |
| ------ | ----------------- | ------------------------------------------------------------------ |
| **P0** | 真实效果基线 | 使用真实 Embedding / Reranker、no-answer 与引用支持标注重新评测 |
| **P0** | 真实业务 E2E | 验证缺参、纠正、复合意图、action_id 与业务拒绝全路径 |
| **P1** | 最终副作用安全 | 在真实下游以数据库唯一约束兜底 Redis action_id 幂等 |
| **P1** | PolicyResolver | 将售后 Skill 从软策略升级为可审计的结构化策略输入 |
| **P2** | 可部署性 | 接入真实业务 Gateway、持久 Trace、审计、限流与压测 |
| **发布阻断** | 身份与资源授权 | 对外部署前必须补齐认证、订单归属校验和最小权限 |

---

**关键代码入口**

`api/main.py` · `core/customer_agent_runtime.py` · `core/turn_engine.py` · `core/conversation_lock.py` · `core/structured_llm.py` · `core/idempotency.py` · `core/internal_rpc.py` · `core/rpc_tools.py` · `agents/domain_agents.py` · `core/tool_registry.py` · `rag/` · `memory/conversation_memory.py`
