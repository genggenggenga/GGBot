# GGBot v2

GGBot 是一个用于展示智能客服、Agent 编排与 Hybrid RAG 的技术原型。v2 覆盖原生结构化输出、同会话分布式锁、受约束 ReAct、内部 RPC Tool、两阶段确认、共享幂等、Grounded RAG、分层记忆和离线评测。

详细设计见：

- [技术实现 v2](docs/technical-implementation-v2.md)
- [实现缺口审计 v2](docs/implementation-gap-audit-v2.md)
- [Prompt 设计 v2](docs/prompt-design-v2/README.md)

## 核心架构

```text
FastAPI /chat
  -> Redis conversation lease lock
  -> Fast Track / Structured Tool Calling NLU
  -> DialogueStateTracker
  -> TurnEngine
  -> Deterministic Router
     -> KnowledgeAgent -> QueryPlanner -> Hybrid RAG -> Grounded Answer
     -> OrderAgent -> bounded ReAct -> Commerce RPC tools
     -> LogisticsAgent -> bounded ReAct -> Fulfillment RPC tools
     -> AfterSalesAgent -> common + intent-specific Skill -> bounded ReAct -> AfterSales RPC tools
  -> Redis State / Memory
  -> Agent Trace
```

`OrderAgent`、`LogisticsAgent` 和 `AfterSalesAgent` 复用同一个有界 `ServiceAgent`。FAQ 使用固定 RAG 路径；退款、退货、取消订单和工单创建必须先进入 `AWAITING_CONFIRMATION`。Planner 失败时，非退款售后不会降级到退款流程。

售后请求按意图组合 Skill：所有请求加载通用安全基线，再加载退款、退货、取消订单、投诉/转人工或通用请求的专属 SOP。Skill 仅约束规划顺序和话术，不能改变 ToolRegistry 的白名单、确认门禁或 RPC 事实。

## 本地启动

建议使用 Python 3.12。部分固定依赖在 Python 3.14 下没有可用 wheel。

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
docker compose up -d redis chromadb

export ANTHROPIC_API_KEY=your_key
.venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## 退款闭环演示

所有请求使用相同的 `conv_id`：

```bash
curl -s http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"demo","conv_id":"refund-demo","message":"我要退款"}'

curl -s http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"demo","conv_id":"refund-demo","message":"订单号 ORD-1001"}'

curl -s http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"demo","conv_id":"refund-demo","message":"确认"}'
```

三轮分别验证缺槽追问、订单与退款资格查询、确认后创建退款。响应保留原有字段，并新增 `trace_id`、`status`、`missing_slots` 和 `citations`。

## 内部 RPC Mock

主应用不再启动 MCP stdio 子进程。ToolRegistry 通过显式 ToolAdapter
直接调用可替换的内部 RPC Client；本地默认使用确定性 Mock：

```bash
.venv/bin/python -m pytest -q tests/test_internal_rpc_and_structured.py
curl -s http://localhost:8000/tools
```

RPC Client 提供订单、物流、售后和工单操作。写操作通过 Redis
`action_id` 幂等仓库跨实例去重；Mock 数据仅用于可重复演示。

同一 `user_id + conv_id` 的请求通过 Redis 租约锁串行执行。锁覆盖上下文读取、Agent 执行、DialogueState 保存、消息写入与情景事件提交；等待超时返回 HTTP 409。

## 结构化 LLM

主运行时不再从模型自由文本中截取 JSON。`StructuredLLMClient` 使用 Anthropic Tool Calling 强制模型返回 Pydantic 可校验的工具参数，覆盖：

- NLU UnderstandingResult；
- ReAct AgentDecision；
- RAG QueryPlan 与 Grounded Answer；
- ResponsePolisher；
- 用户画像提取。

缺失工具调用、Schema 不匹配、超时或校验失败均进入各自的安全降级路径。

## RAG 与评测

RAG 链路由 ChromaDB Dense、独立 BM25、RRF、Cross-Encoder Reranker、受约束 Top-N 证据生成和 Citation 校验组成。评测采用分层的 Code-based Evaluator 与可选 LLM-as-Judge；详见 [评测框架](docs/evaluation-framework.md)。

文档切片按模型无关 token 预算执行，默认每个 chunk 最多 500 tokens、相邻 chunk 重叠 80 tokens。可通过环境变量调整：

```bash
export RAG_CHUNK_SIZE_TOKENS=500
export RAG_CHUNK_OVERLAP_TOKENS=80
export RAG_MULTI_QUERY_ENABLED=true
export RAG_MULTI_QUERY_MAX_QUERIES=3
export RAG_QUERY_REWRITE_MIN_CONFIDENCE=0.5
export RAG_GENERATION_ENABLED=true
export RAG_ANSWER_MAX_CHUNKS=5
export RAG_ANSWER_MAX_CONTEXT_TOKENS=1800
export RAG_ANSWER_TIMEOUT_S=8
export REACT_ENABLED=true
export ACTION_IDEMPOTENCY_TTL_S=86400
export CONVERSATION_LOCK_LEASE_S=60
export CONVERSATION_LOCK_WAIT_TIMEOUT_S=30
export CONVERSATION_LOCK_RETRY_INTERVAL_S=0.05
```

修改参数只影响新导入或重新索引的文档。Chunking 黄金评测集位于 `data/eval/rag_chunking_cases.json`，实际执行文档解析、切片和 BM25 检索：

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m evaluation.run --suite smoke --mode deterministic
.venv/bin/python -m evaluation.run --suite bad_cases --mode deterministic
.venv/bin/python -m evaluation.run --suite golden --mode deterministic
.venv/bin/python -m evaluation.run --suite golden --mode deterministic --judge
.venv/bin/python -m evaluation.chunking_eval
.venv/bin/python -m evaluation.chunking_eval --chunk-size 256 --chunk-overlap 32
curl -s -X POST http://localhost:8000/eval/run \
  -H 'Content-Type: application/json' \
  -d '{"mode":"customer_agent"}'
curl -s http://localhost:8000/traces/<trace_id>
```

KnowledgeAgent 会在一次结构化 LLM 调用中完成指代消解与 Multi-Query 改写，保留原始问题，并将最多 3 条查询分别执行 Dense/BM25 召回；候选跨查询 RRF 融合后只执行一次 Reranker。模型调用失败、置信度不足或修改了错误码/数字等硬实体时，自动回退原问题。

重排后的 Top-N 证据会按 chunk 和正文去重，并受统一 Token 预算约束。AnswerGenerator 只能基于编号证据生成回答，每个事实段必须携带合法引用；新增数字、期限、金额、ID 或错误码会被本地校验拒绝。模型明确判断证据不足时返回拒答；生成超时、非法 JSON 或校验失败时降级为原有 Top-1 抽取式回答。

知识导入支持版本和生效区间：

```bash
curl -s -X POST http://localhost:8000/knowledge/add \
  -H 'Content-Type: application/json' \
  -d '{
    "documents": [{
      "title": "退款政策",
      "content": "退款将在审核通过后原路退回。",
      "knowledge_id": "refund-policy",
      "version": "v2",
      "effective_at": "2026-08-08T00:00:00Z"
    }]
  }'

curl -s http://localhost:8000/knowledge/refund-policy/versions
curl -s -X POST \
  http://localhost:8000/knowledge/refund-policy/versions/v2/revoke
curl -s -X POST \
  'http://localhost:8000/search?query=退款政策&as_of=2026-08-08T00:00:00Z'
```

同一 `knowledge_id` 的已发布版本会按 `effective_at` 自动形成不重叠时间线。旧版本保留用于审计和历史检索，`draft`、`revoked` 以及查询时间点未生效或已失效的版本不会参与召回。

评测输入数据位于 `data/eval/`，运行产生的 JSON 与 Markdown 报告统一写入 `data/eval/reports/`，该目录不纳入版本控制。当前 Smoke、Bad-case、Golden Candidate 和 Chunking 集分别包含 90、50、280 和 80 条用例。

Tool 与 E2E 用例通过 `CustomerAgentRuntime → TurnEngine → DomainAgentRuntime → ToolRegistry` 执行，但默认仍使用确定性 Mock Tool、Fake Dense/Reranker 与固定语料。因此报告仅用于本地代码和契约回归，不可用于宣称真实模型、真实知识库或线上业务效果。
