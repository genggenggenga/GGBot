# GGBot

GGBot 是一个用于展示智能客服、Agent 编排与 Hybrid RAG 的技术原型。项目以退款申请为主链路，覆盖结构化 NLU、Dialogue State Tracking、自研 Turn-level 状态机、领域 Agent、标准 MCP、工具确认门禁、引用式 RAG、记忆和离线评测。

## 核心架构

```text
FastAPI /chat
  -> Structured NLU
  -> DialogueStateTracker
  -> TurnEngine
  -> Deterministic Router
     -> KnowledgeAgent -> Hybrid RAG
     -> OrderAgent -> MCP order tools
     -> LogisticsAgent -> MCP logistics tools
     -> AfterSalesAgent -> MCP refund tools
  -> Redis State / Memory
  -> Agent Trace
```

`OrderAgent`、`LogisticsAgent` 和 `AfterSalesAgent` 复用同一个有界 `ServiceAgent`。FAQ 使用固定 RAG 路径，退款写操作必须先进入 `AWAITING_CONFIRMATION`，用户确认后才允许调用 `create_refund`。

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

## MCP Demo

客服工具由标准 MCP stdio Server 提供：

```bash
.venv/bin/python -m mcp_server.customer_service_server
.venv/bin/python -m pytest -q tests/test_mcp_integration.py
```

Server 提供 `query_order`、`track_package`、`check_refund_eligibility`、`create_refund` 和 `create_ticket`。Mock 数据仅用于可重复演示，不依赖真实电商后台。

## RAG 与评测

RAG 链路由 ChromaDB Dense、独立 BM25、RRF、Cross-Encoder Reranker 和 Citation 组成。50 条固定评测样本位于 `data/eval/customer_agent_cases.json`。

文档切片按模型无关 token 预算执行，默认每个 chunk 最多 500 tokens、相邻 chunk 重叠 80 tokens。可通过环境变量调整：

```bash
export RAG_CHUNK_SIZE_TOKENS=500
export RAG_CHUNK_OVERLAP_TOKENS=80
```

修改参数只影响新导入或重新索引的文档。Chunking 黄金评测集位于 `data/eval/rag_chunking_cases.json`，实际执行文档解析、切片和 BM25 检索：

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m evaluation.local_eval_runner
.venv/bin/python -m evaluation.chunking_eval
.venv/bin/python -m evaluation.chunking_eval --chunk-size 256 --chunk-overlap 32
curl -s -X POST http://localhost:8000/eval/run \
  -H 'Content-Type: application/json' \
  -d '{"mode":"customer_agent"}'
curl -s http://localhost:8000/traces/<trace_id>
```

最新本地确定性评测报告位于 `data/eval/eval_report.md`。该报告实际逐条执行 50 条固定样本，结果如下：

| 指标 | 实测值 |
|------|--------|
| Intent Accuracy / Macro-F1 | 1.0000 / 1.0000 |
| User Act Accuracy | 1.0000 |
| Slot F1 / DST Joint Goal Accuracy | 1.0000 / 1.0000 |
| Recall@5 / MRR | 0.9000 / 0.8500 |
| Tool Selection / Parameter Accuracy | 1.0000 / 1.0000 |
| Task Completion Rate | 1.0000 |
| Citation Precision / Faithfulness | 0.4091 / 0.9000 |

消融结果位于 `data/eval/ablation_report.json`。Dense 与 Hybrid 的 Recall@5、MRR 均为 `0.9000`、`0.8500`；Hybrid + Reranker 的 Recall@5 仍为 `0.9000`，MRR 降至 `0.5250`。这说明当前确定性评测集没有证明 BM25 融合带来增益，测试用 Reranker 也未改善排序，不能据此宣称检索效果提升。

Tool 与 E2E 用例现在通过真实 `CustomerAgentRuntime → TurnEngine → DomainAgentRuntime → ToolRegistry` 执行。`citation_precision` 和 `faithfulness_rate` 仍是确定性 fixture 上的证据覆盖指标，不等价于真实模型回答的语义忠实度。

`data/eval/baseline_comparison.json` 的 `comparison_type` 为 `synthetic_or_legacy_rules`。该文件仅用于本地回归对照，不是真实线上基线，也不能用于宣称生产效果或线上提升。
