"""
GGBot 智能客服系统 — FastAPI 入口

启动时打印小熊饼干图案。
所有核心组件在 lifespan 中初始化，通过环境变量配置。
"""
import asyncio
import logging
import os
import pathlib
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional


_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BANNER = r"""
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
   ╔══════════════════════╗
   ║   GGBot  v2.0     ║
   ║   智能客服 AI 系统    ║
   ╚══════════════════════╝
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
"""

# ── 全局组件（lifespan 中初始化）─────────────────────────────────────────────
_orchestrator = None
_memory       = None
_monitor      = None
_evaluator    = None
_skill_manager = None
_customer_runtime = None
_mcp_client = None
_trace_store = None
_knowledge_runtime = None
_tool_registry = None
_background_tasks: set[asyncio.Task] = set()


def _spawn_background_task(coro, *, name: str) -> asyncio.Task:
    """Track fire-and-forget work so failures and shutdown are observable."""
    task = asyncio.create_task(coro, name=name)
    _background_tasks.add(task)

    def done(completed: asyncio.Task) -> None:
        _background_tasks.discard(completed)
        if completed.cancelled():
            return
        error = completed.exception()
        if error is not None:
            logger.error("后台任务 %s 失败: %s", completed.get_name(), error)

    task.add_done_callback(done)
    return task


async def _drain_background_tasks(timeout_s: float = 5.0) -> None:
    """Wait for tracked work during shutdown, then cancel stragglers."""
    if not _background_tasks:
        return
    tasks = list(_background_tasks)
    done, pending = await asyncio.wait(tasks, timeout=timeout_s)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    for task in done:
        _background_tasks.discard(task)

def _anthropic_cfg() -> Dict[str, Any]:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("未设置 ANTHROPIC_API_KEY")
    cfg: Dict[str, Any] = {
        "api_key":  key,
        "model":    os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022").strip(),
    }
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()
    if base_url:
        cfg["base_url"] = base_url
    return cfg


@asynccontextmanager
async def _runtime_components(app: FastAPI):
    global _orchestrator, _memory, _monitor, _evaluator, _skill_manager
    global _customer_runtime, _mcp_client, _trace_store, _knowledge_runtime
    global _tool_registry

    print(BANNER, flush=True)

    from agents.domain_agents import (
        AfterSalesAgent,
        DomainAgentRuntime,
        KnowledgeAgent,
        LogisticsAgent,
        OrderAgent,
        Router,
    )
    from core.customer_agent_runtime import CustomerAgentRuntime
    from core.dialogue_state_tracker import DialogueStateTracker
    from core.intent_recognizer import IntentRecognizer
    from core.mcp_adapter import MCPClient, MCPToolAdapter
    from core.react_planner import ReActPlanner
    from core.state_store import RedisStateStore
    from core.tool_registry import ToolRegistry
    from core.trace_store import TraceStore
    from core.turn_engine import TurnEngine
    from mcp.knowledge_base import KnowledgeBase
    from memory.conversation_memory import MemoryManager
    from monitor.performance_monitor import PerformanceMonitor
    from core.skill_loader import SkillManager
    from rag.runtime import KnowledgeRuntime, local_models_enabled
    from rag.query_planner import QueryPlanner
    from rag.tool import register_rag_tool
    from core.llm_utils import extract_text_content
    import redis

    cfg = _anthropic_cfg()
    logger.info(f"模型: {cfg['model']}  base_url: {cfg.get('base_url', '(官方)')}")

    # 意图识别器（Orchestrator 内部也会创建，这里单独暴露给 Evaluator）
    recognizer = IntentRecognizer(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    # Skills：启动时从目录加载业务能力说明，并在 Agent 调用 LLM 时动态注入。
    skills_dir = os.getenv("GGBOT_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills"))
    _skill_manager = SkillManager(
        root_dir=skills_dir,
        max_prompt_chars=int(os.getenv("GGBOT_SKILLS_MAX_PROMPT_CHARS", "5000")),
    )
    _skill_manager.load()

    # 记忆与 Dialogue State 使用同一个 Redis 连接，但保存到不同 key。
    redis_client = redis.from_url(
        os.getenv("REDIS_URL", "redis://redis:6379/0"),
        decode_responses=True,
    )
    state_store = RedisStateStore(redis_client)
    _memory = MemoryManager(
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        state_store=state_store,
        redis_client=redis_client,
    )

    # RAG 知识库（基于 ChromaDB 的真实检索）
    kb = KnowledgeBase(
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
    )
    logger.info(f"知识库已加载: {kb.doc_count} 个文档片段")

    # 新运行时统一通过 ToolRegistry 调用 Hybrid RAG 与标准 MCP 工具。
    registry = ToolRegistry()
    _tool_registry = registry
    legacy_threshold = os.getenv("RAG_RELEVANCE_THRESHOLD")
    _knowledge_runtime = KnowledgeRuntime.build(
        kb,
        enable_local_models=local_models_enabled(),
        embedding_provider=os.getenv("RAG_EMBEDDING_PROVIDER"),
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
        reranker_model=os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        relevance_threshold=(
            float(legacy_threshold)
            if legacy_threshold not in {None, ""}
            else None
        ),
        dense_threshold=float(os.getenv("RAG_DENSE_THRESHOLD", "0.2")),
        rrf_threshold=float(os.getenv("RAG_RRF_THRESHOLD", "0.01")),
        rerank_threshold=float(os.getenv("RAG_RERANK_THRESHOLD", "0.1")),
    )
    register_rag_tool(registry, _knowledge_runtime.retriever)

    async def plan_query(prompt: str) -> str:
        response = await recognizer.client.messages.create(
            model=cfg["model"],
            max_tokens=512,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        return extract_text_content(response.content)

    query_planner = QueryPlanner(
        plan_query,
        enabled=os.getenv(
            "RAG_MULTI_QUERY_ENABLED",
            "true",
        ).lower() in {"1", "true", "yes", "on"},
        max_queries=int(os.getenv("RAG_MULTI_QUERY_MAX_QUERIES", "3")),
        min_confidence=float(
            os.getenv("RAG_QUERY_REWRITE_MIN_CONFIDENCE", "0.5"),
        ),
    )
    react_planner = (
        ReActPlanner(plan_query)
        if os.getenv("REACT_ENABLED", "true").lower()
        in {"1", "true", "yes", "on"}
        else None
    )

    _mcp_client = MCPClient(
        command=sys.executable,
        args=["-m", "mcp_server.customer_service_server"],
        env={**os.environ, "PYTHONPATH": _ROOT},
    )
    await _mcp_client.connect()
    for adapter in await MCPToolAdapter.discover(
        _mcp_client,
        write_tools={
            "create_refund",
            "create_return",
            "cancel_order",
            "create_ticket",
        },
    ):
        registry.register(adapter)

    router = Router()
    domain_runtime = DomainAgentRuntime(router, {
        "knowledge": KnowledgeAgent(registry, query_planner),
        "order": OrderAgent(registry, planner=react_planner),
        "logistics": LogisticsAgent(registry, planner=react_planner),
        "after_sales": AfterSalesAgent(registry, planner=react_planner),
    })

    async def validate_recovered_slot(slot_name: str, value: str) -> bool:
        validation_tools = {
            "order_id": ("order", "query_order", {"order_id": value}),
            "tracking_no": (
                "logistics",
                "track_package",
                {"tracking_no": value},
            ),
        }
        validation = validation_tools.get(slot_name)
        if validation is None:
            return False
        agent_name, tool_name, params = validation
        result = await registry.call(agent_name, tool_name, params)
        payload = result.data if isinstance(result.data, dict) else {}
        return result.success and payload.get("found") is True

    recognizer.set_slot_validator(validate_recovered_slot)
    turn_engine = TurnEngine(state_store)
    _trace_store = TraceStore()
    _customer_runtime = CustomerAgentRuntime(
        recognizer=recognizer,
        tracker=DialogueStateTracker(),
        turn_engine=turn_engine,
        domain_runtime=domain_runtime,
        router=router,
        trace_store=_trace_store,
    )

    # 性能监控（可选启动 Prometheus）
    prom_port = int(os.getenv("PROMETHEUS_PORT", "0")) or None
    _monitor = PerformanceMonitor(
        runtime=_customer_runtime,
        tool_registry=registry,
        interval_s=float(os.getenv("MONITOR_INTERVAL", "10")),
        webhook_url=os.getenv("ALERT_WEBHOOK_URL") or None,
        prometheus_port=prom_port,
        anomaly_sensitivity=float(
            os.getenv("ANOMALY_DETECTION_THRESHOLD", "2.5"),
        ),
    )
    await _monitor.start()

    # Legacy LLM evaluator 仅在显式开关下初始化。
    _orchestrator = None
    _evaluator = None
    if os.getenv("ENABLE_LEGACY_EVAL", "false").lower() in {
        "1", "true", "yes", "on",
    }:
        from agents.agent_orchestrator import AgentOrchestrator
        from evaluation.evaluator import EndToEndEvaluator

        _orchestrator = AgentOrchestrator(
            api_key=cfg["api_key"],
            base_url=cfg.get("base_url"),
            model=cfg["model"],
            skill_manager=_skill_manager,
        )
        _evaluator = EndToEndEvaluator(
            orchestrator=_orchestrator,
            recognizer=recognizer,
            api_key=cfg["api_key"],
            base_url=cfg.get("base_url"),
            model=cfg["model"],
            baseline_path=os.getenv(
                "EVAL_BASELINE_PATH",
                "/app/data/eval/runtime_baseline.json",
            ),
        )

    logger.info("GGBot 已就绪")
    yield


async def _shutdown_components() -> None:
    """Best-effort cleanup for normal shutdown and partial startup failures."""
    await _drain_background_tasks()
    resources = (
        ("monitor", _monitor, "stop"),
        ("mcp_client", _mcp_client, "close"),
        ("memory", _memory, "close"),
    )
    for name, resource, method_name in resources:
        if resource is None:
            continue
        try:
            await getattr(resource, method_name)()
        except Exception as ex:
            logger.warning("关闭 %s 失败: %s", name, ex)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with _runtime_components(app):
            yield
    finally:
        await _shutdown_components()
        logger.info("GGBot 已关闭")


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="GGBot 智能客服",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:     str
    user_id:     str = "anonymous"
    conv_id:     Optional[str] = None


class ChatResponse(BaseModel):
    conv_id:     str
    response:    str
    intent:      str
    agent_type:  str
    escalated:   bool
    latency_ms:  float
    knowledge_used: bool = False
    trace_id: str = ""
    status: str = "completed"
    missing_slots: List[str] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)


class MCPToolInfo(BaseModel):
    name: str
    title: Optional[str] = None
    description: str = ""
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Optional[Dict[str, Any]] = None
    annotations: Optional[Dict[str, Any]] = None


class MCPToolListResponse(BaseModel):
    total: int
    tools: List[MCPToolInfo] = Field(default_factory=list)


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    components = {
        "customer_runtime": _customer_runtime is not None,
        "tool_registry": _tool_registry is not None,
        "memory": _memory is not None,
        "mcp_client": _mcp_client is not None,
        "knowledge_runtime": _knowledge_runtime is not None,
    }
    missing = [name for name, ready in components.items() if not ready]
    if missing:
        raise HTTPException(
            503,
            f"服务未就绪: {', '.join(missing)}",
        )
    return {
        "status": "ok",
        "components": components,
        "agents": _customer_runtime.get_stats(),
        "tools": _tool_registry.get_stats(),
    }


@app.get(
    "/mcp/tools",
    response_model=MCPToolListResponse,
    tags=["MCP"],
)
async def list_mcp_tools() -> MCPToolListResponse:
    """返回当前 MCP Server 暴露的全部工具定义。"""
    if _mcp_client is None:
        raise HTTPException(503, "MCP Client 未初始化")
    try:
        discovered = await _mcp_client.list_tools()
    except Exception as ex:
        logger.warning("查询 MCP 工具失败: %s", ex)
        raise HTTPException(503, "MCP Server 不可用") from ex

    tools = [
        MCPToolInfo(
            name=tool.name,
            title=getattr(tool, "title", None),
            description=getattr(tool, "description", None) or "",
            input_schema=getattr(tool, "inputSchema", None) or {},
            output_schema=getattr(tool, "outputSchema", None),
            annotations=(
                tool.annotations.model_dump(
                    mode="json",
                    by_alias=True,
                    exclude_none=True,
                )
                if getattr(tool, "annotations", None) is not None
                else None
            ),
        )
        for tool in discovered
    ]
    return MCPToolListResponse(total=len(tools), tools=tools)


@app.get("/skills", tags=["Skills"])
async def skills_summary():
    """查看当前已加载的 Skills，便于确认热加载结果和排查解析错误。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    return _skill_manager.summary()


@app.post("/skills/reload", tags=["Skills"])
async def reload_skills():
    """运行时重新扫描 Skill 目录，不需要重启服务。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    _skill_manager.reload()
    if _orchestrator is not None:
        _orchestrator.set_skill_manager(_skill_manager)
    return _skill_manager.summary()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    主对话接口。完整流程：
      上下文读取 → 结构化 NLU → DST → TurnEngine → Agent/Tool → 状态与记忆写入
    """
    if _customer_runtime is None or _memory is None:
        raise HTTPException(503, "服务未就绪")

    from memory.conversation_memory import EpisodicEventType, MsgRole

    conv_id = req.conv_id or str(uuid.uuid4())
    mem_ctx = await _memory.get_context(req.user_id, conv_id, query=req.message)
    history = [
        {"role": m.role.value, "content": m.content}
        for m in mem_ctx.recent_messages[-5:]
    ] if mem_ctx.recent_messages else None
    skill_prompt = (
        _skill_manager.prompt_for(req.message)
        if _skill_manager is not None
        else ""
    )
    to_prompt_text = getattr(mem_ctx, "to_prompt_text", None)
    agent_context = (
        to_prompt_text(skill_prompt=skill_prompt)
        if to_prompt_text is not None
        else ""
    )
    runtime_args = dict(
        user_id=req.user_id,
        conv_id=conv_id,
        message=req.message,
        history=history,
    )
    if agent_context:
        runtime_args["agent_context"] = agent_context
    result = await _customer_runtime.run(**runtime_args)

    await _memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message)
    await _memory.add_message(req.user_id, conv_id, MsgRole.ASSISTANT, result.response)
    update_profile = getattr(_memory, "update_profile", None)
    if update_profile is not None:
        await update_profile(req.user_id, conv_id)
    if result.status == "completed":
        _spawn_background_task(_memory.record_episodic_event(
            req.user_id,
            conv_id,
            EpisodicEventType.TASK_COMPLETED,
            metadata={"intent": result.intent, "trace_id": result.trace_id},
        ), name=f"episodic:{conv_id}:completed")
    elif result.escalated:
        _spawn_background_task(_memory.record_episodic_event(
            req.user_id,
            conv_id,
            EpisodicEventType.HANDOFF,
            metadata={"intent": result.intent, "trace_id": result.trace_id},
        ), name=f"episodic:{conv_id}:handoff")

    return ChatResponse(
        conv_id=conv_id,
        response=result.response,
        intent=result.intent,
        agent_type=result.agent_type,
        escalated=result.escalated,
        latency_ms=round(result.latency_ms, 1),
        knowledge_used=result.knowledge_used,
        trace_id=result.trace_id,
        status=result.status,
        missing_slots=result.missing_slots,
        citations=result.citations,
    )


@app.get("/traces/{trace_id}", tags=["Agent Trace"])
async def get_trace(trace_id: str):
    if _trace_store is None:
        raise HTTPException(503, "Trace Store 未初始化")
    events = _trace_store.get(trace_id)
    if not events:
        raise HTTPException(404, "Trace 不存在")
    return {"trace_id": trace_id, "events": events}


@app.get("/monitor")
async def monitor_summary():
    """实时监控摘要：Agent 成功率、工具统计、告警、优化建议。"""
    if _monitor is None:
        raise HTTPException(503, "服务未就绪")
    return _monitor.summary()


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus 指标入口。"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/search")
async def search(
    query: str,
    top_k: int = 5,
    as_of: Optional[str] = None,
):
    """使用主 HybridRetriever 执行检索、重排和引用生成。"""
    if _knowledge_runtime is None:
        raise HTTPException(503, "服务未就绪")
    from rag.versioning import RetrievalFilter

    try:
        filters = RetrievalFilter.current(as_of=as_of)
    except ValueError as ex:
        raise HTTPException(400, str(ex)) from ex
    result = await asyncio.to_thread(
        _knowledge_runtime.retriever.search,
        query,
        top_k=top_k,
        use_sparse=True,
        use_reranker=True,
        filters=filters,
    )
    payload = result.model_dump(mode="json")
    return {
        "query": query,
        "results": payload["hits"],
        "citations": payload["citations"],
        "answered": payload["answered"],
        "reason": payload["reason"],
        "reranked": True,
    }


class DocInput(BaseModel):
    """单篇文档输入。"""
    title:   str
    content: str
    knowledge_id: Optional[str] = None
    version: str = "v1"
    version_seq: Optional[int] = None
    status: str = "published"
    effective_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchDocInput(BaseModel):
    """批量文档导入请求体。"""
    documents: List[DocInput]


class PublishVersionInput(BaseModel):
    """可选覆盖知识版本的生效和失效时间。"""

    effective_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None


class EvalIntentInput(BaseModel):
    """意图识别评测用例。"""
    message: str
    expected_intent: str
    context: Optional[Dict[str, Any]] = None


class EvalDialogInput(BaseModel):
    """对话质量评测用例。question 单轮，turns 多轮。"""
    question: Optional[str] = None
    turns: Optional[List[str]] = None
    user_id: Optional[str] = None
    conv_id: Optional[str] = None


class EvalRunInput(BaseModel):
    """评测请求。为空时使用内置默认用例。"""
    mode: str = "customer_agent"
    intent_cases: Optional[List[EvalIntentInput]] = None
    dialog_cases: Optional[List[EvalDialogInput]] = None


@app.post("/knowledge/add", tags=["知识库"])
async def add_knowledge(body: BatchDocInput):
    """
    批量导入文档到知识库。

    文档会自动切片（每片 500 字）并存入 ChromaDB，ChromaDB 内置 Embedding 模型自动向量化。

    示例请求体：
    ```json
    {
      "documents": [
        {"title": "退款政策", "content": "用户在购买后 7 天内可以申请无理由退款..."},
        {"title": "配送说明", "content": "标准配送 3-5 个工作日..."}
      ]
    }
    ```
    """
    if _knowledge_runtime is None:
        raise HTTPException(503, "知识库未初始化")
    kb = _knowledge_runtime.knowledge_base
    documents = []
    for document in body.documents:
        metadata = {
            **document.metadata,
            "version": document.version,
            "status": document.status,
        }
        optional_metadata = {
            "knowledge_id": document.knowledge_id,
            "version_seq": document.version_seq,
            "effective_at": document.effective_at,
            "expires_at": document.expires_at,
        }
        metadata.update({
            key: value
            for key, value in optional_metadata.items()
            if value is not None
        })
        documents.append({
            "title": document.title,
            "content": document.content,
            "source": document.source or document.title,
            "metadata": metadata,
        })
    try:
        count = _knowledge_runtime.add_documents(documents)
    except ValueError as ex:
        raise HTTPException(400, str(ex)) from ex
    return {"message": f"成功导入 {count} 个文档片段", "added_chunks": count, "total_chunks": kb.doc_count}


@app.post("/knowledge/upload", tags=["知识库"])
async def upload_knowledge(file: UploadFile = File(...)):
    """
    上传文件导入知识库。

    支持格式：
    - `.txt` / `.md` / `.pdf`：使用结构感知 Loader 导入
    - `.json`：JSON 数组格式 `[{"title": "...", "content": "..."}, ...]`

    文件大小限制：10MB
    """
    if _knowledge_runtime is None:
        raise HTTPException(503, "知识库未初始化")
    kb = _knowledge_runtime.knowledge_base

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "文件大小超过 10MB 限制")

    filename = file.filename or "unknown"

    if filename.endswith(".json"):
        import json as _json
        text = content.decode("utf-8", errors="ignore")
        try:
            docs = _json.loads(text)
            if not isinstance(docs, list):
                raise HTTPException(400, "JSON 文件应为数组格式: [{title, content}, ...]")
        except _json.JSONDecodeError as e:
            raise HTTPException(400, f"JSON 解析失败: {e}")
        count = _knowledge_runtime.add_documents(docs)
    else:
        try:
            count = _knowledge_runtime.add_file(filename, content)
        except ValueError as ex:
            raise HTTPException(400, str(ex)) from ex
    return {
        "message": f"文件 {filename} 导入成功",
        "added_chunks": count,
        "total_chunks": kb.doc_count,
    }


@app.get("/knowledge/stats", tags=["知识库"])
async def knowledge_stats():
    """查看知识库统计信息（文档片段总数）。"""
    if _knowledge_runtime is None:
        raise HTTPException(503, "知识库未初始化")
    kb = _knowledge_runtime.knowledge_base
    return {"total_chunks": kb.doc_count}


@app.get(
    "/knowledge/{knowledge_id}/versions",
    tags=["知识库版本"],
)
async def list_knowledge_versions(knowledge_id: str):
    if _knowledge_runtime is None:
        raise HTTPException(503, "知识库未初始化")
    return {
        "knowledge_id": knowledge_id,
        "versions": _knowledge_runtime.knowledge_base.list_versions(
            knowledge_id,
        ),
    }


@app.post(
    "/knowledge/{knowledge_id}/versions/{version}/publish",
    tags=["知识库版本"],
)
async def publish_knowledge_version(
    knowledge_id: str,
    version: str,
    body: Optional[PublishVersionInput] = None,
):
    if _knowledge_runtime is None:
        raise HTTPException(503, "知识库未初始化")
    payload = body or PublishVersionInput()
    try:
        result = _knowledge_runtime.knowledge_base.publish_version(
            knowledge_id,
            version,
            effective_at=payload.effective_at,
            expires_at=payload.expires_at,
        )
        _knowledge_runtime.refresh()
        return result
    except KeyError as ex:
        raise HTTPException(404, str(ex)) from ex
    except ValueError as ex:
        raise HTTPException(400, str(ex)) from ex


@app.post(
    "/knowledge/{knowledge_id}/versions/{version}/revoke",
    tags=["知识库版本"],
)
async def revoke_knowledge_version(
    knowledge_id: str,
    version: str,
):
    if _knowledge_runtime is None:
        raise HTTPException(503, "知识库未初始化")
    try:
        result = _knowledge_runtime.knowledge_base.revoke_version(
            knowledge_id,
            version,
        )
        _knowledge_runtime.refresh()
        return result
    except KeyError as ex:
        raise HTTPException(404, str(ex)) from ex


@app.post("/eval/run")
async def run_eval(body: Optional[EvalRunInput] = None):
    """运行内置评测用例，返回评测报告。"""
    if body is None or body.mode == "customer_agent":
        from evaluation.local_eval_runner import run_local_eval

        report = await run_local_eval()
        return {
            "mode": "customer_agent",
            "generated_at": report.generated_at,
            "reproduce_command": report.reproduce_command,
            "sample_size": report.sample_size,
            "summary": report.summary,
        }

    if body.mode != "legacy":
        raise HTTPException(400, f"不支持的评测模式: {body.mode}")

    if _evaluator is None:
        raise HTTPException(503, "服务未就绪")
    from evaluation.evaluator import DEFAULT_DIALOG_CASES, DEFAULT_INTENT_CASES, IntentTestCase

    if body and body.intent_cases is not None:
        intent_cases = [
            IntentTestCase(
                message=c.message,
                expected_intent=c.expected_intent,
                context=c.context,
            )
            for c in body.intent_cases
        ]
    else:
        intent_cases = DEFAULT_INTENT_CASES

    if body and body.dialog_cases is not None:
        dialog_cases = [
            c.model_dump(exclude_none=True)
            for c in body.dialog_cases
        ]
    else:
        dialog_cases = DEFAULT_DIALOG_CASES

    report = await _evaluator.run(
        intent_cases=intent_cases,
        dialog_cases=dialog_cases,
    )
    return {
        "pass_rate":       report.pass_rate,
        "total":           report.total,
        "passed":          report.passed,
        "avg_scores":      report.avg_scores,
        "regressions":     report.regressions,
        "recommendations": report.recommendations,
        "results": [
            {
                "test_id": r.test_id,
                "passed": r.passed,
                "scores": r.scores,
                "detail": r.detail,
                "metadata": r.metadata,
            }
            for r in report.results
        ],
    }


# ── 交互式 CLI ────────────────────────────────────────────────────────────────
# ANSI 颜色：用户=青色，客服=绿色，系统=黄色
_CLI_CYAN = "\033[1;36m"      # 用户侧（加粗青）
_CLI_GREEN = "\033[1;32m"     # 客服侧（加粗绿）
_CLI_YELLOW = "\033[0;33m"    # 系统提示
_CLI_DIM = "\033[2;37m"       # 暗灰（分隔线）
_CLI_RESET = "\033[0m"


def _redirect_logs_to_file():
    """CLI 模式下把日志重定向到文件，避免污染终端对话界面。

    日志按会话时间戳保存到 logs/cli-<timestamp>.log，终端只保留对话内容。
    """
    log_dir = pathlib.Path(_ROOT) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"cli-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"

    root = logging.getLogger()
    # 移除 basicConfig 默认的 stderr handler
    for h in list(root.handlers):
        root.removeHandler(h)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(root.level or logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root.addHandler(file_handler)
    return log_file


async def _cli():
    log_file = _redirect_logs_to_file()
    print(BANNER)
    print("GGBot CLI — 输入 quit 退出")
    print(f"日志已重定向到: {log_file}\n")

    user_id, conv_id = "cli_user", str(uuid.uuid4())
    async with lifespan(app):
        while True:
            try:
                msg = input(f"{_CLI_CYAN}你{_CLI_RESET}: ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{_CLI_YELLOW}再见 ʕ•ᴥ•ʔ{_CLI_RESET}")
                break
            if not msg or msg.lower() in ("quit", "exit", "退出"):
                print(f"{_CLI_YELLOW}再见 ʕ•ᴥ•ʔ{_CLI_RESET}")
                break

            result = await chat(ChatRequest(
                message=msg,
                user_id=user_id,
                conv_id=conv_id,
            ))
            print(
                f"\n{_CLI_GREEN}GGBot{_CLI_RESET} "
                f"{_CLI_DIM}[{result.agent_type}]{_CLI_RESET}: "
                f"{result.response}\n"
            )


if __name__ == "__main__":
    if "--cli" in sys.argv:
        asyncio.run(_cli())
    else:
        uvicorn.run(
            "api.main:app",
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            reload=os.getenv("APP_ENV") == "development",
        )
