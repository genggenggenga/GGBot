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

from core.conversation_lock import ConversationLockTimeout

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
_memory       = None
_monitor      = None
_skill_manager = None
_customer_runtime = None
_trace_store = None
_knowledge_runtime = None
_tool_registry = None
_conversation_locks = None

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
    global _memory, _monitor, _skill_manager
    global _customer_runtime, _trace_store, _knowledge_runtime
    global _tool_registry, _conversation_locks

    # CLI 模式下由 _cli() 负责打印横幅与欢迎语，避免重复输出。
    if "--cli" not in sys.argv:
        print(BANNER, flush=True)

    from agents.domain_agents import (
        AfterSalesAgent,
        DomainAgentRuntime,
        FallbackAgent,
        KnowledgeAgent,
        LogisticsAgent,
        OrderAgent,
        Router,
    )
    from core.customer_agent_runtime import CustomerAgentRuntime
    from core.conversation_lock import RedisConversationLockManager
    from core.dialogue_state_tracker import DialogueStateTracker
    from core.idempotency import RedisActionExecutionRepository
    from core.internal_rpc import build_mock_rpc_clients
    from core.intent_recognizer import IntentRecognizer
    from core.react_planner import ReActPlanner
    from core.response_polisher import ResponsePolisher
    from core.rpc_tools import register_internal_rpc_tools
    from core.state_store import RedisStateStore
    from core.structured_llm import StructuredLLMClient
    from core.tool_names import canonical_tool_name
    from core.tool_registry import ToolRegistry
    from core.trace_store import TraceStore
    from core.turn_engine import TurnEngine
    from mcp.knowledge_base import KnowledgeBase
    from memory.conversation_memory import MemoryManager
    from monitor.performance_monitor import PerformanceMonitor
    from core.skill_loader import SkillManager
    from rag.answer_generator import RAGAnswerGenerator
    from rag.runtime import KnowledgeRuntime, local_models_enabled
    from rag.query_planner import QueryPlanner
    from rag.tool import register_rag_tool
    from core.llm_utils import extract_text_content
    from core.prompts.types import PromptSpec
    import redis.asyncio as redis

    cfg = _anthropic_cfg()
    logger.info(f"模型: {cfg['model']}  base_url: {cfg.get('base_url', '(官方)')}")

    # 意图识别器（Orchestrator 内部也会创建，这里单独暴露给 Evaluator）
    recognizer = IntentRecognizer(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )
    structured_client = StructuredLLMClient(
        recognizer.client,
        cfg["model"],
    )
    recognizer.set_structured_client(structured_client)

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
    _conversation_locks = RedisConversationLockManager(
        redis_client,
        lease_s=float(os.getenv("CONVERSATION_LOCK_LEASE_S", "60")),
        wait_timeout_s=float(
            os.getenv("CONVERSATION_LOCK_WAIT_TIMEOUT_S", "30"),
        ),
        retry_interval_s=float(
            os.getenv("CONVERSATION_LOCK_RETRY_INTERVAL_S", "0.05"),
        ),
    )
    state_store = RedisStateStore(redis_client)
    _memory = await asyncio.to_thread(
        MemoryManager,
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        state_store=state_store,
        structured_client=structured_client,
        redis_client=redis_client,
    )

    # RAG 知识库（基于 ChromaDB 的真实检索）
    kb = await asyncio.to_thread(
        KnowledgeBase,
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
    )
    doc_count = await asyncio.to_thread(lambda: kb.doc_count)
    logger.info("知识库已加载: %s 个文档片段", doc_count)

    # 新运行时统一通过 ToolRegistry 调用 Hybrid RAG 与标准 MCP 工具。
    registry = ToolRegistry()
    _tool_registry = registry
    legacy_threshold = os.getenv("RAG_RELEVANCE_THRESHOLD")
    _knowledge_runtime = await asyncio.to_thread(
        KnowledgeRuntime.build,
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
    async def plan_query(prompt: PromptSpec) -> str:
        response = await recognizer.client.messages.create(
            model=cfg["model"],
            max_tokens=512,
            temperature=0.1,
            system=prompt.system,
            messages=[{"role": "user", "content": prompt.user}],
        )
        return extract_text_content(response.content)

    query_planner = QueryPlanner(
        plan_query,
        structured_client=structured_client,
        enabled=os.getenv(
            "RAG_MULTI_QUERY_ENABLED",
            "true",
        ).lower() in {"1", "true", "yes", "on"},
        max_queries=int(os.getenv("RAG_MULTI_QUERY_MAX_QUERIES", "3")),
        min_confidence=float(
            os.getenv("RAG_QUERY_REWRITE_MIN_CONFIDENCE", "0.5"),
        ),
    )
    answer_generator = (
        RAGAnswerGenerator(
            plan_query,
            structured_client=structured_client,
            max_chunks=int(os.getenv("RAG_ANSWER_MAX_CHUNKS", "5")),
            max_context_tokens=int(
                os.getenv("RAG_ANSWER_MAX_CONTEXT_TOKENS", "1800"),
            ),
            timeout_s=float(os.getenv("RAG_ANSWER_TIMEOUT_S", "8")),
        )
        if os.getenv("RAG_GENERATION_ENABLED", "true").lower()
        in {"1", "true", "yes", "on"}
        else None
    )
    react_planner = (
        ReActPlanner(
            plan_query,
            structured_client=structured_client,
        )
        if os.getenv("REACT_ENABLED", "true").lower()
        in {"1", "true", "yes", "on"}
        else None
    )
    response_polisher = ResponsePolisher(
        plan_query,
        structured_client=structured_client,
        enabled=os.getenv(
            "RESPONSE_POLISH_ENABLED",
            "true",
        ).lower() in {"1", "true", "yes", "on"},
        min_chars=int(os.getenv("RESPONSE_POLISH_MIN_CHARS", "120")),
        timeout_s=float(os.getenv("RESPONSE_POLISH_TIMEOUT_S", "3")),
    )

    action_repository = RedisActionExecutionRepository(
        redis_client,
        ttl_s=int(os.getenv("ACTION_IDEMPOTENCY_TTL_S", "86400")),
    )
    register_internal_rpc_tools(
        registry,
        build_mock_rpc_clients(),
        action_repository,
    )
    register_rag_tool(
        registry,
        _knowledge_runtime.retriever,
        tool_name=canonical_tool_name("rag_search"),
    )

    router = Router()
    domain_runtime = DomainAgentRuntime(router, {
        "fallback": FallbackAgent(),
        "knowledge": KnowledgeAgent(
            registry,
            query_planner,
            answer_generator,
        ),
        "order": OrderAgent(registry, planner=react_planner),
        "logistics": LogisticsAgent(registry, planner=react_planner),
        "after_sales": AfterSalesAgent(
            registry,
            planner=react_planner,
            skill_manager=_skill_manager,
        ),
    })

    async def validate_recovered_slot(slot_name: str, value: str) -> bool:
        validation_tools = {
            "order_id": (
                "order",
                canonical_tool_name("query_order"),
                {"order_id": value},
            ),
            "tracking_no": (
                "logistics",
                canonical_tool_name("track_package"),
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
        response_polisher=response_polisher,
    )

    # 性能监控（可选启动 Prometheus）
    # CLI 模式作为第二个进程运行在已有 API 容器内，端口已被占用，
    # 且交互对话无需独立指标端点，故跳过 Prometheus。
    prom_port = int(os.getenv("PROMETHEUS_PORT", "0")) or None
    if "--cli" in sys.argv:
        prom_port = None
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

    logger.info("GGBot 已就绪")
    yield


async def _shutdown_components() -> None:
    """Best-effort cleanup for normal shutdown and partial startup failures."""
    resources = (
        ("monitor", _monitor, "stop"),
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
        "knowledge_runtime": _knowledge_runtime is not None,
        "conversation_locks": _conversation_locks is not None,
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
    tags=["Internal Tools"],
    deprecated=True,
)
@app.get(
    "/tools",
    response_model=MCPToolListResponse,
    tags=["Internal Tools"],
)
async def list_mcp_tools() -> MCPToolListResponse:
    """兼容接口：返回内部 RPC ToolRegistry 的全部工具定义。"""
    if _tool_registry is None:
        raise HTTPException(503, "Tool Registry 未初始化")
    tools = [
        MCPToolInfo(
            name=spec.name,
            description=spec.description,
            input_schema=spec.input_schema,
            output_schema=spec.output_schema,
        )
        for spec in _tool_registry.list_tools()
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
    return _skill_manager.summary()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    主对话接口。完整流程：
      上下文读取 → 结构化 NLU → DST → TurnEngine → Agent/Tool → 状态与记忆写入
    """
    if _customer_runtime is None or _memory is None:
        raise HTTPException(503, "服务未就绪")

    conv_id = req.conv_id or str(uuid.uuid4())
    if _conversation_locks is None:
        return await _chat_locked(req, conv_id)
    try:
        async with _conversation_locks.lock(req.user_id, conv_id):
            return await _chat_locked(req, conv_id)
    except ConversationLockTimeout as ex:
        raise HTTPException(
            409,
            "当前会话仍在处理上一条消息，请稍后重试。",
        ) from ex


async def _chat_locked(req: ChatRequest, conv_id: str) -> ChatResponse:
    """Execute one complete chat turn while the conversation lock is held."""
    from memory.conversation_memory import EpisodicEventType, MsgRole

    mem_ctx = await _memory.get_context(req.user_id, conv_id, query=req.message)
    history = [
        {"role": m.role.value, "content": m.content}
        for m in mem_ctx.recent_messages[-5:]
    ] if mem_ctx.recent_messages else None
    to_prompt_text = getattr(mem_ctx, "to_prompt_text", None)
    agent_context = (
        to_prompt_text()
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
        await _memory.record_episodic_event(
            req.user_id,
            conv_id,
            EpisodicEventType.TASK_COMPLETED,
            metadata={"intent": result.intent, "trace_id": result.trace_id},
        )
    elif result.escalated:
        await _memory.record_episodic_event(
            req.user_id,
            conv_id,
            EpisodicEventType.HANDOFF,
            metadata={"intent": result.intent, "trace_id": result.trace_id},
        )

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


class EvalRunInput(BaseModel):
    """Versioned evaluation request for the current CustomerAgentRuntime."""
    mode: str = "customer_agent"
    suite: str = "smoke"
    execution_mode: str = "deterministic"
    judge: bool = False


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
        count = await asyncio.to_thread(
            _knowledge_runtime.add_documents,
            documents,
        )
    except ValueError as ex:
        raise HTTPException(400, str(ex)) from ex
    total = await asyncio.to_thread(lambda: kb.doc_count)
    return {
        "message": f"成功导入 {count} 个文档片段",
        "added_chunks": count,
        "total_chunks": total,
    }


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
        count = await asyncio.to_thread(
            _knowledge_runtime.add_documents,
            docs,
        )
    else:
        try:
            count = await asyncio.to_thread(
                _knowledge_runtime.add_file,
                filename,
                content,
            )
        except ValueError as ex:
            raise HTTPException(400, str(ex)) from ex
    total = await asyncio.to_thread(lambda: kb.doc_count)
    return {
        "message": f"文件 {filename} 导入成功",
        "added_chunks": count,
        "total_chunks": total,
    }


@app.get("/knowledge/stats", tags=["知识库"])
async def knowledge_stats():
    """查看知识库统计信息（文档片段总数）。"""
    if _knowledge_runtime is None:
        raise HTTPException(503, "知识库未初始化")
    kb = _knowledge_runtime.knowledge_base
    return {
        "total_chunks": await asyncio.to_thread(lambda: kb.doc_count),
    }


@app.get(
    "/knowledge/{knowledge_id}/versions",
    tags=["知识库版本"],
)
async def list_knowledge_versions(knowledge_id: str):
    if _knowledge_runtime is None:
        raise HTTPException(503, "知识库未初始化")
    versions = await asyncio.to_thread(
        _knowledge_runtime.knowledge_base.list_versions,
        knowledge_id,
    )
    return {
        "knowledge_id": knowledge_id,
        "versions": versions,
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
        result = await asyncio.to_thread(
            _knowledge_runtime.knowledge_base.publish_version,
            knowledge_id,
            version,
            effective_at=payload.effective_at,
            expires_at=payload.expires_at,
        )
        await asyncio.to_thread(_knowledge_runtime.refresh)
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
        result = await asyncio.to_thread(
            _knowledge_runtime.knowledge_base.revoke_version,
            knowledge_id,
            version,
        )
        await asyncio.to_thread(_knowledge_runtime.refresh)
        return result
    except KeyError as ex:
        raise HTTPException(404, str(ex)) from ex


@app.post("/eval/run")
async def run_eval(body: Optional[EvalRunInput] = None):
    """运行内置评测用例，返回评测报告。"""
    body = body or EvalRunInput()
    if body.mode not in {"customer_agent", "deterministic"}:
        raise HTTPException(400, f"不支持的评测模式: {body.mode}")
    if body.suite not in {"smoke", "golden", "bad_cases"}:
        raise HTTPException(400, f"不支持的评测集: {body.suite}")
    if body.execution_mode != "deterministic":
        raise HTTPException(
            400,
            "API 仅运行 deterministic 评测；realistic 模式需要受控离线环境。",
        )
    from evaluation.local_eval_runner import run_local_eval

    judge = None
    if body.judge:
        from evaluation.judge import LLMJudge

        try:
            cfg = _anthropic_cfg()
        except RuntimeError as ex:
            raise HTTPException(400, str(ex)) from ex
        judge = LLMJudge.from_api_key(
            cfg["api_key"],
            cfg["model"],
            cfg.get("base_url"),
        )
    if (
        body.suite == "smoke"
        and body.execution_mode == "deterministic"
        and judge is None
    ):
        report = await run_local_eval()
    else:
        report = await run_local_eval(
            suite=body.suite,
            execution_mode=body.execution_mode,
            judge=judge,
        )
    return {
        "mode": "customer_agent",
        "suite": getattr(report, "suite", body.suite),
        "execution_mode": getattr(
            report,
            "execution_mode",
            body.execution_mode,
        ),
        "dataset": getattr(report, "dataset", {}),
        "generated_at": report.generated_at,
        "reproduce_command": report.reproduce_command,
        "sample_size": report.sample_size,
        "summary": report.summary,
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
