"""Deterministic local evaluation runner for GGBot.

Executes the 50 cases in data/eval/customer_agent_cases.json using real
project components: NLU fast-track, DialogueStateTracker, Router,
domain Agents, ToolRegistry with MCP mock tools, and HybridRetriever
with injectable fake embedding/reranker.

Does NOT call remote LLMs or download models. All LLM-dependent paths
use the deterministic fast-track or structured fallback.

Outputs:
  - JSON report with generated_at, reproduce_command, sample_size,
    summary metrics, per-mode ablation, and per-case results.
  - Markdown report with the same data in human-readable form.
"""
from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evaluation.datasets import dataset_summary, load_dataset
from core.agent_models import (
    ConfirmationStatus,
    DialogueState,
    UnderstandingResult,
)
from core.dialogue_state_tracker import DialogueStateTracker
from core.nlu_fast_track import fast_track_extract, build_understanding_from_fast_track
from core.nlu_llm import make_fallback_understanding
from core.state_store import InMemoryStateStore
from core.tool_registry import LocalToolAdapter, ToolRegistry, ToolSpec, ToolType
from core.trace_store import TraceStore
from rag.models import DocumentChunk

logger = logging.getLogger(__name__)

_EVAL_DIR = pathlib.Path(__file__).parent.parent / "data" / "eval"
_REPORTS_DIR = _EVAL_DIR / "reports" / datetime.now().date().isoformat()
_CASES_FILE = _EVAL_DIR / "customer_agent_cases.json"
_CORPUS_FILE = _EVAL_DIR / "knowledge" / "corpus-v1.json"
_EVAL_DATE = date(2026, 8, 7)


# ── Fake embedding / reranker for hybrid retrieval ───────────────────────────

class FakeDenseIndex:
    """Deterministic fake dense index: matches by token overlap."""

    def __init__(self) -> None:
        self._chunks: Dict[str, Any] = {}

    def add(self, chunks: Sequence[Any]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    def search(self, query: str, top_k: int) -> List[Any]:
        from rag.models import SearchHit
        from rag.indexes import tokenize
        query_tokens = set(tokenize(query))
        scored: List[Tuple[float, Any]] = []
        for chunk in self._chunks.values():
            chunk_tokens = set(tokenize(chunk.content))
            overlap = len(query_tokens & chunk_tokens)
            score = overlap / (len(query_tokens) + 1)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            SearchHit(chunk=chunk, score=score, dense_score=score)
            for score, chunk in scored[:top_k]
        ]


class FakeSparseIndex:
    """Deterministic BM25 index from rag.indexes."""

    def __init__(self) -> None:
        from rag.indexes import BM25Index
        self._index = BM25Index()

    def add(self, chunks: Sequence[Any]) -> None:
        self._index.add(chunks)

    def search(self, query: str, top_k: int) -> List[Any]:
        return self._index.search(query, top_k)


class FakeReranker:
    """Deterministic fake reranker: scores by chunk_id hash for reproducibility."""

    def score(self, query: str, chunks: Sequence[Any]) -> List[float]:
        import hashlib
        scores = []
        for chunk in chunks:
            digest = hashlib.md5(f"{query}:{chunk.chunk_id}".encode()).hexdigest()
            scores.append(int(digest[:8], 16) / 0xFFFFFFFF)
        return scores


# ── Mock MCP tool handlers ──────────────────────────────────────────────────

_MOCK_ORDERS: Dict[str, Dict[str, Any]] = {
    "ORD-1001": {"order_id": "ORD-1001", "status": "delivered", "amount": 299.0,
                 "found": True, "refundable_until": "2026-08-08"},
    "ORD-1002": {"order_id": "ORD-1002", "status": "in_transit", "amount": 89.0,
                 "found": True, "refundable_until": None},
    "ORD-EXPIRED": {"order_id": "ORD-EXPIRED", "status": "delivered", "amount": 49.0,
                    "found": True, "refundable_until": "2026-06-10"},
}

_MOCK_LOGISTICS: Dict[str, Dict[str, Any]] = {
    "ORD-1001": {"order_id": "ORD-1001", "status": "delivered",
                 "tracking_no": "SF1001001", "found": True},
    "ORD-1002": {"order_id": "ORD-1002", "status": "in_transit",
                 "tracking_no": "SF1001002", "found": True},
}

_refund_actions: Dict[str, Dict[str, Any]] = {}


async def _query_order(params: Dict, ctx: Optional[Dict]) -> Dict:
    oid = params.get("order_id", "")
    order = _MOCK_ORDERS.get(oid)
    if order is None:
        return {"found": False, "order_id": oid, "error": "order_not_found"}
    return dict(order)


async def _track_package(params: Dict, ctx: Optional[Dict]) -> Dict:
    oid = params.get("order_id", "")
    log = _MOCK_LOGISTICS.get(oid)
    if log is None:
        return {"found": False, "order_id": oid, "error": "tracking_not_available"}
    return dict(log)


async def _check_refund(params: Dict, ctx: Optional[Dict]) -> Dict:
    oid = params.get("order_id", "")
    order = _MOCK_ORDERS.get(oid)
    if order is None:
        return {"eligible": False, "order_id": oid, "reason": "order_not_found"}
    rfu = order.get("refundable_until")
    eligible = (
        order["status"] == "delivered"
        and rfu is not None
        and _EVAL_DATE <= date.fromisoformat(rfu)
    )
    return {"eligible": eligible, "order_id": oid, "reason":
            "within_refund_window" if eligible else "outside_refund_window"}


async def _create_refund(params: Dict, ctx: Optional[Dict]) -> Dict:
    action_id = params.get("action_id", "unknown")
    oid = params.get("order_id", "")
    existing = _refund_actions.get(action_id)
    if existing is not None:
        return {**existing, "idempotent_replay": True}
    eligibility = await _check_refund({"order_id": oid}, None)
    if not eligibility["eligible"]:
        return {"created": False, "order_id": oid, "action_id": action_id,
                "error": eligibility["reason"]}
    refund = {"created": True, "refund_id": f"REF-{len(_refund_actions)+1:04d}",
              "order_id": oid, "action_id": action_id, "status": "submitted"}
    _refund_actions[action_id] = refund
    return dict(refund)


async def _create_ticket(params: Dict, ctx: Optional[Dict]) -> Dict:
    return {"created": True, "ticket_id": "TKT-0001",
            "action_id": params.get("action_id", "unknown"),
            "subject": params.get("subject", "handoff"), "status": "open"}


async def _rag_search(params: Dict, ctx: Optional[Dict]) -> Dict:
    q = params.get("query", "")
    return {"query": q, "hits": [], "citations": [], "answered": False,
            "reason": "no_indexed_knowledge_in_local_eval"}


def _build_tool_registry(
    *,
    call_log: Optional[List[Dict[str, Any]]] = None,
    fail_tools: Optional[set[str]] = None,
    retriever: Optional[Any] = None,
    rag_mode: str = "hybrid",
) -> ToolRegistry:
    """Build deterministic tools while recording the calls made by Agents."""
    registry = ToolRegistry()
    call_log = call_log if call_log is not None else []
    fail_tools = fail_tools or set()

    async def rag_handler(params: Dict, ctx: Optional[Dict]) -> Dict:
        if retriever is None:
            return await _rag_search(params, ctx)
        result = retriever.search(
            params.get("query", ""),
            top_k=params.get("top_k", 5),
            candidate_k=params.get("candidate_k", 10),
            use_sparse=rag_mode != "dense",
            use_reranker=rag_mode == "rerank",
        )
        return result.model_dump(mode="json")

    for name, handler, tool_type, required in [
        ("query_order", _query_order, ToolType.READ, ["order_id"]),
        ("track_package", _track_package, ToolType.READ, ["order_id"]),
        ("check_refund_eligibility", _check_refund, ToolType.READ, ["order_id"]),
        ("create_refund", _create_refund, ToolType.WRITE, ["order_id", "action_id"]),
        ("create_ticket", _create_ticket, ToolType.WRITE, ["subject", "action_id"]),
        ("rag_search", rag_handler, ToolType.READ, ["query"]),
    ]:
        def wrap(tool_name, tool_handler):
            async def recorded(params, context):
                entry = {
                    "tool": tool_name,
                    "params": dict(params),
                }
                call_log.append(entry)
                if tool_name in fail_tools:
                    entry["error"] = f"forced failure: {tool_name}"
                    raise RuntimeError(f"forced failure: {tool_name}")
                try:
                    data = await tool_handler(params, context)
                    entry["data"] = data
                    return data
                except Exception as ex:
                    entry["error"] = str(ex)
                    raise

            return recorded

        spec = ToolSpec(
            name=name, description=f"Mock {name}",
            input_schema={"type": "object", "properties": {}, "required": required},
            tool_type=tool_type, timeout_s=5.0,
        )
        registry.register(LocalToolAdapter(spec, wrap(name, handler)))
    registry.set_agent_whitelist("order", {"query_order", "query_payment"})
    registry.set_agent_whitelist("logistics", {"query_order", "track_package", "rag_search"})
    registry.set_agent_whitelist("after_sales", {
        "query_order", "check_refund_eligibility", "create_refund",
        "create_ticket", "rag_search",
    })
    registry.set_agent_whitelist("knowledge", {"rag_search"})
    return registry


# ── Structured NLU (fast-track only, no LLM) ───────────────────────────────


def _nlu_fast(message: str, current_state: Optional[Dict] = None) -> UnderstandingResult:
    """Determine intent and slots using only the deterministic fast-track."""
    ft = fast_track_extract(
        message,
        confirmation_pending=(
            (current_state or {}).get("confirmation_status") == "pending"
        ),
    )
    result = build_understanding_from_fast_track(ft, message)
    if result is not None and result.confidence >= 0.9:
        active_intent = (current_state or {}).get("active_intent")
        if active_intent and (ft.intent is None or result.corrected_slots):
            from core.agent_models import INTENT_SCHEMAS
            result = result.model_copy(update={
                "intents": [active_intent],
                "primary_intent": active_intent,
                "route_to": INTENT_SCHEMAS[active_intent].allowed_agents[0]
                if active_intent in INTENT_SCHEMAS else None,
            })
        return result
    return make_fallback_understanding(message, current_state)


class _DeterministicRecognizer:
    """Structured recognizer used by local runtime evaluation without network."""

    async def recognize_structured(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        current_state: Optional[Dict[str, Any]] = None,
    ) -> UnderstandingResult:
        del history
        return _nlu_fast(message, current_state)


def _build_eval_runtime(
    *,
    retriever: Any,
    rag_mode: str,
    call_log: List[Dict[str, Any]],
    fail_tools: Optional[set[str]] = None,
):
    """Assemble the same runtime classes used by the /chat endpoint."""
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
    from core.turn_engine import TurnEngine

    store = InMemoryStateStore()
    registry = _build_tool_registry(
        call_log=call_log,
        fail_tools=fail_tools,
        retriever=retriever,
        rag_mode=rag_mode,
    )
    router = Router()
    domain_runtime = DomainAgentRuntime(router, {
        "fallback": FallbackAgent(),
        "knowledge": KnowledgeAgent(registry),
        "order": OrderAgent(registry),
        "logistics": LogisticsAgent(registry),
        "after_sales": AfterSalesAgent(registry),
    })
    runtime = CustomerAgentRuntime(
        recognizer=_DeterministicRecognizer(),
        tracker=DialogueStateTracker(),
        turn_engine=TurnEngine(store),
        domain_runtime=domain_runtime,
        router=router,
        trace_store=TraceStore(),
    )
    return runtime, store


# ── Per-case execution ──────────────────────────────────────────────────────

@dataclass
class CaseResult:
    case_id: str
    category: str
    risk_level: str = "normal"
    predicted_intent: Optional[str] = None
    expected_intent: Optional[str] = None
    predicted_tool: Optional[str] = None
    expected_tool: Optional[str] = None
    predicted_slots: Dict[str, Any] = field(default_factory=dict)
    expected_slots: Dict[str, Any] = field(default_factory=dict)
    predicted_act: Optional[str] = None
    expected_act: Optional[str] = None
    predicted_status: Optional[str] = None
    expected_status: Optional[str] = None
    predicted_response: Optional[str] = None
    dialogue_state: Optional[Dict[str, Any]] = None
    observed_tools: List[str] = field(default_factory=list)
    observed_tool_results: List[Dict[str, Any]] = field(default_factory=list)
    expected_tool_trace: Optional[List[str]] = None
    forbidden_tools: List[str] = field(default_factory=list)
    expected_postconditions: Dict[str, Any] = field(default_factory=dict)
    observed_postconditions: Dict[str, Any] = field(default_factory=dict)
    required_evidence_ids: List[str] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    grounded: Optional[bool] = None
    must_abstain: Optional[bool] = None
    abstained: Optional[bool] = None
    judge_scores: Optional[Dict[str, Any]] = None
    completed: bool = False
    error: Optional[str] = None


async def _execute_dst_case(
    case: Dict[str, Any],
    tracker: DialogueStateTracker,
) -> CaseResult:
    """Execute a DST (slot_fill/correction/switch/multi_turn) case."""
    turns = case.get("turns", [])
    state = DialogueState()
    last_intent = None
    last_slots: Dict[str, Any] = {}

    for turn_text in turns:
        understanding = _nlu_fast(turn_text, state.model_dump(mode="json"))
        state = tracker.update(state, understanding)
        last_intent = state.active_intent
        last_slots = dict(state.slots)

    predicted_status = "completed"
    if state.missing_slots:
        predicted_status = "awaiting_user"
    if state.confirmation_status == ConfirmationStatus.PENDING:
        predicted_status = "awaiting_user"

    return CaseResult(
        case_id=case["id"],
        category=case["category"],
        risk_level=case.get("risk_level", "normal"),
        predicted_intent=last_intent,
        expected_intent=case.get("expected_intent"),
        predicted_slots=last_slots,
        expected_slots=dict(case.get("expected_slots", {})),
        dialogue_state=state.model_dump(mode="json"),
        completed=predicted_status == "completed" or predicted_status == "awaiting_user",
    )


async def _execute_tool_case(
    case: Dict[str, Any],
    retriever: Any,
    mode: str,
) -> CaseResult:
    """Execute a tool case through the real customer-agent runtime."""
    calls: List[Dict[str, Any]] = []
    runtime, store = _build_eval_runtime(
        retriever=retriever,
        rag_mode=mode,
        call_log=calls,
        fail_tools=set(case.get("fail_tools", [])),
    )
    turns = case.get("turns") or [case.get("message", "")]
    result = None
    conv_id = f"eval-{case['id']}"
    confirmation_seen = False
    for turn_index, turn in enumerate(turns):
        is_confirmation = (
            _nlu_fast(
                turn,
                {"confirmation_status": "pending"},
            ).user_act.value == "confirm"
        )
        start = len(calls)
        result = await runtime.run("eval-user", conv_id, turn)
        for call in calls[start:]:
            call["turn_index"] = turn_index
            call["confirmation_before_write"] = confirmation_seen or is_confirmation
        confirmation_seen = confirmation_seen or is_confirmation

    state = await store.load("eval-user", conv_id)
    observed_tools = [item["tool"] for item in calls]
    predicted_tool = observed_tools[-1] if observed_tools else None
    expected_tool = case.get("expected_tool")
    expected_params = dict(case.get("expected_params", {}))
    predicted_params: Dict[str, Any] = {}
    matching = next(
        (
            item for item in reversed(calls)
            if item["tool"] == predicted_tool
        ),
        None,
    )
    if matching is not None:
        predicted_params = {
            key: matching["params"].get(key)
            for key in expected_params
        }

    return CaseResult(
        case_id=case["id"],
        category=case["category"],
        risk_level=case.get("risk_level", "normal"),
        predicted_intent=result.intent if result else None,
        predicted_tool=predicted_tool,
        expected_tool=expected_tool,
        predicted_slots=predicted_params,
        expected_slots=expected_params,
        predicted_status=result.status if result else None,
        predicted_response=result.response if result else None,
        dialogue_state=state.model_dump(mode="json") if state else None,
        observed_tools=observed_tools,
        observed_tool_results=calls,
        expected_tool_trace=case.get("expected_tool_trace"),
        forbidden_tools=list(case.get("forbidden_tools", [])),
        expected_postconditions=dict(case.get("expected_postconditions", {})),
        observed_postconditions=_derive_postconditions(calls),
        completed=True,
    )


async def _execute_rag_case(
    case: Dict[str, Any],
    retriever: Any,
    mode: str = "rerank",
) -> CaseResult:
    """Execute a RAG retrieval case using the hybrid retriever."""
    query = case.get("query", "")
    relevant_ids = set(case.get("relevant_ids", []))

    result = retriever.search(
        query, top_k=5, candidate_k=10,
        use_sparse=(mode != "dense"),
        use_reranker=(mode == "rerank"),
    )
    ranked_ids = [hit.chunk.chunk_id for hit in result.hits]
    chunk_types = [
        str(hit.chunk.metadata.get("chunk_type", ""))
        for hit in result.hits
    ]
    parent_context_ids: List[str] = []
    parent_contexts = getattr(retriever, "parent_contexts", None)
    if parent_contexts is not None and result.hits:
        try:
            parent_context_ids = list(parent_contexts(result.hits))
        except Exception:
            parent_context_ids = []
    citations = [
        {
            **citation.model_dump(mode="json"),
            "supported": citation.chunk_id in relevant_ids,
        }
        for citation in result.citations
    ]
    grounded = (
        (not relevant_ids and not result.answered)
        or (
            bool(relevant_ids)
            and result.answered
            and any(chunk_id in relevant_ids for chunk_id in ranked_ids)
        )
    )

    return CaseResult(
        case_id=case["id"],
        category=case["category"],
        risk_level=case.get("risk_level", "normal"),
        predicted_slots={
            "ranked_ids": ranked_ids,
            "relevant_ids": list(relevant_ids),
            "chunk_types": chunk_types,
            "parent_context_ids": parent_context_ids,
            "rerank_fallback": bool(result.metadata.get("rerank_fallback")),
            "metadata_boosted": int(result.metadata.get("metadata_boosted", 0)),
        },
        expected_slots={
            "expected_chunk_types": list(case.get("expected_chunk_types", [])),
            "expected_parent_ids": list(case.get("expected_parent_ids", [])),
        },
        required_evidence_ids=list(case.get("required_evidence_ids", relevant_ids)),
        citations=citations,
        grounded=grounded,
        must_abstain=case.get("must_abstain"),
        abstained=not result.answered,
        completed=True,
    )


async def _execute_e2e_case(
    case: Dict[str, Any],
    retriever: Any,
    mode: str,
) -> CaseResult:
    """Execute all turns through CustomerAgentRuntime and TurnEngine."""
    calls: List[Dict[str, Any]] = []
    runtime, store = _build_eval_runtime(
        retriever=retriever,
        rag_mode=mode,
        call_log=calls,
        fail_tools=set(case.get("fail_tools", [])),
    )
    conv_id = f"eval-{case['id']}"
    result = None
    confirmation_seen = False
    for turn_index, turn_text in enumerate(case.get("turns", [])):
        is_confirmation = (
            _nlu_fast(
                turn_text,
                {"confirmation_status": "pending"},
            ).user_act.value == "confirm"
        )
        start = len(calls)
        result = await runtime.run("eval-user", conv_id, turn_text)
        for call in calls[start:]:
            call["turn_index"] = turn_index
            call["confirmation_before_write"] = confirmation_seen or is_confirmation
        confirmation_seen = confirmation_seen or is_confirmation

    expected_status = case.get("expected_status", "completed")
    predicted_status = result.status if result else None
    completed = predicted_status == expected_status
    state = await store.load("eval-user", conv_id)

    return CaseResult(
        case_id=case["id"],
        category=case["category"],
        risk_level=case.get("risk_level", "normal"),
        predicted_status=predicted_status,
        expected_status=expected_status,
        predicted_response=result.response if result else None,
        predicted_intent=result.intent if result else None,
        expected_intent=case.get("expected_intent"),
        predicted_slots=dict(state.slots) if state else {},
        expected_slots=dict(case.get("expected_slots", {})),
        dialogue_state=state.model_dump(mode="json") if state else None,
        observed_tools=[item["tool"] for item in calls],
        observed_tool_results=calls,
        expected_tool_trace=case.get("expected_tool_trace"),
        forbidden_tools=list(case.get("forbidden_tools", [])),
        expected_postconditions=dict(case.get("expected_postconditions", {})),
        observed_postconditions=_derive_postconditions(calls),
        citations=list(result.citations) if result else [],
        completed=completed,
    )


async def _execute_nlu_case(
    case: Dict[str, Any],
) -> CaseResult:
    """Execute an NLU intent/act/slot case."""
    message = case.get("message", "")
    understanding = _nlu_fast(message, case.get("current_state"))

    return CaseResult(
        case_id=case["id"],
        category=case["category"],
        risk_level=case.get("risk_level", "normal"),
        predicted_intent=understanding.primary_intent,
        expected_intent=case.get("expected_intent"),
        predicted_act=understanding.user_act.value,
        expected_act=case.get("expected_act"),
        predicted_slots=understanding.extracted_slots,
        expected_slots=dict(case.get("expected_slots", {})),
        completed=True,
    )


def _derive_postconditions(
    calls: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Derive stable business assertions from recorded deterministic tools."""
    values: Dict[str, Any] = {
        "confirmation_observed": any(
            call["tool"] in {"create_refund", "create_ticket"}
            and call.get("confirmation_before_write") is True
            for call in calls
        ),
        "refund_created": False,
        "ticket_created": False,
    }
    for call in calls:
        data = call.get("data") or {}
        if call["tool"] == "create_refund" and data.get("created") is True:
            values["refund_created"] = True
        if call["tool"] == "create_ticket" and data.get("created") is True:
            values["ticket_created"] = True
    return values


# ── Metrics computation ─────────────────────────────────────────────────────

from evaluation.agent_metrics import (
    abstention_metrics,
    citation_precision,
    citation_recall,
    faithfulness_rate,
    forbidden_tool_rate,
    joint_goal_accuracy,
    mean_reciprocal_rank,
    postcondition_success_rate,
    recall_at_k,
    slot_f1,
    task_completion_rate,
    tool_call_accuracy,
    tool_trace_exact_match,
)


def _compute_intent_metrics(results: List[CaseResult]) -> Dict[str, Any]:
    """Intent accuracy and Macro-F1 from NLU/DST cases."""
    pairs = [(r.expected_intent, r.predicted_intent)
             for r in results if r.expected_intent is not None]
    if not pairs:
        return {"accuracy": 0.0, "macro_f1": 0.0, "total": 0, "correct": 0}

    correct = sum(exp == pred for exp, pred in pairs)
    accuracy = correct / len(pairs)

    labels = sorted(set(exp for exp, _ in pairs) | set(pred for _, pred in pairs))
    per_class: Dict[str, Dict[str, float]] = {}
    for label in labels:
        tp = sum(pred == label and exp == label for exp, pred in pairs)
        fp = sum(pred == label and exp != label for exp, pred in pairs)
        fn = sum(pred != label and exp == label for exp, pred in pairs)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[label] = {"precision": prec, "recall": rec, "f1": f1}

    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class) if per_class else 0.0
    return {"accuracy": round(accuracy, 4), "macro_f1": round(macro_f1, 4),
            "total": len(pairs), "correct": correct, "per_class": per_class}


def _chunk_type_recall(results: List[CaseResult]) -> float:
    values = []
    for result in results:
        expected = set(result.expected_slots.get("expected_chunk_types", []))
        if not expected:
            continue
        predicted = set(result.predicted_slots.get("chunk_types", []))
        values.append(len(expected & predicted) / len(expected))
    return sum(values) / len(values) if values else 0.0


def _guardrail_recall(results: List[CaseResult]) -> float:
    values = []
    for result in results:
        expected = set(result.expected_slots.get("expected_chunk_types", []))
        if "guardrail" not in expected:
            continue
        predicted = set(result.predicted_slots.get("chunk_types", []))
        values.append(1.0 if "guardrail" in predicted else 0.0)
    return sum(values) / len(values) if values else 0.0


def _parent_context_recall(results: List[CaseResult]) -> float:
    values = []
    for result in results:
        expected = set(result.expected_slots.get("expected_parent_ids", []))
        if not expected:
            continue
        predicted = set(result.predicted_slots.get("parent_context_ids", []))
        values.append(len(expected & predicted) / len(expected))
    return sum(values) / len(values) if values else 0.0


def _rate(values: Sequence[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _compute_all_metrics(
    all_results: Dict[str, List[CaseResult]],
) -> Dict[str, Any]:
    """Compute the full metric suite from per-category results."""
    dst_cases = all_results.get("dst", [])
    tool_cases = all_results.get("tool", [])
    rag_cases = all_results.get("rag", [])
    e2e_cases = all_results.get("e2e", [])
    nlu_cases = all_results.get("nlu", [])

    intent_results = [r for r in nlu_cases + dst_cases if r.expected_intent is not None]
    intent_metrics = _compute_intent_metrics(intent_results)
    act_pairs = [
        (result.expected_act, result.predicted_act)
        for result in nlu_cases
        if result.expected_act is not None
    ]
    act_accuracy = (
        sum(expected == predicted for expected, predicted in act_pairs)
        / len(act_pairs)
        if act_pairs else 0.0
    )

    dst_expected = [r.expected_slots for r in dst_cases if r.expected_slots]
    dst_predicted = [r.predicted_slots for r in dst_cases if r.expected_slots]
    slot_metrics = slot_f1(dst_expected, dst_predicted) if dst_expected else {
        "precision": 0.0, "recall": 0.0, "f1": 0.0}

    # DST JGA: full state match on slots (using expected_slots as ground truth)
    dst_goal_expected = [r.expected_slots for r in dst_cases if r.expected_slots]
    dst_goal_predicted = [r.predicted_slots for r in dst_cases if r.expected_slots]
    dst_jga = joint_goal_accuracy(dst_goal_expected, dst_goal_predicted) if dst_goal_expected else 0.0

    rag_relevant = [set(r.predicted_slots.get("relevant_ids", [])) for r in rag_cases]
    rag_ranked = [r.predicted_slots.get("ranked_ids", []) for r in rag_cases]
    r_at_5 = recall_at_k(rag_relevant, rag_ranked, 5) if rag_cases else 0.0
    mrr_val = mean_reciprocal_rank(rag_relevant, rag_ranked) if rag_cases else 0.0
    chunk_type_rec = _chunk_type_recall(rag_cases)
    guardrail_rec = _guardrail_recall(rag_cases)
    parent_context_rec = _parent_context_recall(rag_cases)
    rerank_fallback_rate = _rate([
        bool(result.predicted_slots.get("rerank_fallback"))
        for result in rag_cases
    ])
    metadata_boost_rate = _rate([
        int(result.predicted_slots.get("metadata_boosted", 0)) > 0
        for result in rag_cases
    ])

    tool_data = [
        {"expected_tool": r.expected_tool, "predicted_tool": r.predicted_tool,
         "expected_params": r.expected_slots, "predicted_params": r.predicted_slots}
        for r in tool_cases if r.expected_tool is not None
    ]
    tool_metrics = tool_call_accuracy(tool_data)

    completion_data = [{"completed": r.completed} for r in e2e_cases]
    tcr = task_completion_rate(completion_data) if e2e_cases else 0.0

    citation_cases = [
        {
            "citations": result.citations,
            "required_evidence_ids": result.required_evidence_ids,
        }
        for result in rag_cases
    ]
    grounded_cases = [
        {"grounded": result.grounded}
        for result in rag_cases
    ]
    citation_prec = citation_precision(citation_cases)
    citation_rec = citation_recall(citation_cases)
    faith = faithfulness_rate(grounded_cases)
    workflow_cases = [
        {
            "expected_tool_trace": result.expected_tool_trace,
            "observed_tools": result.observed_tools,
            "forbidden_tools": result.forbidden_tools,
            "expected_postconditions": result.expected_postconditions,
            "observed_postconditions": result.observed_postconditions,
        }
        for result in tool_cases + e2e_cases
    ]
    trace_cases = [
        case for case in workflow_cases
        if case["expected_tool_trace"] is not None
    ]
    safety_cases = [
        case for case in workflow_cases
        if case["forbidden_tools"]
    ]
    postcondition_cases = [
        case for case in workflow_cases
        if case["expected_postconditions"]
    ]
    abstention_cases = [
        result for result in rag_cases
        if result.must_abstain is not None
    ]
    forbidden_rate = forbidden_tool_rate(workflow_cases)
    abstention = abstention_metrics([
        {
            "must_abstain": result.must_abstain,
            "abstained": result.abstained,
        }
        for result in rag_cases
    ])
    judged = [
        result.judge_scores for result in (
            tool_cases + e2e_cases
        ) if result.judge_scores and not result.judge_scores.get("judge_failed")
    ]
    judge_summary: Dict[str, Any] = {}
    if judged:
        judge_summary = {
            f"judge_{field}": round(
                sum(float(scores[field]) for scores in judged) / len(judged),
                4,
            )
            for field in (
                "relevance",
                "accuracy",
                "completeness",
                "helpfulness",
                "safety",
                "groundedness",
                "overall",
            )
        }
        judge_summary["judge_sample_size"] = len(judged)

    return {
        "intent_accuracy": intent_metrics["accuracy"],
        "intent_macro_f1": intent_metrics["macro_f1"],
        "user_act_accuracy": round(act_accuracy, 4),
        "slot_f1": slot_metrics["f1"],
        "slot_precision": slot_metrics["precision"],
        "slot_recall": slot_metrics["recall"],
        "dst_joint_goal_accuracy": round(dst_jga, 4),
        "recall_at_5": round(r_at_5, 4),
        "mrr": round(mrr_val, 4),
        "chunk_type_recall": round(chunk_type_rec, 4),
        "guardrail_recall": round(guardrail_rec, 4),
        "parent_context_recall": round(parent_context_rec, 4),
        "rerank_fallback_rate": round(rerank_fallback_rate, 4),
        "metadata_boost_rate": round(metadata_boost_rate, 4),
        "tool_selection_accuracy": tool_metrics["selection_accuracy"],
        "tool_parameter_accuracy": tool_metrics["parameter_accuracy"],
        "task_completion_rate": round(tcr, 4),
        "citation_precision": citation_prec,
        "citation_recall": citation_rec,
        "faithfulness_rate": faith,
        "tool_trace_exact_match": tool_trace_exact_match(workflow_cases),
        "tool_trace_sample_size": len(trace_cases),
        "unsafe_action_rate": forbidden_rate,
        "confirmation_safety_rate": round(1.0 - forbidden_rate, 4),
        "confirmation_safety_sample_size": len(safety_cases),
        "postcondition_success_rate": postcondition_success_rate(workflow_cases),
        "postcondition_sample_size": len(postcondition_cases),
        "abstention_precision": abstention["precision"],
        "abstention_recall": abstention["recall"],
        "abstention_sample_size": len(abstention_cases),
        "intent_detail": intent_metrics,
        "slot_detail": slot_metrics,
        **judge_summary,
    }


# ── Main runner ─────────────────────────────────────────────────────────────

@dataclass
class EvalReport:
    generated_at: str
    reproduce_command: str
    sample_size: int
    summary: Dict[str, Any]
    suite: str = "smoke"
    dataset: Dict[str, Any] = field(default_factory=dict)
    execution_mode: str = "deterministic"
    per_mode: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    per_case: List[Dict[str, Any]] = field(default_factory=list)


async def run_local_eval(
    rag_mode: str = "hybrid",
    seed_chunks: Optional[Sequence[Any]] = None,
    *,
    suite: str = "smoke",
    dataset_path: Optional[pathlib.Path] = None,
    execution_mode: str = "deterministic",
    judge: Optional[Any] = None,
) -> EvalReport:
    """Run a versioned suite with deterministic local dependencies.

    ``realistic`` executions need externally injected real components and are
    intentionally not silently substituted with fake models here.
    """
    if execution_mode != "deterministic":
        raise ValueError(
            "local_eval_runner only supports deterministic mode; "
            "use the realistic evaluation harness with injected components"
        )
    tracker = DialogueStateTracker()

    from rag.retriever import HybridRetriever
    dense = FakeDenseIndex()
    sparse = FakeSparseIndex()
    reranker = FakeReranker()

    if seed_chunks is None:
        seed_chunks = _default_seed_chunks()
    dense.add(list(seed_chunks))
    sparse.add(list(seed_chunks))

    retriever = HybridRetriever(
        dense, sparse, reranker=reranker, relevance_threshold=0.0,
    )

    dataset = load_dataset(suite, path=dataset_path)
    cases = [case.raw() for case in dataset.cases]

    all_results: Dict[str, List[CaseResult]] = {
        "dst": [], "tool": [], "rag": [], "e2e": [], "nlu": [],
    }
    per_case: List[Dict[str, Any]] = []

    executors = {
        "DST": ("dst", lambda case: _execute_dst_case(case, tracker)),
        "TOOL": (
            "tool",
            lambda case: _execute_tool_case(case, retriever, rag_mode),
        ),
        "RAG": (
            "rag",
            lambda case: _execute_rag_case(case, retriever, mode=rag_mode),
        ),
        "E2E": (
            "e2e",
            lambda case: _execute_e2e_case(case, retriever, rag_mode),
        ),
        "NLU": ("nlu", _execute_nlu_case),
    }
    for case in cases:
        case_id = case.get("id", "")
        prefix = case_id.partition("-")[0]
        if prefix not in executors:
            raise ValueError(f"unsupported evaluation case id: {case_id!r}")
        bucket, executor = executors[prefix]
        result = await executor(case)
        if judge is not None and result.predicted_response:
            from evaluation.judge import JudgeInput

            scores = await judge.judge(JudgeInput(
                user_question=case.get("turns", [case.get("message", "")])[-1],
                candidate_response=result.predicted_response,
                tool_observations=result.observed_tool_results,
                evidence=result.citations,
                expected_behavior=case.get("expected_status"),
            ))
            result.judge_scores = scores.model_dump()
        all_results[bucket].append(result)
        per_case.append(asdict(result))

    summary = _compute_all_metrics(all_results)

    return EvalReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        reproduce_command=(
            ".venv/bin/python -m evaluation.run "
            f"--suite {suite} --mode {execution_mode}"
        ),
        sample_size=len(cases),
        summary=summary,
        suite=suite,
        dataset=dataset_summary(dataset),
        execution_mode=execution_mode,
        per_case=per_case,
    )


async def run_ablation() -> Dict[str, EvalReport]:
    """Run Dense / Hybrid / Hybrid+Reranker ablation."""
    seed = _default_seed_chunks()
    results: Dict[str, EvalReport] = {}
    for mode in ("dense", "hybrid", "rerank"):
        report = await run_local_eval(
            rag_mode=mode,
            seed_chunks=seed,
            suite="smoke",
        )
        results[mode] = report
    return results


async def run_baseline_comparison() -> Dict[str, Any]:
    """Compare deterministic legacy rules with the current local runtime.

    This is a synthetic comparison because the historical implementation is
    not executed from a pinned revision. It must not be presented as a
    measured before/after improvement.
    """
    from core.agent_models import INTENT_SCHEMAS
    cases = [case.raw() for case in load_dataset("smoke").cases]
    e2e_cases = [case for case in cases if case.get("id", "").startswith("E2E-")]

    baseline_completed = 0
    baseline_total = len(e2e_cases)
    for case in e2e_cases:
        expected = case.get("expected_status", "completed")
        turns = case.get("turns", [])
        if len(turns) == 1:
            ft = fast_track_extract(turns[0])
            if ft.intent and ft.slots:
                baseline_completed += (expected == "completed")
            elif ft.intent and ft.intent in INTENT_SCHEMAS:
                schema = INTENT_SCHEMAS[ft.intent]
                if not schema.required_slots:
                    baseline_completed += (expected == "completed")
        elif expected == "awaiting_user":
            baseline_completed += 1

    current_report = await run_local_eval()
    current_tcr = current_report.summary["task_completion_rate"]
    baseline_tcr = baseline_completed / baseline_total if baseline_total else 0.0

    return {
        "comparison_type": "synthetic_or_legacy_rules",
        "supports_measured_improvement_claim": False,
        "legacy_rules_task_completion_rate": round(baseline_tcr, 4),
        "current_task_completion_rate": current_tcr,
        "rate_difference": round(current_tcr - baseline_tcr, 4),
        "e2e_sample_size": baseline_total,
    }


def _default_seed_chunks() -> List[DocumentChunk]:
    """Load deterministic chunks from the versioned evaluation corpus."""
    payload = json.loads(_CORPUS_FILE.read_text(encoding="utf-8"))
    chunks: List[DocumentChunk] = []
    for idx, item in enumerate(payload["chunks"]):
        chunks.append(DocumentChunk(
            chunk_id=item["chunk_id"],
            content=item["content"],
            source=f"evaluation:{payload['version']}",
            title=item.get("title", item["chunk_id"]),
            section=item.get("section", ""),
            chunk_index=idx,
            parent_id=f"doc-{item['chunk_id']}",
            metadata={"corpus_version": payload["version"]},
        ))
    return chunks


# ── Reporting ───────────────────────────────────────────────────────────────

def write_json_report(report: EvalReport, path: pathlib.Path) -> pathlib.Path:
    data = {
        "generated_at": report.generated_at,
        "reproduce_command": report.reproduce_command,
        "sample_size": report.sample_size,
        "suite": report.suite,
        "dataset": report.dataset,
        "execution_mode": report.execution_mode,
        "summary": report.summary,
        "per_mode": {k: {"summary": v.summary} for k, v in report.per_mode.items()} if report.per_mode else {},
        "per_case": report.per_case,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_ablation_json(
    ablation: Dict[str, EvalReport],
    path: pathlib.Path,
) -> pathlib.Path:
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reproduce_command": ".venv/bin/python -m evaluation.local_eval_runner --ablation",
        "modes": {
            mode: {
                "summary": report.summary,
                "sample_size": report.sample_size,
            }
            for mode, report in ablation.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_markdown_report(
    report: EvalReport,
    ablation: Optional[Dict[str, EvalReport]] = None,
    path: Optional[pathlib.Path] = None,
) -> pathlib.Path:
    if path is None:
        path = _REPORTS_DIR / "smoke-deterministic.md"
    lines = [
        "# GGBot Evaluation Report",
        "",
        f"- **Generated at**: {report.generated_at}",
        f"- **Reproduce command**: `{report.reproduce_command}`",
        f"- **Sample size**: {report.sample_size}",
        f"- **Suite / mode**: `{report.suite}` / `{report.execution_mode}`",
        "",
        "## Summary Metrics",
        "",
    ]
    for key, value in report.summary.items():
        if isinstance(value, (int, float)):
            lines.append(f"- **{key}**: {value}")
        elif isinstance(value, dict) and key not in ("intent_detail", "slot_detail", "per_class"):
            lines.append(f"- **{key}**: {json.dumps(value, ensure_ascii=False)}")

    if ablation:
        lines.extend(["", "## Ablation: Dense / Hybrid / Hybrid+Reranker", ""])
        lines.append("| Mode | Intent Acc | Macro-F1 | Slot F1 | DST JGA | R@5 | MRR | Tool Sel | Task Comp |")
        lines.append("|------|-----------|----------|---------|---------|-----|-----|----------|-----------|")
        for mode, rep in ablation.items():
            s = rep.summary
            lines.append(
                f"| {mode} | {s.get('intent_accuracy', 0):.4f} | "
                f"{s.get('intent_macro_f1', 0):.4f} | "
                f"{s.get('slot_f1', 0):.4f} | "
                f"{s.get('dst_joint_goal_accuracy', 0):.4f} | "
                f"{s.get('recall_at_5', 0):.4f} | "
                f"{s.get('mrr', 0):.4f} | "
                f"{s.get('tool_selection_accuracy', 0):.4f} | "
                f"{s.get('task_completion_rate', 0):.4f} |"
            )

    lines.extend(["", "", "## Per-Case Results", ""])
    for case in report.per_case:
        lines.append(f"- **{case.get('case_id', '?')}** ({case.get('category', '?')}): "
                     f"intent={case.get('predicted_intent')}, "
                     f"completed={case.get('completed')}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ── CLI entry point ─────────────────────────────────────────────────────────

async def _main() -> None:
    import sys
    ablation_mode = "--ablation" in sys.argv
    output_dir = _REPORTS_DIR

    if ablation_mode:
        ablation = await run_ablation()
        baseline_report = ablation.get("dense", list(ablation.values())[0])
        write_ablation_json(ablation, output_dir / "ablation_report.json")
        # Baseline vs current comparison
        comparison = await run_baseline_comparison()
        comp_path = output_dir / "baseline_comparison.json"
        comp_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
        write_markdown_report(baseline_report, ablation, output_dir / "eval_report.md")
        for mode, report in ablation.items():
            write_json_report(report, output_dir / f"eval_{mode}.json")
            s = report.summary
            print(f"[{mode}] Intent Acc={s['intent_accuracy']:.4f} "
                  f"Macro-F1={s['intent_macro_f1']:.4f} "
                  f"Slot F1={s['slot_f1']:.4f} "
                  f"DST JGA={s['dst_joint_goal_accuracy']:.4f} "
                  f"R@5={s['recall_at_5']:.4f} "
                  f"MRR={s['mrr']:.4f} "
                  f"Tool Sel={s['tool_selection_accuracy']:.4f} "
                  f"Task Comp={s['task_completion_rate']:.4f}")
        print(f"Baseline vs Current: {comparison}")
    else:
        report = await run_local_eval()
        write_json_report(report, output_dir / "eval_report.json")
        write_markdown_report(report, path=output_dir / "eval_report.md")
        s = report.summary
        print(f"Intent Acc={s['intent_accuracy']:.4f} "
              f"Macro-F1={s['intent_macro_f1']:.4f} "
              f"Slot F1={s['slot_f1']:.4f} "
              f"DST JGA={s['dst_joint_goal_accuracy']:.4f} "
              f"R@5={s['recall_at_5']:.4f} "
              f"MRR={s['mrr']:.4f} "
              f"Tool Sel={s['tool_selection_accuracy']:.4f} "
              f"Task Comp={s['task_completion_rate']:.4f}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(_main())
