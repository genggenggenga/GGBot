"""
亮点：多轮对话记忆管理

三级记忆架构，模拟人类记忆机制：
  1. 工作记忆（Redis）—— 当前会话的最近 N 条消息，毫秒级读写
  2. 情景记忆（ChromaDB）—— 跨会话的历史对话，按语义相似度检索
  3. 用户画像（ChromaDB）—— 从对话中提炼的长期稳定偏好

关键设计（Task 8 调整后）：
  - 消息、DialogueState、情景记忆、用户偏好四类数据源边界清晰
  - 会话摘要采用覆盖式压缩，每次压缩由 LLM 基于旧摘要+新消息生成全新摘要，禁止无限追加
  - 情景记忆只在任务完成（task_completed）或转人工（handoff）事件触发时写入
  - 用户画像只记录稳定偏好，订单号/物流号/订单状态等时效事实禁止写入画像
  - 保持 API 向后兼容，不修改 /chat 调用方式
"""
import asyncio
import hashlib
import inspect
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

import chromadb
import redis
from anthropic import AsyncAnthropic
from pydantic import BaseModel, ConfigDict, Field

from core.llm_utils import extract_text_content
from core.prompts.memory import build_profile_prompt, build_summary_prompt

if TYPE_CHECKING:
    from core.agent_models import DialogueState
    from core.state_store import StateStore

logger = logging.getLogger(__name__)
_EMBEDDING_FUNCTION_UNSET = object()


class _UserProfileOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preferences: List[str] = Field(default_factory=list)
    language: Optional[str] = None
    communication_preference: Optional[str] = None
    contact_preference: Optional[str] = None
    timezone: Optional[str] = None
    accessibility: Optional[str] = None


async def _backend_call(func, *args, **kwargs):
    """Run sync storage clients off-loop while accepting async clients too."""
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    result = await asyncio.to_thread(func, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _resolve_memory_embedding_function():
    """根据 RAG_EMBEDDING_PROVIDER 返回 embedding function 或 None。

    与 mcp/knowledge_base.py 保持一致的策略，避免 ChromaDB 默认 ONNX
    模型（all-MiniLM-L6-v2）的 79MB 运行时下载。
    """
    import os
    provider = os.getenv("RAG_EMBEDDING_PROVIDER", "api").strip().lower()
    model = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-m3")
    if provider in {"api", "siliconflow"}:
        from rag.indexes import build_siliconflow_embedding_function
        return build_siliconflow_embedding_function(model)
    if provider in {"local", "bge"}:
        from rag.indexes import build_bge_embedding_function
        return build_bge_embedding_function(model)
    return None


class MsgRole(Enum):
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"


class EpisodicEventType(str, Enum):
    """情景记忆写入触发事件类型。"""
    TASK_COMPLETED = "task_completed"
    HANDOFF = "handoff"


@dataclass
class Message:
    role:       MsgRole
    content:    str
    timestamp:  datetime = field(default_factory=datetime.now)
    metadata:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryContext:
    """传给 Agent 的完整上下文。四类数据源边界清晰。"""
    recent_messages:  List[Message]              # 工作记忆：最近对话
    relevant_history: List[str]                  # 情景记忆：语义相关的历史片段
    user_profile:     Dict[str, Any]             # 用户画像：长期稳定偏好
    summary:          str                        # 当前会话摘要（覆盖式压缩，有界）
    dialogue_state:   Optional["DialogueState"] = None  # DST 结构化业务状态（独立来源）

    @staticmethod
    def _clean(text: str) -> str:
        """移除 Unicode 代理字符，防止编码错误。"""
        return text.encode("utf-8", errors="ignore").decode("utf-8")

    def to_prompt_text(
        self,
        skill_prompt: str = "",
        observations: Optional[List[Any]] = None,
    ) -> str:
        """按 Skill、状态、近期消息、相关记忆、Observation 组装上下文。"""
        parts = []
        if skill_prompt:
            parts.append(f"[Skills]\n{self._clean(skill_prompt)}")
        if self.dialogue_state is not None:
            # Build meaningful state: only include non-default fields.
            # We compare against known defaults without importing DialogueState at runtime.
            state_dict = self.dialogue_state.model_dump(exclude_none=True)
            defaults = {
                "slots": {},
                "required_slots": [],
                "missing_slots": [],
                "confirmation_status": "not_required",
                "completed_goals": [],
                "queued_goals": [],
                "state_version": 0,
            }
            meaningful = {}
            for k, v in state_dict.items():
                if k in defaults and v == defaults[k]:
                    continue
                meaningful[k] = v
            if meaningful:
                parts.append(f"[当前业务状态]\n{json.dumps(meaningful, ensure_ascii=False)}")
        if self.recent_messages:
            parts.append("[最近对话]")
            for m in self.recent_messages:
                parts.append(f"{m.role.value}: {self._clean(m.content)}")
        if self.summary:
            parts.append(f"[会话摘要]\n{self._clean(self.summary)}")
        if self.relevant_history:
            parts.append("[相关历史]\n" + "\n".join(f"- {self._clean(h)}" for h in self.relevant_history[:3]))
        if self.user_profile:
            parts.append(f"[用户画像]\n{json.dumps(self.user_profile, ensure_ascii=True)}")
        if observations:
            payload = [
                item.model_dump(mode="json")
                if hasattr(item, "model_dump")
                else item
                for item in observations
            ]
            parts.append(f"[Observations]\n{json.dumps(payload, ensure_ascii=False)}")
        return "\n\n".join(parts)


# 时效事实（临时业务数据）模式：这些永远不应写入长期用户画像
_TRANSIENT_PATTERNS = [
    # 订单号 / 物流单号 / 退款编号等（前缀后4位以上数字）
    re.compile(r"\b(ORD|ORDER|SF|JD|YT)\s*-?\s*\d{4,}\b", re.IGNORECASE),
    re.compile(r"\b\d{10,}\b"),  # 长数字ID
    # 订单状态等临时状态词
    re.compile(r"(待发货|已发货|运输中|已签收|退款中|已退款|待审核|处理中|已完成|已取消)"),
    # 问题/业务细节（一次性业务，不是偏好）
    re.compile(r"(我的订单|我的快递|订单号|运单号|物流|退款|退货|取消订单|订单|快递|售后|工单)"),
]

# 稳定偏好关键词：明确表达偏好、习惯、长期选择的语句特征
_STABLE_PREFERENCE_KEYWORDS = [
    "我喜欢", "我偏好", "我习惯", "我希望", "请用", "请说", "以后都", "不要给我",
    "prefer", "like", "always", "never", "please use", "in English", "用中文",
    "用英文", "用日语", "语言", "联系我", "发短信", "发邮件", "不要打电话",
]

# 用户画像字段白名单：只有这些字段可以出现在画像中
_ALLOWED_PROFILE_FIELDS = {
    "preferences", "language", "communication_preference",
    "contact_preference", "timezone", "accessibility",
}

# 摘要最大长度，防止无界增长
SUMMARY_MAX_CHARS = 600


class MemoryManager:
    """
    三级记忆管理器。

    工作记忆存 Redis（TTL 24h），情景记忆和用户画像存 ChromaDB（持久化）。
    Task 8 关键改动：
      - 摘要覆盖式压缩（不追加）
      - 情景记忆事件触发写入（任务完成/转人工）
      - 用户画像稳定偏好门控
      - DialogueState 作为独立数据源注入 MemoryContext
    """

    WORKING_MAX   = 20    # 工作记忆最大条数，超过则触发压缩
    COMPRESS_AT   = 15    # 达到此条数时压缩，保留摘要 + 最近 5 条
    HISTORY_TOP_K = 5     # 情景记忆检索返回条数
    PROFILE_TOP_K = 3     # 当前问题相关的用户画像分片数
    KEEP_RECENT   = 5     # 压缩时保留最近消息条数

    def __init__(
        self,
        redis_url:    str = "redis://localhost:6379/0",
        chroma_host:  str = "localhost",
        chroma_port:  int = 8000,
        chroma_path:  str = "./data/chroma",
        api_key:      str = "",
        base_url:     Optional[str] = None,
        model:        str = "claude-3-5-sonnet-20241022",
        state_store:  Optional["StateStore"] = None,
        durable_store: Optional[Any] = None,
        structured_client: Optional[Any] = None,
        # 测试用注入点
        redis_client: Optional[Any] = None,
        chroma_client: Optional[Any] = None,
        embedding_function: Any = _EMBEDDING_FUNCTION_UNSET,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
        self._model  = model
        self._state_store = state_store
        self._durable_store = durable_store
        self._structured_client = structured_client

        # Redis 客户端（支持注入 fake）
        if redis_client is not None:
            self._redis = redis_client
        else:
            self._redis = redis.from_url(redis_url, decode_responses=True)

        # ChromaDB 客户端（支持注入 fake）
        if chroma_client is not None:
            chroma = chroma_client
        else:
            try:
                chroma = chromadb.HttpClient(
                    host=chroma_host,
                    port=chroma_port,
                    settings=chromadb.Settings(anonymized_telemetry=False),
                )
                chroma.heartbeat()
                logger.info(f"ChromaDB 已连接: {chroma_host}:{chroma_port}")
            except Exception:
                logger.info(f"ChromaDB 服务不可用，使用本地嵌入式模式: {chroma_path}")
                chroma = chromadb.PersistentClient(
                    path=chroma_path,
                    settings=chromadb.Settings(anonymized_telemetry=False),
                )

        # 情景记忆：存储已完成会话/转人工的历史片段
        # 用户画像：存储稳定偏好
        # Embedding function 由 RAG_EMBEDDING_PROVIDER 控制（默认 SiliconFlow API，
        # 避免 ChromaDB 默认 ONNX 模型的 79MB 下载）。
        if embedding_function is _EMBEDDING_FUNCTION_UNSET:
            embedding_function = (
                None
                if chroma_client is not None
                else _resolve_memory_embedding_function()
            )
        self._episodic = chroma.get_or_create_collection(
            "episodic", embedding_function=embedding_function
        )
        self._profile = chroma.get_or_create_collection(
            "user_profile", embedding_function=embedding_function
        )

    # ── 写入 ──────────────────────────────────────────────────────────────────

    async def add_message(
        self,
        user_id: str,
        conv_id: str,
        role:    MsgRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """将一条消息写入工作记忆，超阈值时自动压缩（覆盖式）。"""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        clean_metadata = {
            self._safe_text(k): self._safe_metadata_value(v)
            for k, v in (metadata or {}).items()
        }
        msg = Message(role=role, content=self._safe_text(content), metadata=clean_metadata)
        key = self._wm_key(user_id, conv_id)

        # 追加到 Redis 列表（左推，最新在前）
        await _backend_call(self._redis.lpush, key, json.dumps({
            "role":      msg.role.value,
            "content":   msg.content,
            "ts":        msg.timestamp.isoformat(),
            "metadata":  msg.metadata,
        }))
        await _backend_call(self._redis.expire, key, 86400)

        # 超过压缩阈值时触发覆盖式压缩
        if await _backend_call(self._redis.llen, key) >= self.COMPRESS_AT:
            await self._compress(user_id, conv_id)

    async def update_profile(self, user_id: str, conv_id: str) -> None:
        """
        从当前工作记忆中提炼**稳定偏好**更新用户画像（带门控）。
        订单状态、订单号等时效事实不会写入画像。
        保持旧API签名兼容。
        """
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        messages = await self._get_working_memory(user_id, conv_id)
        if not messages:
            return

        # 快速门控：若最近对话中没有稳定偏好信号，跳过LLM调用
        recent_text = " ".join(m.content for m in messages[-6:])
        if not self._has_stable_preference_signal(recent_text):
            return

        prompt = build_profile_prompt([
            {"role": m.role.value, "content": self._safe_text(m.content)}
            for m in messages[-10:]
        ])

        try:
            if self._structured_client is not None:
                output = await self._structured_client.generate(
                    prompt,
                    _UserProfileOutput,
                    tool_name="submit_user_profile",
                    max_tokens=256,
                    temperature=0.0,
                )
                profile_data = output.model_dump(exclude_none=True)
            else:
                resp = await self._client.messages.create(
                    model=self._model, max_tokens=256, temperature=0.0,
                    system=prompt.system,
                    messages=[{"role": "user", "content": prompt.user}],
                )
                raw = extract_text_content(resp.content)
                profile_data = json.loads(raw)

            # 字段白名单过滤 + 时效事实内容过滤
            filtered = self._filter_profile_data(profile_data)
            if not filtered or not any(v for v in filtered.values() if v):
                return  # 过滤后没有有效偏好，不写入

            # 合并已有画像
            existing = await self._get_profile(user_id)
            merged = self._merge_profile(existing, filtered)

            profile_version_id = str(uuid.uuid4())
            snapshot_id = f"{user_id}_profile_{profile_version_id}"
            ids = [snapshot_id]
            documents = [
                self._safe_text(json.dumps(merged, ensure_ascii=False)),
            ]
            metadatas = [{
                "user_id": user_id,
                "profile_kind": "snapshot",
                "profile_version_id": profile_version_id,
                "ts": datetime.now().isoformat(),
            }]
            for index, (document, payload) in enumerate(
                self._profile_fragments(merged),
            ):
                digest = hashlib.sha256(document.encode("utf-8")).hexdigest()[:12]
                ids.append(f"{snapshot_id}_{index}_{digest}")
                documents.append(document)
                metadatas.append({
                    "user_id": user_id,
                    "profile_kind": "fragment",
                    "profile_version_id": profile_version_id,
                    "profile_payload": json.dumps(
                        payload,
                        ensure_ascii=False,
                    ),
                    "ts": datetime.now().isoformat(),
                })

            if self._durable_store is None:
                existing = await _backend_call(
                    self._profile.get,
                    where={"user_id": user_id},
                )
                existing_ids = existing.get("ids", [])
                if existing_ids:
                    await _backend_call(self._profile.delete, ids=existing_ids)
            await _backend_call(
                self._profile.add,
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            if self._durable_store is not None:
                activate = getattr(
                    self._durable_store,
                    "activate_user_profile_version",
                    None,
                )
                if activate is not None:
                    await _backend_call(
                        activate,
                        user_id,
                        snapshot_id,
                        merged,
                        profile_version_id=profile_version_id,
                    )
            logger.info(f"用户稳定偏好已更新: {user_id}")
        except Exception as ex:
            logger.warning(f"更新用户画像失败: {ex}")

    async def record_episodic_event(
        self,
        user_id: str,
        conv_id: str,
        event_type: EpisodicEventType,
        summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        事件触发式写入情景记忆。只在以下时机调用：
          - 任务完成（TASK_COMPLETED）
          - 转人工（HANDOFF）
        压缩工作记忆时**不会**自动写入情景记忆。
        """
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        messages = await self._get_working_memory(user_id, conv_id)
        existing_summary = (
            await _backend_call(
                self._redis.get,
                self._summary_key(user_id, conv_id),
            )
            or ""
        )

        # 如果没有提供摘要，从最近消息 + 旧摘要生成一个简短摘要
        if summary is None:
            full_context = ""
            if existing_summary:
                full_context += f"[之前摘要]\n{existing_summary}\n\n"
            full_context += "[最近对话]\n" + "\n".join(
                f"{m.role.value}: {m.content}" for m in messages
            )
            summary = await self._generate_summary(
                full_context,
                max_chars=400,
                instruction=(
                    "请用2-4句话总结本次客服会话的核心内容、处理过程和结果。"
                    "突出用户需求、处理结果、是否完成或转人工。"
                ),
                fallback=f"客服会话（{event_type.value}），共{len(messages)}条消息。",
                summary_kind="episodic",
            )

        event_label = "任务完成" if event_type == EpisodicEventType.TASK_COMPLETED else "转人工"
        event_summary = self._safe_text(f"[{event_label}] {summary}")
        full_text = self._safe_text(
            "\n".join(f"{m.role.value}: {m.content}" for m in messages)
        )

        await self._store_episodic(
            user_id=user_id,
            conv_id=conv_id,
            text=full_text,
            summary=event_summary,
            metadata={
                "event_type": event_type.value,
                **(metadata or {}),
            },
        )
        logger.info(f"情景记忆已写入: {user_id}/{conv_id} ({event_type.value})")

    # ── 读取 ──────────────────────────────────────────────────────────────────

    async def get_context(
        self,
        user_id: str,
        conv_id: str,
        query: str = "",
        dialogue_state: Optional["DialogueState"] = None,
    ) -> MemoryContext:
        """
        构建完整的记忆上下文。

        四类数据源：工作记忆、情景记忆、用户画像、DialogueState（DST）。
        保持旧签名兼容：不传入 dialogue_state 时若配置了 state_store 会自动加载。
        """
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        query = self._safe_text(query)

        # 1. 工作记忆（当前会话最近消息）
        recent = await self._get_working_memory(user_id, conv_id)

        # 2. 情景记忆（跨会话语义检索）
        history = await self._search_episodic(
            user_id, query or (recent[-1].content if recent else "")
        )

        # 3. 用户画像
        profile = await self._get_profile(user_id, query=query)

        # 4. 会话摘要（覆盖式，有界）
        summary = (
            await _backend_call(
                self._redis.get,
                self._summary_key(user_id, conv_id),
            )
            or ""
        )

        # 5. DialogueState（DST），独立来源
        state = dialogue_state
        if state is None and self._state_store is not None:
            try:
                state = await self._state_store.load(user_id, conv_id)
            except Exception as ex:
                logger.warning(f"加载 DialogueState 失败: {ex}")

        return MemoryContext(
            recent_messages=recent,
            relevant_history=history,
            user_profile=profile,
            summary=summary,
            dialogue_state=state,
        )

    async def close(self) -> None:
        """Release the injected or internally-created Redis client."""
        close = getattr(self._redis, "aclose", None) or getattr(self._redis, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result

    # ── 压缩（覆盖式，防止 context 爆炸）─────────────────────────────────────

    async def _compress(self, user_id: str, conv_id: str) -> None:
        """
        工作记忆**覆盖式**压缩：
          1. 将旧摘要 + 待压缩消息一起交给 LLM 生成**全新**摘要
          2. 新摘要直接覆盖旧摘要（不追加），并截断到 SUMMARY_MAX_CHARS
          3. 压缩时**不**写情景记忆（情景记忆仅由任务完成/转人工事件触发）
          4. 工作记忆只保留最近 KEEP_RECENT 条
        """
        messages = await self._get_working_memory(user_id, conv_id)
        if len(messages) < self.COMPRESS_AT:
            return

        to_compress = messages[:-self.KEEP_RECENT]
        keep        = messages[-self.KEEP_RECENT:]

        # 构建输入：旧摘要 + 待压缩消息
        skey = self._summary_key(user_id, conv_id)
        old_summary = await _backend_call(self._redis.get, skey) or ""

        parts = []
        if old_summary:
            parts.append(f"[之前摘要]\n{old_summary}")
        parts.append("[新对话]")
        parts.extend(f"{m.role.value}: {m.content}" for m in to_compress)
        combined = "\n".join(parts)

        new_summary = await self._generate_summary(
            combined,
            max_chars=SUMMARY_MAX_CHARS,
            instruction="请用简洁、连贯的一段话总结本次客服对话的完整进展，涵盖之前摘要和新对话中的关键信息。控制在200字以内。",
            fallback=f"对话包含{len(messages)}条消息（摘要生成失败）。",
        )

        # 直接覆盖旧摘要（核心改动：不追加）
        await _backend_call(self._redis.setex, skey, 86400, new_summary)

        # 重置工作记忆为最近 KEEP_RECENT 条
        key = self._wm_key(user_id, conv_id)
        await _backend_call(self._redis.delete, key)
        for m in reversed(keep):
            await _backend_call(self._redis.lpush, key, json.dumps({
                "role": m.role.value, "content": m.content,
                "ts": m.timestamp.isoformat(), "metadata": m.metadata,
            }))
        await _backend_call(self._redis.expire, key, 86400)
        logger.info(
            f"工作记忆覆盖式压缩完成: {user_id}/{conv_id}，摘要 {len(new_summary)} 字"
        )

    async def _generate_summary(
        self,
        text: str,
        max_chars: int,
        instruction: str,
        fallback: str,
        summary_kind: Literal[
            "working_memory",
            "episodic",
        ] = "working_memory",
    ) -> str:
        """调用 LLM 生成摘要，失败时返回 fallback。结果按 max_chars 截断。"""
        prompt = build_summary_prompt(
            self._safe_text(text),
            self._safe_text(instruction),
            summary_kind=summary_kind,
        )
        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=300, temperature=0.0,
                system=prompt.system,
                messages=[{"role": "user", "content": prompt.user}],
            )
            summary = self._safe_text(extract_text_content(resp.content)).strip()
            if not summary:
                summary = fallback
        except Exception:
            summary = fallback
        # 有界截断：防止摘要无限增长
        if len(summary) > max_chars:
            summary = summary[:max_chars].rstrip() + "…"
        return summary

    # ── 偏好门控辅助 ──────────────────────────────────────────────────────────

    def _has_stable_preference_signal(self, text: str) -> bool:
        """快速判断文本中是否可能包含稳定偏好信号。"""
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in _STABLE_PREFERENCE_KEYWORDS)

    def _filter_profile_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """字段白名单 + 内容模式过滤，移除时效事实。"""
        result: Dict[str, Any] = {}
        for key, value in data.items():
            if key not in _ALLOWED_PROFILE_FIELDS:
                continue
            if isinstance(value, str):
                if self._contains_transient(value):
                    continue
                result[key] = value
            elif isinstance(value, list):
                filtered_list = [
                    item for item in value
                    if isinstance(item, str) and not self._contains_transient(item)
                ]
                if filtered_list:
                    result[key] = filtered_list
            elif isinstance(value, dict):
                filtered_dict = self._filter_profile_data(value)
                if filtered_dict:
                    result[key] = filtered_dict
            else:
                result[key] = value
        return result

    def _contains_transient(self, text: str) -> bool:
        """判断文本是否包含时效事实（订单号/状态/当前问题等）。"""
        return any(p.search(text) for p in _TRANSIENT_PATTERNS)

    def _merge_profile(
        self,
        existing: Dict[str, Any],
        new: Dict[str, Any],
    ) -> Dict[str, Any]:
        """合并新偏好到已有画像，新值优先，列表去重。"""
        merged = dict(existing)
        for key, value in new.items():
            if key == "preferences" and isinstance(value, list):
                existing_prefs = existing.get("preferences", [])
                if isinstance(existing_prefs, list):
                    merged["preferences"] = list(dict.fromkeys(
                        [*existing_prefs, *value]
                    ))
                else:
                    merged["preferences"] = value
            else:
                merged[key] = value
        return merged

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    async def _get_working_memory(self, user_id: str, conv_id: str) -> List[Message]:
        key  = self._wm_key(user_id, conv_id)
        raws = await _backend_call(
            self._redis.lrange,
            key,
            0,
            self.WORKING_MAX - 1,
        )
        msgs = []
        for raw in reversed(raws):  # Redis lpush 最新在前，reversed 还原时序
            d = json.loads(raw)
            msgs.append(Message(
                role=MsgRole(d["role"]),
                content=d["content"],
                timestamp=datetime.fromisoformat(d["ts"]),
                metadata=d.get("metadata", {}),
            ))
        return msgs

    async def _search_episodic(self, user_id: str, query: str) -> List[str]:
        """语义检索情景记忆。"""
        query_text = self._safe_text(query).strip()
        if not query_text:
            return []
        try:
            results = await _backend_call(
                self._episodic.query,
                query_texts=[query_text],
                n_results=self.HISTORY_TOP_K,
                where={"user_id": self._safe_text(user_id)},
            )
            docs = results["documents"][0] if results["documents"] else []
            return [self._safe_text(doc) for doc in docs if isinstance(doc, str) and doc.strip()]
        except Exception as ex:
            logger.warning(f"情景记忆检索失败: {ex}")
            return []

    async def _store_episodic(
        self,
        user_id: str,
        conv_id: str,
        text: str,
        summary: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """将一段对话摘要存入情景记忆。仅由 record_episodic_event 调用。"""
        try:
            user_id = self._safe_text(user_id)
            conv_id = self._safe_text(conv_id)
            text = self._safe_text(text)
            summary = self._safe_text(summary)
            doc_id = hashlib.md5(f"{user_id}{conv_id}{time.time()}".encode()).hexdigest()
            meta = {
                "user_id": user_id,
                "conv_id": conv_id,
                "ts": datetime.now().isoformat(),
                "full_text": self._safe_text(text[:500]),
            }
            if metadata:
                meta.update(self._safe_metadata_value(metadata))
            await _backend_call(
                self._episodic.add,
                ids=[doc_id],
                documents=[summary],
                metadatas=[meta],
            )
        except Exception as ex:
            logger.warning(f"存储情景记忆失败: {ex}")

    async def _get_profile(
        self,
        user_id: str,
        query: str = "",
    ) -> Dict[str, Any]:
        """Get the full snapshot or vector-relevant profile fragments."""
        try:
            active_profile: Optional[Dict[str, Any]] = None
            if self._durable_store is not None:
                get_active = getattr(
                    self._durable_store,
                    "get_active_profile_version",
                    None,
                )
                if get_active is not None:
                    active_profile = await _backend_call(get_active, user_id)

            if active_profile is not None:
                version_id = active_profile["profile_version_id"]
                if query.strip():
                    try:
                        results = await _backend_call(
                            self._profile.query,
                            query_texts=[self._safe_text(query)],
                            n_results=self.PROFILE_TOP_K,
                            where={
                                "$and": [
                                    {"user_id": {"$eq": user_id}},
                                    {"profile_kind": {"$eq": "fragment"}},
                                    {"profile_version_id": {"$eq": version_id}},
                                ],
                            },
                        )
                        metadatas = (
                            results.get("metadatas", [[]])[0]
                            if results.get("metadatas")
                            else []
                        )
                        relevant: Dict[str, Any] = {}
                        for metadata in metadatas:
                            payload = (metadata or {}).get("profile_payload")
                            if not payload:
                                continue
                            relevant = self._merge_profile(
                                relevant,
                                json.loads(payload),
                            )
                        if relevant:
                            return relevant
                    except Exception as ex:
                        logger.warning(f"用户画像分片检索失败，回退到画像快照: {ex}")
                return dict(active_profile["profile"])

            if query.strip():
                results = await _backend_call(
                    self._profile.query,
                    query_texts=[self._safe_text(query)],
                    n_results=self.PROFILE_TOP_K,
                    where={
                        "$and": [
                            {"user_id": {"$eq": user_id}},
                            {"profile_kind": {"$eq": "fragment"}},
                        ],
                    },
                )
                metadatas = (
                    results.get("metadatas", [[]])[0]
                    if results.get("metadatas")
                    else []
                )
                relevant: Dict[str, Any] = {}
                for metadata in metadatas:
                    payload = (metadata or {}).get("profile_payload")
                    if not payload:
                        continue
                    relevant = self._merge_profile(
                        relevant,
                        json.loads(payload),
                    )
                if relevant:
                    return relevant

            results = await _backend_call(
                self._profile.get,
                where={
                    "$and": [
                        {"user_id": {"$eq": user_id}},
                        {"profile_kind": {"$eq": "snapshot"}},
                    ],
                },
                limit=1,
            )
            if results["documents"]:
                return json.loads(results["documents"][0])
            # Backward compatibility for profiles stored before fragmentation.
            legacy = await _backend_call(
                self._profile.get,
                where={"user_id": user_id},
                limit=1,
            )
            if legacy["documents"]:
                return json.loads(legacy["documents"][0])
        except Exception:
            pass
        return {}

    @staticmethod
    def _profile_fragments(
        profile: Dict[str, Any],
    ) -> List[tuple[str, Dict[str, Any]]]:
        """Build semantically meaningful documents from profile fields."""
        labels = {
            "preferences": "用户偏好",
            "language": "语言偏好",
            "communication_preference": "沟通偏好",
            "contact_preference": "联系方式偏好",
            "timezone": "时区偏好",
            "accessibility": "无障碍需求",
        }
        fragments: List[tuple[str, Dict[str, Any]]] = []
        for field_name, value in profile.items():
            label = labels.get(field_name, field_name)
            values = value if isinstance(value, list) else [value]
            for item in values:
                if item is None or item == "":
                    continue
                payload_value = [item] if isinstance(value, list) else item
                fragments.append((
                    f"{label}：{item}",
                    {field_name: payload_value},
                ))
        return fragments

    @staticmethod
    def _wm_key(user_id: str, conv_id: str) -> str:
        return f"wm:{user_id}:{conv_id}"

    @staticmethod
    def _summary_key(user_id: str, conv_id: str) -> str:
        return f"summary:{user_id}:{conv_id}"

    @staticmethod
    def _safe_text(value: Any) -> str:
        """转成 ChromaDB 可接受的普通 UTF-8 字符串。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @classmethod
    def _safe_metadata_value(cls, value: Any) -> Any:
        """递归清洗 metadata，避免 Redis/ChromaDB 后续读写遇到非法 UTF-8。"""
        if isinstance(value, str):
            return cls._safe_text(value)
        if isinstance(value, dict):
            return {cls._safe_text(k): cls._safe_metadata_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._safe_metadata_value(v) for v in value]
        return value
