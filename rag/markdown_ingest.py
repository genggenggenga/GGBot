"""Annotated Markdown ingestion for enterprise customer-service knowledge."""
from __future__ import annotations

import hashlib
import re
import shlex
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rag.loaders import ChunkingConfig, _recursive_split
from rag.models import DocumentChunk
from rag.versioning import normalize_knowledge_metadata


SUPPORTED_CHUNK_TYPES = {
    "faq",
    "policy_rule",
    "sop",
    "table",
    "guardrail",
    "section_parent",
}

_BEGIN_RE = re.compile(r"<!--\s*GGKB:BEGIN\s+(.*?)-->", re.DOTALL)
_END_RE = re.compile(r"<!--\s*GGKB:END\s*-->")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class ParsedBlock:
    """A marked knowledge candidate block extracted from Markdown."""

    attrs: Dict[str, str]
    content: str
    start_line: int
    title: str
    section_path: str


def chunks_from_annotated_markdown(
    filename: str,
    content: bytes | str,
    *,
    chunking_config: Optional[ChunkingConfig] = None,
    metadata_overrides: Optional[Dict[str, Any]] = None,
) -> List[DocumentChunk]:
    """Parse marked Markdown knowledge blocks and return canonical chunks."""
    if isinstance(content, bytes):
        try:
            text = content.decode("utf-8", errors="strict")
        except UnicodeDecodeError as ex:
            raise ValueError("Markdown 文件必须使用 UTF-8 编码") from ex
    else:
        text = content
    document_metadata, body = _split_frontmatter(text)
    document_metadata.update({
        key: value
        for key, value in (metadata_overrides or {}).items()
        if value is not None
    })
    blocks = parse_annotated_markdown(body)
    if not blocks:
        raise ValueError("Markdown 文件未包含 GGKB 标注块")

    config = chunking_config or ChunkingConfig.from_env()
    chunks: List[DocumentChunk] = []
    for block in blocks:
        chunks.extend(_chunks_for_block(filename, document_metadata, block, config))
    return chunks


def parse_annotated_markdown(text: str) -> List[ParsedBlock]:
    """Extract GGKB marked candidate blocks from Markdown text."""
    blocks: List[ParsedBlock] = []
    position = 0
    while True:
        begin = _BEGIN_RE.search(text, position)
        if begin is None:
            break
        end = _END_RE.search(text, begin.end())
        if end is None:
            raise ValueError(
                f"GGKB 标注块缺少结束标记，line={_line_number(text, begin.start())}",
            )
        attrs = _parse_attrs(begin.group(1))
        chunk_type = attrs.get("type", "")
        if chunk_type not in SUPPORTED_CHUNK_TYPES:
            raise ValueError(f"unsupported GGKB chunk type: {chunk_type}")
        if not attrs.get("id"):
            raise ValueError("GGKB 标注块缺少 id")
        content = text[begin.end():end.start()].strip()
        if not content:
            raise ValueError(f"GGKB 标注块内容为空: {attrs['id']}")
        title, section_path = _title_and_section(content)
        blocks.append(ParsedBlock(
            attrs=attrs,
            content=content,
            start_line=_line_number(text, begin.start()),
            title=title,
            section_path=section_path,
        ))
        position = end.end()
    return blocks


def _chunks_for_block(
    filename: str,
    document_metadata: Dict[str, Any],
    block: ParsedBlock,
    config: ChunkingConfig,
) -> List[DocumentChunk]:
    chunk_type = block.attrs["type"]
    if chunk_type == "table":
        table_chunks = _table_chunks(filename, document_metadata, block)
        if table_chunks:
            return table_chunks

    metadata = _block_metadata(filename, document_metadata, block)
    normalized = normalize_knowledge_metadata(
        metadata,
        source=filename,
        title=block.title,
    )
    pieces = _typed_text_pieces(block.content, chunk_type, config)
    parent_id = _parent_id(filename, block, normalized)
    chunks = []
    for index, piece in enumerate(pieces):
        chunks.append(DocumentChunk(
            chunk_id=_chunk_id(parent_id, index, piece),
            content=piece,
            source=filename,
            title=block.title,
            section=block.section_path,
            chunk_index=index,
            parent_id=parent_id,
            metadata={
                **normalized,
                "unit_id": block.attrs["id"],
                "block_start_line": block.start_line,
            },
        ))
    return chunks


def _typed_text_pieces(
    text: str,
    chunk_type: str,
    config: ChunkingConfig,
) -> List[str]:
    if chunk_type in {"faq", "sop", "guardrail", "section_parent"}:
        return [text]
    return _recursive_split(
        text,
        chunk_size=config.chunk_size,
        overlap=config.chunk_overlap,
    )


def _table_chunks(
    filename: str,
    document_metadata: Dict[str, Any],
    block: ParsedBlock,
) -> List[DocumentChunk]:
    rows = _extract_markdown_table_rows(block.content)
    if not rows:
        return []

    metadata = _block_metadata(filename, document_metadata, block)
    normalized = normalize_knowledge_metadata(
        metadata,
        source=filename,
        title=block.title,
    )
    parent_id = _parent_id(filename, block, normalized)
    chunks = []
    for index, row in enumerate(rows):
        cells = "，".join(f"{key}={value}" for key, value in row.items())
        piece = f"{block.title}\n{cells}"
        chunks.append(DocumentChunk(
            chunk_id=_chunk_id(parent_id, index, piece),
            content=piece,
            source=filename,
            title=block.title,
            section=block.section_path,
            chunk_index=index,
            parent_id=parent_id,
            metadata={
                **normalized,
                "unit_id": block.attrs["id"],
                "block_start_line": block.start_line,
                "table_row_index": index,
            },
        ))
    return chunks


def _extract_markdown_table_rows(text: str) -> List[Dict[str, str]]:
    lines = [line.strip() for line in text.splitlines()]
    rows: List[Dict[str, str]] = []
    for index, line in enumerate(lines[:-2]):
        if not _is_table_row(line) or not _is_separator_row(lines[index + 1]):
            continue
        headers = _table_cells(line)
        for row_line in lines[index + 2:]:
            if not _is_table_row(row_line):
                break
            values = _table_cells(row_line)
            if len(values) != len(headers):
                continue
            rows.append(dict(zip(headers, values)))
        break
    return rows


def _is_table_row(line: str) -> bool:
    return line.startswith("|") and line.endswith("|") and line.count("|") >= 2


def _is_separator_row(line: str) -> bool:
    if not _is_table_row(line):
        return False
    cells = _table_cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _table_cells(line: str) -> List[str]:
    return [cell.strip() for cell in line.strip("|").split("|")]


def _block_metadata(
    filename: str,
    document_metadata: Dict[str, Any],
    block: ParsedBlock,
) -> Dict[str, Any]:
    attrs = dict(block.attrs)
    chunk_type = attrs.pop("type")
    unit_id = attrs.pop("id")
    risk_level = attrs.pop("risk_level", attrs.pop("risk", "normal"))
    metadata: Dict[str, Any] = {
        **document_metadata,
        **attrs,
        "source": filename,
        "title": block.title,
        "section_path": block.section_path,
        "chunk_type": chunk_type,
        "unit_id": unit_id,
        "risk_level": risk_level,
        "guardrail": chunk_type == "guardrail",
    }
    metadata.setdefault("status", "published")
    return metadata


def _parent_id(
    filename: str,
    block: ParsedBlock,
    metadata: Dict[str, Any],
) -> str:
    explicit = block.attrs.get("parent_id")
    if explicit:
        return explicit
    version_id = str(metadata.get("version_id", ""))
    identity = f"{filename}|{block.attrs['id']}|{version_id}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _chunk_id(parent_id: str, index: int, content: str) -> str:
    return hashlib.sha256(
        f"{parent_id}|{index}|{content}".encode("utf-8"),
    ).hexdigest()[:20]


def _split_frontmatter(text: str) -> tuple[Dict[str, Any], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            frontmatter = "\n".join(lines[1:index])
            body = "\n".join(lines[index + 1:])
            return _parse_frontmatter(frontmatter), body
    raise ValueError("Markdown frontmatter 缺少结束标记 ---")


def _parse_frontmatter(text: str) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"unsupported frontmatter line: {raw_line}")
        key, value = line.split(":", 1)
        values[key.strip()] = _scalar(value.strip())
    return values


def _parse_attrs(text: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for item in shlex.split(text):
        if "=" not in item:
            raise ValueError(f"unsupported GGKB attribute: {item}")
        key, value = item.split("=", 1)
        attrs[key.strip()] = value.strip()
    return attrs


def _scalar(value: str) -> Any:
    if value == "":
        return ""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value.strip("\"'")


def _title_and_section(text: str) -> tuple[str, str]:
    headings = [match.group(2).strip() for match in _HEADING_RE.finditer(text)]
    if headings:
        return headings[-1], " > ".join(headings)
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    title = first_line[:80] or "untitled"
    return title, title


def _line_number(text: str, position: int) -> int:
    return text.count("\n", 0, position) + 1
