"""Document loaders and structure-aware chunking."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, List

from rag.models import DocumentChunk, LoadedSection


def load_document(path: str | Path) -> List[LoadedSection]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return [LoadedSection(
            text=file_path.read_text(encoding="utf-8"),
            source=str(file_path),
            title=file_path.stem,
        )]
    if suffix in {".md", ".markdown"}:
        return _load_markdown(file_path)
    if suffix == ".pdf":
        return _load_pdf(file_path)
    raise ValueError(f"unsupported document type: {suffix}")


def _load_markdown(path: Path) -> List[LoadedSection]:
    text = path.read_text(encoding="utf-8")
    sections: List[LoadedSection] = []
    heading_path: List[str] = []
    body: List[str] = []
    title = path.stem

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            sections.append(LoadedSection(
                text=content,
                source=str(path),
                title=title,
                section=" > ".join(heading_path),
            ))
        body.clear()

    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not match:
            body.append(line)
            continue
        flush()
        level = len(match.group(1))
        heading = match.group(2)
        heading_path[:] = heading_path[:level - 1]
        heading_path.append(heading)
        if level == 1:
            title = heading
    flush()
    return sections or [LoadedSection(text=text, source=str(path), title=title)]


def _load_pdf(path: Path) -> List[LoadedSection]:
    try:
        from pypdf import PdfReader
    except ImportError as ex:
        raise RuntimeError("PDF loading requires pypdf") from ex

    reader = PdfReader(str(path))
    sections: List[LoadedSection] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append(LoadedSection(
                text=text,
                source=str(path),
                title=path.stem,
                section=f"page {page_number}",
                page=page_number,
            ))
    return sections


def chunk_sections(
    sections: Iterable[LoadedSection],
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> List[DocumentChunk]:
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap smaller than chunk_size")

    chunks: List[DocumentChunk] = []
    for section in sections:
        parent_id = hashlib.sha256(
            f"{section.source}|{section.section}|{section.page}".encode()
        ).hexdigest()[:16]
        pieces = _recursive_split(section.text, chunk_size, chunk_overlap)
        for index, content in enumerate(pieces):
            chunk_id = hashlib.sha256(
                f"{parent_id}|{index}|{content}".encode()
            ).hexdigest()[:20]
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                content=content,
                source=section.source,
                title=section.title,
                section=section.section,
                page=section.page,
                chunk_index=index,
                parent_id=parent_id,
                metadata=dict(section.metadata),
            ))
    return chunks


def _recursive_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    separators = ("\n\n", "\n", "。", "！", "？", ". ", " ")
    pieces = [text]
    for separator in separators:
        next_pieces: List[str] = []
        for piece in pieces:
            if len(piece) <= chunk_size:
                next_pieces.append(piece)
            else:
                next_pieces.extend(_pack(piece.split(separator), separator, chunk_size))
        pieces = next_pieces

    bounded: List[str] = []
    for piece in pieces:
        bounded.extend(piece[i:i + chunk_size] for i in range(0, len(piece), chunk_size))

    result: List[str] = []
    previous = ""
    for piece in bounded:
        content = f"{previous[-overlap:]}{piece}" if previous and overlap else piece
        result.append(content.strip())
        previous = piece
    return [item for item in result if item]


def _pack(parts: List[str], separator: str, limit: int) -> List[str]:
    packed: List[str] = []
    current = ""
    for part in (part.strip() for part in parts if part.strip()):
        candidate = f"{current}{separator}{part}" if current else part
        if current and len(candidate) > limit:
            packed.append(current)
            current = part
        else:
            current = candidate
    if current:
        packed.append(current)
    return packed
