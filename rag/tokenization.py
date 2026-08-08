"""Lightweight, model-independent token budgeting for document chunking."""
from __future__ import annotations

import re
from typing import List, Tuple


_TOKEN_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]"
    r"|[A-Za-z0-9]+(?:[._#/-][A-Za-z0-9]+)*"
    r"|[^\s]",
)


def token_spans(text: str) -> List[Tuple[int, int]]:
    """Return deterministic lexical token spans without loading a model."""
    return [match.span() for match in _TOKEN_PATTERN.finditer(text)]


def count_tokens(text: str) -> int:
    """Count conservative token units for multilingual chunk budgets.

    CJK characters, Latin words/identifiers, and punctuation each count as one
    unit. This is stable across deployments and intentionally does not claim
    byte-for-byte parity with any specific embedding or LLM tokenizer.
    """
    return len(token_spans(text))


def split_by_token_budget(text: str, max_tokens: int) -> List[str]:
    """Split text into contiguous pieces containing at most ``max_tokens``."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    spans = token_spans(text)
    if not spans:
        return []
    pieces = []
    for start in range(0, len(spans), max_tokens):
        group = spans[start:start + max_tokens]
        pieces.append(text[group[0][0]:group[-1][1]].strip())
    return [piece for piece in pieces if piece]


def token_suffix(text: str, max_tokens: int) -> str:
    """Return at most the last ``max_tokens`` lexical units from text."""
    if max_tokens <= 0:
        return ""
    spans = token_spans(text)
    if len(spans) <= max_tokens:
        return text.strip()
    return text[spans[-max_tokens][0]:].strip()
