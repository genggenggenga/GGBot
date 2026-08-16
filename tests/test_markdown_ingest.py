import pytest

from rag.loaders import ChunkingConfig
from rag.markdown_ingest import (
    chunks_from_annotated_markdown,
    parse_annotated_markdown,
)


def test_parse_annotated_markdown_requires_marked_blocks():
    with pytest.raises(ValueError, match="未包含 GGKB 标注块"):
        chunks_from_annotated_markdown("kb.md", "# 普通文档\n\n未标注内容")


def test_markdown_ingest_rejects_non_utf8_bytes():
    with pytest.raises(ValueError, match="UTF-8"):
        chunks_from_annotated_markdown("kb.md", b"\xff\xfe\xfd")


def test_annotated_markdown_generates_typed_chunks_with_metadata():
    markdown = """---
knowledge_id: enterprise-customer-service
version: v1
business_domain: customer_service
---

<!-- GGKB:BEGIN type=faq id=refund.apply intent=refund_request risk=normal -->
## 退款申请 FAQ

用户可以在订单详情页申请退款。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=guardrail id=privacy.protect intent=privacy_protection risk=high -->
## 隐私保护规则

不得透露他人订单、手机号、地址或支付信息。
<!-- GGKB:END -->
"""

    chunks = chunks_from_annotated_markdown(
        "enterprise.md",
        markdown,
        chunking_config=ChunkingConfig(chunk_size=120, chunk_overlap=20),
    )

    assert [chunk.metadata["chunk_type"] for chunk in chunks] == [
        "faq",
        "guardrail",
    ]
    assert chunks[0].metadata["knowledge_id"] == "enterprise-customer-service"
    assert chunks[0].metadata["intent"] == "refund_request"
    assert chunks[1].metadata["guardrail"] is True
    assert chunks[1].metadata["risk_level"] == "high"
    assert chunks[0].section == "退款申请 FAQ"


def test_metadata_overrides_replace_frontmatter_version():
    markdown = """---
knowledge_id: enterprise-customer-service
version: v1
---

<!-- GGKB:BEGIN type=faq id=refund.apply intent=refund_request -->
## 退款申请 FAQ

用户可以在订单详情页申请退款。
<!-- GGKB:END -->
"""

    chunks = chunks_from_annotated_markdown(
        "enterprise.md",
        markdown,
        metadata_overrides={"version": "v2", "version_seq": 2},
    )

    assert chunks[0].metadata["version"] == "v2"
    assert chunks[0].metadata["version_id"] == "enterprise-customer-service:v2"
    assert chunks[0].metadata["version_seq"] == 2


def test_markdown_table_expands_rows_with_header_context():
    markdown = """---
knowledge_id: enterprise-customer-service
version: v1
---

<!-- GGKB:BEGIN type=table id=logistics.delay intent=logistics_delay -->
## 物流延迟补偿标准

| 场景 | 判定条件 | 补偿标准 |
|---|---|---|
| 标准配送延迟 | 超过承诺送达时间 3-5 天 | 10 元优惠券 |
| 严重配送延迟 | 超过承诺送达时间 6 天及以上 | 20 元优惠券 |
<!-- GGKB:END -->
"""

    chunks = chunks_from_annotated_markdown("enterprise.md", markdown)

    assert len(chunks) == 2
    assert chunks[0].metadata["chunk_type"] == "table"
    assert chunks[0].metadata["table_row_index"] == 0
    assert "场景=标准配送延迟" in chunks[0].content
    assert "补偿标准=20 元优惠券" in chunks[1].content


def test_parse_rejects_unsupported_type():
    markdown = """<!-- GGKB:BEGIN type=unknown id=x -->
## X
body
<!-- GGKB:END -->
"""

    with pytest.raises(ValueError, match="unsupported GGKB chunk type"):
        parse_annotated_markdown(markdown)
