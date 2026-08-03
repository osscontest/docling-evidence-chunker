"""
llamaindex.py

EvidenceUnit → LlamaIndex 변환 래퍼
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..unit import EvidenceUnit

try:
    from llama_index.core.schema import TextNode as LlamaIndexTextNode
except ImportError:
    LlamaIndexTextNode = None  # type: ignore


def eu_to_llamaindex(eu: "EvidenceUnit") -> "LlamaIndexTextNode":
    """
    EvidenceUnit → LlamaIndex TextNode 변환.
    LlamaIndex RAG 파이프라인에 넘길 때 사용.

    Args:
        eu: 완성된 EvidenceUnit (split 포함)

    Returns:
        LlamaIndex TextNode
    """
    if LlamaIndexTextNode is None:
        raise ImportError("llama-index-core is not installed. pip install llama-index-core")

    return LlamaIndexTextNode(
        id_=eu.eu_id,
        text=eu.text,
        metadata={
            "eu_id": eu.eu_id,
            "page_no": eu.page_no,
            "section_header": eu.section_header,
            "caption_text": eu.caption_text,
            "bbox": list(eu.bbox),
            "is_split": eu.is_split,
            "split_index": eu.split_index,
            "total_splits": eu.total_splits,
            "caption_confidence": eu.caption_confidence,
            # 임베딩 검색용 축약 텍스트. eu_to_langchain()의 동일 필드 설명 참고.
            "retrieval_text": eu.retrieval_text,
        },
    )


def eu_to_llamaindex_units(eu: "EvidenceUnit") -> list["LlamaIndexTextNode"]:
    """
    EvidenceUnit → LlamaIndex TextNode 여러 개 (행 단위 다중 벡터, small-to-big).
    eu_to_langchain_units()와 동일 목적. metadata["parent_text"](eu.text)를
    실제 LLM 컨텍스트로, node.text(작은 조각)는 검색 전용으로 쓸 것.
    """
    if LlamaIndexTextNode is None:
        raise ImportError("llama-index-core is not installed. pip install llama-index-core")

    base_metadata = {
        "eu_id": eu.eu_id,
        "page_no": eu.page_no,
        "section_header": eu.section_header,
        "caption_text": eu.caption_text,
        "bbox": list(eu.bbox),
        "is_split": eu.is_split,
        "split_index": eu.split_index,
        "total_splits": eu.total_splits,
        "caption_confidence": eu.caption_confidence,
        "parent_text": eu.text,
    }

    return [
        LlamaIndexTextNode(
            id_=f"{eu.eu_id}-u{i}",
            text=unit_text,
            metadata=dict(base_metadata),
        )
        for i, unit_text in enumerate(eu.retrieval_units)
    ]


def text_chunk_to_llamaindex(chunk) -> "LlamaIndexTextNode":
    """
    표와 무관한 일반 본문 청크(Docling HybridChunker가 만든 chunk) →
    LlamaIndex TextNode 변환. text_chunk_to_langchain()과 동일 목적.
    """
    if LlamaIndexTextNode is None:
        raise ImportError("llama-index-core is not installed. pip install llama-index-core")

    page_no = None
    try:
        page_no = chunk.meta.doc_items[0].prov[0].page_no
    except (AttributeError, IndexError):
        pass

    return LlamaIndexTextNode(
        text=chunk.text,
        metadata={"eu_id": None, "page_no": page_no, "source": "hybrid"},
    )
