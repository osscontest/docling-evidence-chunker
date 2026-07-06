"""
langchain_wrapper.py

EvidenceUnit → LangChain / LlamaIndex 변환 래퍼

26.07.01 기준
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from interfaces import EvidenceUnit

# ---------------------------------------------------------------------------
# LangChain
# ---------------------------------------------------------------------------

try:
    from langchain_core.documents import Document as LangChainDocument
except ImportError:
    LangChainDocument = None  # type: ignore


def eu_to_langchain(eu: "EvidenceUnit") -> "LangChainDocument":
    """
    EvidenceUnit → LangChain Document 변환.
    RAG 파이프라인에 넘길 때 사용.

    Args:
        eu: 완성된 EvidenceUnit (split 포함)

    Returns:
        LangChain Document
    """
    if LangChainDocument is None:
        raise ImportError("langchain_core is not installed. pip install langchain-core")

    return LangChainDocument(
        page_content=eu.text,
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
        },
    )


# ---------------------------------------------------------------------------
# LlamaIndex (W4 예정)
# ---------------------------------------------------------------------------

def eu_to_llamaindex(eu: "EvidenceUnit"):
    """
    EvidenceUnit → LlamaIndex TextNode 변환.
    W4에 구현.
    """
    raise NotImplementedError  # W4에 구현