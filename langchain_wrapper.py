"""
langchain_wrapper.py

EvidenceUnit → LangChain / LlamaIndex 변환 래퍼
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
            # 임베딩 검색용 축약 텍스트 (table_html 제외). 임베딩 모델은
            # 보통 입력을 256토큰 안팎에서 truncate하므로, page_content(전체
            # 내용) 대신 이 필드로 벡터스토어를 구성하면 Recall이 개선됨.
            # 예: LangChain MultiVectorRetriever에서 이 값으로 임베딩하고
            # page_content는 그대로 LLM 컨텍스트로 반환.
            "retrieval_text": eu.retrieval_text,
        },
    )


def eu_to_langchain_units(eu: "EvidenceUnit") -> list["LangChainDocument"]:
    """
    EvidenceUnit → LangChain Document 여러 개 (행 단위 다중 벡터, small-to-big).

    eu_to_langchain()은 EU 하나를 문서 하나로 통째로 임베딩하는데, 표
    안의 특정 셀 값을 묻는 질의는 같은 표의 나머지 행 데이터에 묻혀 코사인
    유사도에서 밀리는 문제가 있다 (EvidenceUnit.retrieval_units 참고).

    이 함수는 eu.retrieval_units의 세부 텍스트(문장)마다 별도 Document를
    만들어 반환한다. 벡터스토어에는 이 작은 Document들을 넣어서 검색
    정밀도를 높이고, 실제 LLM에 넘길 때는 page_content(작은 조각) 대신
    metadata["parent_text"](eu.text, 표 전체 맥락)를 사용할 것.

    Returns:
        EU 하나당 len(eu.retrieval_units)개의 작은 Document 목록.
        모두 같은 eu_id/parent_text를 metadata로 공유한다.
    """
    if LangChainDocument is None:
        raise ImportError("langchain_core is not installed. pip install langchain-core")

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
        # LLM 컨텍스트로 반환할 전체 내용 (table_html 포함). 검색은 작은
        # page_content로 하되, 매칭되면 이 필드를 실제 답변 생성에 사용.
        "parent_text": eu.text,
    }

    return [
        LangChainDocument(page_content=unit_text, metadata=dict(base_metadata))
        for unit_text in eu.retrieval_units
    ]


# ---------------------------------------------------------------------------
# LlamaIndex
# ---------------------------------------------------------------------------

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