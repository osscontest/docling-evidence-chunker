"""
langchain.py

EvidenceUnit → LangChain 변환 래퍼
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..unit import EvidenceUnit

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


def text_chunk_to_langchain(chunk) -> "LangChainDocument":
    if LangChainDocument is None:
        raise ImportError("langchain_core is not installed. pip install langchain-core")

    page_no = None
    try:
        page_no = chunk.meta.doc_items[0].prov[0].page_no
    except (AttributeError, IndexError):
        pass

    return LangChainDocument(
        page_content=chunk.text,
        metadata={"eu_id": None, "page_no": page_no, "source": "hybrid"},
    )


# ---------------------------------------------------------------------------
# 검색 결과 후처리 (EU-흡수 문단 카니발라이제이션 제거)
# ---------------------------------------------------------------------------
#
# attach_context_paragraphs()가 표 주변 단락을 eu.context_before/after로
# EU에 편입시켜도, 원본 doc.texts에서는 그 단락을 지우거나 표시하지 않는다.
# 그래서 별도로 HybridChunker(또는 다른 청커)를 돌려 "표와 무관한 일반
# 청크"를 만들면, EU가 이미 흡수한 바로 그 문단이 거기에도 그대로 중복
# 등장한다. 표+캡션+문단이 뭉쳐진 EU(텍스트가 길고 여러 내용이 섞임)와
# 그 문단만 단독으로 담긴 일반 청크(짧고 쿼리와 순수하게 유사도가 높음)가
# 같은 검색 코퍼스 안에서 서로 경쟁하게 되고, 실측(Version01)상 EU가
# 지는 가장 흔한 원인(hybrid_pageX)의 상당 부분이 여기서 기인한 것으로
# 추정된다.
#
# smart-chunker는 벡터스토어/청커를 직접 소유하지 않으므로, 이 제거는
# EU 생성 단계가 아니라 "사용자가 자신의 HybridChunker(등)로 비표 청크를
# 만든 뒤" 적용하는 후처리 유틸로 제공한다.

def filter_consumed_paragraphs(
    chunks: list,
    eu_list: list,
    min_substring_len: int = 20,
) -> list:
    """
    HybridChunker 등이 만든 일반 청크 중, 이미 EU의 context_before/
    context_after로 흡수된 문단과 겹치는(카니발라이제이션) 청크를 제거.

    Args:
        chunks: 텍스트를 가진 청크/Document/Node 리스트. 각 항목은
            page_content(LangChain Document) 또는 text(LlamaIndex
            TextNode, Docling HybridChunker 청크) 속성 중 하나를 가지고
            있어야 함.
        eu_list: SmartChunker가 만든 EvidenceUnit 리스트.
        min_substring_len: 이 길이(정규화 후 문자 수) 미만인 문자열은
            완전 일치(exact match)만 적용하고 부분 포함(substring) 매칭은
            건너뛴다. 짧은 각주/라벨("Results" 같은)이 무관한 긴 청크에
            우연히 포함돼 오탐(false positive)되는 걸 방지하기 위함.

    Returns:
        카니발라이제이션 청크가 제거된 리스트 (원본 순서 유지).
        eu_list의 어떤 EU도 문단을 흡수하지 않았다면 chunks를 그대로 반환.
    """
    def _normalize(text: str) -> str:
        return " ".join((text or "").split())

    consumed_texts = [
        _normalize(p)
        for eu in eu_list
        for p in eu.context_before + eu.context_after
        if p.strip()
    ]

    filtered_chunks = []
    for c in chunks:
        c_text = _normalize(getattr(c, "page_content", None) or getattr(c, "text", None) or "")
        if not c_text:
            continue

        is_consumed = False
        for consumed in consumed_texts:
            if c_text == consumed:
                is_consumed = True
                break
            shorter_len = min(len(c_text), len(consumed))
            if shorter_len < min_substring_len:
                continue
            if c_text in consumed or consumed in c_text:
                is_consumed = True
                break

        if not is_consumed:
            filtered_chunks.append(c)

    return filtered_chunks