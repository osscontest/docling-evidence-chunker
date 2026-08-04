"""
langchain.py

RetrievalChunk(EvidenceUnit 또는 export.TextChunk) → LangChain 변환.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import RetrievalChunk

try:
    from langchain_core.documents import Document as LangChainDocument
except ImportError:
    LangChainDocument = None  # type: ignore


def to_langchain(chunks: list["RetrievalChunk"]) -> list["LangChainDocument"]:
    """
    RetrievalChunk 리스트 → LangChain Document 리스트 (1 chunk = 1 Document).
    RAG 파이프라인에 넘길 때 사용.

    chunk.metadata를 그대로 옮긴다 — EvidenceUnit.metadata는 "retrieval_text"
    (임베딩 검색용 축약 텍스트, table_html 제외)를 포함하고, export.TextChunk
    (표와 무관한 일반 청크)의 metadata는 {eu_id: None, page_no, source}뿐이다.
    임베딩 모델은 보통 입력을 256토큰 안팎에서 truncate하므로, page_content
    (전체 내용) 대신 metadata["retrieval_text"]로 벡터스토어를 구성하면
    Recall이 개선된다 (LangChain MultiVectorRetriever 등에서 활용).
    """
    if LangChainDocument is None:
        raise ImportError("langchain_core is not installed. pip install langchain-core")

    return [
        LangChainDocument(page_content=c.text, metadata=dict(c.metadata))
        for c in chunks
    ]


def to_langchain_units(chunks: list["RetrievalChunk"]) -> list["LangChainDocument"]:
    """
    RetrievalChunk 리스트 → LangChain Document 리스트 (행 단위 다중 벡터, small-to-big).

    to_langchain()은 chunk 하나를 문서 하나로 통째로 임베딩하는데, 표 안의
    특정 셀 값을 묻는 질의는 같은 표의 나머지 행 데이터에 묻혀 코사인
    유사도에서 밀리는 문제가 있다 (EvidenceUnit.retrieval_units 참고).

    chunk_id가 있는(=EvidenceUnit) chunk만 retrieval_units 단위로 쪼갠다.
    chunk_id가 없는(=TextChunk, 표와 무관한 일반 청크) 항목은 그 자체가
    이미 최소 단위라 쪼갤 게 없으므로 to_langchain()과 동일하게 1개 그대로
    반환 — retrieval_units가 [text] 하나뿐이라 쪼개도 결과는 같지만,
    metadata에 parent_text가 잘못 붙는 걸 막기 위해 분기한다(원본 일반
    청크에는 parent_text 개념이 없음).

    벡터스토어에는 이 작은 Document들을 넣어서 검색 정밀도를 높이고,
    실제 LLM에 넘길 때는 page_content(작은 조각) 대신
    metadata["parent_text"](chunk.text, 표 전체 맥락)를 사용할 것.
    """
    if LangChainDocument is None:
        raise ImportError("langchain_core is not installed. pip install langchain-core")

    docs = []
    for c in chunks:
        if c.chunk_id is None:
            docs.append(LangChainDocument(page_content=c.text, metadata=dict(c.metadata)))
            continue

        base_metadata = {k: v for k, v in c.metadata.items() if k != "retrieval_text"}
        base_metadata["parent_text"] = c.text
        docs.extend(
            LangChainDocument(page_content=unit_text, metadata=dict(base_metadata))
            for unit_text in c.retrieval_units
        )
    return docs
