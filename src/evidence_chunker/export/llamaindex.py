"""
llamaindex.py

RetrievalChunk(EvidenceUnit 또는 export.TextChunk) → LlamaIndex 변환.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import RetrievalChunk

try:
    from llama_index.core.schema import TextNode as LlamaIndexTextNode
except ImportError:
    LlamaIndexTextNode = None  # type: ignore


def to_llamaindex(chunks: list["RetrievalChunk"]) -> list["LlamaIndexTextNode"]:
    """
    RetrievalChunk 리스트 → LlamaIndex TextNode 리스트 (1 chunk = 1 TextNode).
    LlamaIndex RAG 파이프라인에 넘길 때 사용.

    chunk.chunk_id를 TextNode.id_로 명시 지정한다. EvidenceUnit(eu_id)과
    export.TextChunk("{doc_id}-hybrid-{index}") 둘 다 항상 안정적인 값을
    주므로(예전엔 TextChunk의 chunk_id가 항상 None이라 LlamaIndex가 자동
    생성하는 임시 id에 의존했음 — PDF 여러 개를 합치면 일반 본문 청크의
    출처를 구분할 방법이 없었던 실사용 버그), 무조건 지정한다.
    """
    if LlamaIndexTextNode is None:
        raise ImportError("llama-index-core is not installed. pip install llama-index-core")

    return [
        LlamaIndexTextNode(text=c.text, metadata=dict(c.metadata), id_=c.chunk_id)
        for c in chunks
    ]


def to_llamaindex_units(chunks: list["RetrievalChunk"]) -> list["LlamaIndexTextNode"]:
    """
    RetrievalChunk 리스트 → LlamaIndex TextNode 리스트 (행 단위 다중 벡터, small-to-big).
    to_langchain_units()와 동일 목적/분기 기준. metadata["parent_text"](chunk.text)를
    실제 LLM 컨텍스트로, node.text(작은 조각)는 검색 전용으로 쓸 것.
    """
    if LlamaIndexTextNode is None:
        raise ImportError("llama-index-core is not installed. pip install llama-index-core")

    nodes = []
    for c in chunks:
        if c.is_atomic:
            nodes.append(LlamaIndexTextNode(text=c.text, metadata=dict(c.metadata), id_=c.chunk_id))
            continue

        base_metadata = {k: v for k, v in c.metadata.items() if k != "retrieval_text"}
        base_metadata["parent_text"] = c.text
        nodes.extend(
            LlamaIndexTextNode(
                id_=f"{c.chunk_id}-u{i}",
                text=unit_text,
                metadata=dict(base_metadata),
            )
            for i, unit_text in enumerate(c.retrieval_units)
        )
    return nodes
