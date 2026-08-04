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

    chunk_id가 있으면(EvidenceUnit) 그 값을 TextNode.id_로 명시 지정한다.
    chunk_id가 없으면(TextChunk, 표와 무관한 일반 청크) id_를 지정하지
    않아 LlamaIndex가 자동 생성하게 둔다 — 표 청크만 eu_id로 안정적으로
    추적할 대상이고, 일반 청크는 원래도 식별자를 부여하지 않았음.
    """
    if LlamaIndexTextNode is None:
        raise ImportError("llama-index-core is not installed. pip install llama-index-core")

    nodes = []
    for c in chunks:
        kwargs = {"text": c.text, "metadata": dict(c.metadata)}
        if c.chunk_id is not None:
            kwargs["id_"] = c.chunk_id
        nodes.append(LlamaIndexTextNode(**kwargs))
    return nodes


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
        if c.chunk_id is None:
            nodes.append(LlamaIndexTextNode(text=c.text, metadata=dict(c.metadata)))
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
