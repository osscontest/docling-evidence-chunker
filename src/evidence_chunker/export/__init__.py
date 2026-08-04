"""
export

EvidenceUnit + 일반 본문 청크를 균일하게 다루기 위한 공통 인터페이스.

EvidenceChunker.build_corpus()는 EvidenceUnit(표)와 TextChunk(HybridChunker가
만든 일반 본문 청크)를 한 리스트에 섞어 반환한다. 두 타입 모두 RetrievalChunk
프로토콜(chunk_id/text/retrieval_text/retrieval_units/metadata)을 만족하므로,
export.langchain/export.llamaindex의 to_*() 함수가 타입을 구분하는 isinstance
분기 없이 두 타입을 동일하게 처리할 수 있다.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RetrievalChunk(Protocol):
    chunk_id: str | None
    is_atomic: bool
    text: str
    retrieval_text: str
    retrieval_units: list[str]
    metadata: dict


class TextChunk:
    """
    표와 무관한 일반 본문 청크(Docling HybridChunker가 만든 chunk)를
    RetrievalChunk 프로토콜에 맞춰 감싼 래퍼.

    doc_id/index를 받아 chunk_id를 "{doc_id}-hybrid-{index}" 형태로 채운다
    (예전엔 chunk_id가 항상 None이라, PDF 여러 개를 build_corpus()로
    합치면 일반 본문 청크끼리 서로 구별이 안 되는 실사용 버그가 있었음 —
    page_no도 문서 간 충돌함). is_atomic=True가 EvidenceUnit과의 실질적
    차이 — export의 to_*_units() 함수들이 이 값으로 "쪼갤 대상인지"를
    구분한다(일반 청크는 애초에 그 자체로 최소 단위라 쪼갤 게 없음).
    chunk_id가 채워진 뒤로는 그 값(None 여부)으로 구분할 수 없어졌다.
    """

    def __init__(self, chunk, doc_id: str, index: int) -> None:
        self.text = chunk.text
        self.retrieval_text = chunk.text
        self.retrieval_units = [chunk.text]
        self.chunk_id = f"{doc_id}-hybrid-{index}"
        self.is_atomic = True

        page_no = None
        try:
            page_no = chunk.meta.doc_items[0].prov[0].page_no
        except (AttributeError, IndexError):
            pass
        self.metadata = {"eu_id": None, "doc_id": doc_id, "page_no": page_no, "source": "hybrid"}


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
# 코퍼스 조립 단계(EvidenceChunker.build_corpus())의 일부이므로 여기에 둔다.

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
        eu_list: EvidenceChunker가 만든 EvidenceUnit 리스트.
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
