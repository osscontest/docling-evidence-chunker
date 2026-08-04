"""
chunker.py

EvidenceChunker 메인 클래스.

Usage:
    from evidence_chunker import EvidenceChunker
    from evidence_chunker.export.langchain import to_langchain, to_langchain_units

    chunker = EvidenceChunker()
    eus = chunker.chunk("paper.pdf")                # List[EvidenceUnit] (표만)

    # build_corpus()는 표(EU) + 일반 본문을 한 번에 합쳐서 반환한다. EU가
    # 이미 흡수한 문단은 자동으로 중복 제거되므로, 이 한 줄 호출만으로
    # 바로 벡터스토어에 넣을 수 있는 최종 코퍼스가 나온다.
    chunks = chunker.build_corpus("paper.pdf")      # List[RetrievalChunk] (표+본문)
    docs = to_langchain(chunks)                     # List[LangChainDocument]

    # 표만 필요하거나 직접 다른 텍스트 청커를 쓰고 싶으면 chunk()만 쓸 것.
    docs_tables_only = to_langchain(chunker.chunk("paper.pdf"))

    # 행 단위 다중 벡터(small-to-big): 표의 특정 셀 값을 묻는 질의에 강함.
    # 벡터스토어에는 이 작은 Document/Node들을 넣고, 검색 후에는
    # metadata["parent_text"](표 전체 맥락)를 LLM에 넘길 것.
    docs = to_langchain_units(chunks)               # List[LangChainDocument]
"""

from __future__ import annotations

from pathlib import Path

from .unit import EvidenceUnit
from .split import split_oversized_units


# ---------------------------------------------------------------------------
# Docling 참조 헬퍼
#
# scripts/06_build_eu.py의 get_prov/resolve_ref를 그대로 옮김 (Stage 2 커밋
# "빌더 통합"). context.py._get_prov / caption.py의 유사 로직과 중복이지만,
# 헬퍼 통합은 별도 커밋("_docling_shim.py") 몫이라 여기서는 건드리지 않는다.
# ---------------------------------------------------------------------------

def get_prov(item_dict: dict) -> tuple[int, dict]:
    prov_list = item_dict.get("prov", [])
    if not prov_list:
        return -1, {}
    p = prov_list[0]
    return p.get("page_no", -1), p.get("bbox", {})


def resolve_ref(doc, cref: str) -> dict:
    """'#/texts/3' 형태의 cref → model_dump() dict."""
    try:
        parts = cref.strip("#/").split("/")
        obj = doc
        for p in parts:
            obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
        return obj.model_dump() if hasattr(obj, "model_dump") else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# EU 빌더
#
# scripts/06_build_eu.py의 build_evidence_units()를 정본으로 채택해 그대로
# 옮김 — chunker.EvidenceChunker._build_eu()(구 버전)가 아니라 이쪽이 벤치마크
# 수치(baseline/recall_before.txt 등)를 만든 구현이기 때문. 호출 순서(먼저
# attach_context_paragraphs, 그 다음 flatten/table_abstract)도 06 그대로
# 유지 — 순서를 바꾸면 abstract-as-similarity-reference 동작이 달라진다
# (별도 커밋에서 의도적으로 다룸).
# ---------------------------------------------------------------------------

def build_evidence_units(
    doc,
    bbox_threshold: float = 300.0,
    sim_threshold: float = 0.40,
    doc_id: str | None = None,
) -> list[EvidenceUnit]:
    if doc_id is None:
        doc_id = getattr(doc, "name", None) or "doc"

    from .geometry import normalize_bbox
    from .context import attach_context_paragraphs
    from .caption import map_table_caption
    from .flatten import (
        build_col_header_map,
        build_table_abstract,
        group_sentences_by_row,
    )
    from .filters import find_duplicate_tables, is_toc_or_lof_decoy
    from .parser.docling import DoclingParser

    # filters.py는 Stage 4에서 ParsedDoc 기반으로 전환됨 — 아직 caption.py/
    # context.py는 raw DoclingDocument를 그대로 쓰므로, 이 함수는 당분간
    # 두 가지를 동시에 들고 간다. 재파싱은 아니라 from_doc()이 이미 파싱된
    # doc을 순회만 하는 가벼운 변환임.
    parsed = DoclingParser().from_doc(doc)

    page_sizes: dict[int, dict] = {}
    if hasattr(doc, "pages"):
        for pg_key, pg_val in doc.pages.items():
            pg_dict = pg_val.model_dump() if hasattr(pg_val, "model_dump") else {}
            page_sizes[int(pg_key)] = pg_dict.get("size", {})

    # ── 중복 표 감지: Docling이 표 1개를 TableItem 2개로 중복 인식하는 경우
    #    제거되는 쪽(loser)의 캡션이 direct로 잡혀 있으면 남는 쪽(winner)에 물려줌 ──
    dup_drop_map = find_duplicate_tables(parsed)
    dup_donor_caption = {}
    for loser_idx, winner_idx in dup_drop_map.items():
        donor_mapping = map_table_caption(doc, doc.tables[loser_idx], loser_idx)
        if donor_mapping.caption_text:
            dup_donor_caption[winner_idx] = donor_mapping

    eu_list: list[EvidenceUnit] = []
    page_counters: dict[int, int] = {}

    for table_index, table in enumerate(doc.tables):
        if table_index in dup_drop_map:
            continue  # 중복 표: 더 세밀하게 구조화된 쪽만 남김

        if is_toc_or_lof_decoy(parsed, parsed.tables[table_index]):
            continue  # v03 p3 필터: 목차/그림·표 목록이 표로 오인식된 경우 EU 생성 대상에서 제외

        t_dict = table.model_dump()
        pg, bbox = get_prov(t_dict)
        if pg == -1:
            continue

        idx = page_counters.get(pg, 0)
        page_counters[pg] = idx + 1
        eu_id = f"{doc_id}-p{pg}-{idx}"

        # ── 캡션 (RefItem 직접 연결 + bbox fallback, caption.py) ──
        cap_mapping = map_table_caption(doc, table, table_index)
        if cap_mapping.confidence == "none" and table_index in dup_donor_caption:
            cap_mapping = dup_donor_caption[table_index]
        caption_text = cap_mapping.caption_text
        caption_confidence = cap_mapping.confidence

        # ── 각주 ────────────────────────────────────────────────────
        footnote_text = None
        fn_refs = t_dict.get("footnotes", [])
        if fn_refs:
            cref = (fn_refs[0].get("cref", "")
                    if isinstance(fn_refs[0], dict)
                    else getattr(fn_refs[0], "cref", ""))
            fn_dict = resolve_ref(doc, cref)
            footnote_text = fn_dict.get("text") or None

        # ── 표 HTML ─────────────────────────────────────────────────
        try:
            table_html = table.export_to_html(doc) or None
        except Exception:
            table_html = None

        # ── bbox 정규화 ──────────────────────────────────────────────
        ps = page_sizes.get(pg, {})
        norm_bbox = normalize_bbox(bbox, ps.get("width", 1.0), ps.get("height", 1.0))

        eu = EvidenceUnit(
            eu_id=eu_id,
            page_no=pg,
            doc_id=doc_id,
            caption_text=caption_text,
            table_html=table_html,
            footnote_text=footnote_text,
            bbox=norm_bbox,
            caption_confidence=caption_confidence,
        )

        # ── 섹션 헤더 + 인접 단락 (bbox 거리 + 임베딩 유사도, context.py) ──
        # table_bbox를 직접 넘김: eu_id의 페이지 내 순번은 dedup으로 doc.tables의
        # 스캔 순서와 어긋날 수 있어, eu_id 기반 재추정에 맡기면 안 됨.
        attach_context_paragraphs(eu, doc, bbox_threshold, sim_threshold, table_bbox=bbox)

        # ── Row Flattening + 다단 헤더 처리 ─────────────────────────
        data = t_dict.get("data", {})
        cells = data.get("table_cells", [])
        num_rows = data.get("num_rows", 0)
        num_cols = data.get("num_cols", 0)

        eu.row_sentence_map = group_sentences_by_row(cells, num_rows, num_cols, footnote_text)
        eu.flattened_rows = [
            s for row in sorted(eu.row_sentence_map) for s in eu.row_sentence_map[row]
        ]

        # ── Table Abstract ───────────────────────────────────────────
        col_map = build_col_header_map(cells, num_cols)
        eu.table_abstract = build_table_abstract(caption_text, col_map, num_rows, eu.section_header)

        eu_list.append(eu)

    return eu_list


# ---------------------------------------------------------------------------
# EvidenceChunker
# ---------------------------------------------------------------------------

class EvidenceChunker:
    """
    PDF → Evidence Unit 파이프라인 래퍼.

    Docling으로 파싱 후 EvidenceUnit을 구성하고,
    context_attacher로 인접 단락을 붙여 반환.

    Args:
        artifacts_path: Docling 로컬 모델 경로.
                        None이면 HuggingFace Hub에서 자동 다운로드.
        bbox_threshold: 표 위아래 단락 수집 범위 (PDF 포인트). 기본 300pt.
        sim_threshold:  코사인 유사도 임계값. 기본 0.40.
    """

    def __init__(
        self,
        artifacts_path: str | None = None,
        bbox_threshold: float = 300.0,
        sim_threshold: float = 0.40,
    ) -> None:
        self.artifacts_path = artifacts_path
        self.bbox_threshold = bbox_threshold
        self.sim_threshold = sim_threshold
        self._converter = None  # 첫 chunk() 호출 시 초기화 (lazy)

    # ------------------------------------------------------------------
    # 퍼블릭 API
    # ------------------------------------------------------------------

    def chunk(
        self,
        pdf_path: str | Path,
        doc_id: str | None = None,
    ) -> list[EvidenceUnit]:
        """
        PDF에서 표(Evidence Unit)만 추출.

        표와 무관한 일반 본문까지 합친 검색 코퍼스가 필요하면
        build_corpus()를 쓸 것.

        Args:
            pdf_path: PDF 파일 경로
            doc_id:   문서 식별자. 기본은 파일명 stem(예: "paper.pdf" → "paper").
                      PDF 여러 개를 한 벡터스토어에 넣을 때 eu_id 충돌(서로 다른
                      문서의 "eu-p3-0" 같은 것)을 막는 접두사로 쓰인다. 같은
                      파일명이 다른 디렉터리에 있을 수 있으면 직접 지정할 것.

        Returns:
            List[EvidenceUnit]
        """
        pdf_path = str(pdf_path)
        doc = self._parse(pdf_path)
        eu_list = build_evidence_units(doc, self.bbox_threshold, self.sim_threshold, doc_id)
        return split_oversized_units(eu_list)

    def build_corpus(
        self,
        pdf_path: str | Path,
        doc_id: str | None = None,
    ) -> list:
        """
        PDF를 표(EU) + 일반 본문을 합친 검색 코퍼스로 변환.

        EU가 context_before/after로 이미 흡수한 문단은 자동으로 중복
        제거되므로, 이 한 번 호출로 바로 벡터스토어에 넣을 수 있는 최종
        코퍼스가 나온다 (같은 문단이 EU와 일반 청크 양쪽에 중복 등장해
        검색 코퍼스 안에서 서로 경쟁하는 카니발라이제이션 방지).

        Args:
            pdf_path: PDF 파일 경로
            doc_id:   chunk() 참고

        Returns:
            List[RetrievalChunk] — EvidenceUnit(표)과 export.TextChunk(일반
            본문)이 섞인 리스트. 둘 다 chunk_id/text/retrieval_text/
            retrieval_units/metadata를 노출하므로 export.to_langchain() 등에
            타입 구분 없이 그대로 넘기면 됨. chunk_id가 None인 항목이 일반
            본문 청크(EU 아님)를 뜻한다.
        """
        from .export import TextChunk

        pdf_path = str(pdf_path)
        doc = self._parse(pdf_path)
        eu_list = build_evidence_units(doc, self.bbox_threshold, self.sim_threshold, doc_id)
        eu_list = split_oversized_units(eu_list)
        text_chunks = self._build_text_chunks(doc, eu_list)
        return eu_list + [TextChunk(c) for c in text_chunks]

    def _build_text_chunks(self, doc, eu_list: list[EvidenceUnit]) -> list:
        """
        표와 무관한 일반 본문 청크 생성 (Docling HybridChunker), 원본 청크
        그대로 반환 (export.TextChunk 래핑은 build_corpus()가 함).

        EU가 context_before/after로 이미 흡수한 문단과 겹치는 청크는
        export.filter_consumed_paragraphs()로 제거한다 — 그렇게 안 하면
        같은 문단이 EU 안에도, 여기 일반 청크로도 중복 등장해서 검색
        코퍼스 안에서 서로 경쟁하는 문제(카니발라이제이션)가 생긴다
        (Version01 벤치마크에서 EU 검색 실패의 주요 원인으로 확인됨).
        """
        from docling.chunking import HybridChunker
        from docling_core.types.doc import DocItemLabel
        from .export import filter_consumed_paragraphs

        all_chunks = list(HybridChunker().chunk(doc))
        is_table_chunk = lambda c: any(
            di.label == DocItemLabel.TABLE for di in c.meta.doc_items
        )
        text_chunks = [c for c in all_chunks if not is_table_chunk(c)]
        return filter_consumed_paragraphs(text_chunks, eu_list)

    # ------------------------------------------------------------------
    # 내부 단계
    # ------------------------------------------------------------------

    def _parse(self, pdf_path: str):
        """Docling으로 PDF 파싱 → DoclingDocument."""
        if self._converter is None:
            self._converter = self._make_converter()
        result = self._converter.convert(pdf_path)
        return result.document

    def _make_converter(self):
        """DocumentConverter 생성. 로컬 모델 경로가 있으면 우선 사용."""
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        pipeline_options = PdfPipelineOptions()
        if self.artifacts_path:
            pipeline_options.artifacts_path = self.artifacts_path

        return DocumentConverter(
            format_options={
                "pdf": PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
