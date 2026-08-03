"""
chunker.py

EvidenceChunker 메인 클래스.

Usage:
    from evidence_chunker import SmartChunker

    chunker = SmartChunker()
    eus = chunker.chunk("paper.pdf")                                # List[EvidenceUnit] (표만)

    # output="langchain"/"llamaindex"(및 _units)는 기본으로 표(EU) + 일반
    # 본문을 한 번에 합쳐서 반환한다 (include_text=True가 기본값). EU가
    # 이미 흡수한 문단은 자동으로 중복 제거되므로, 이 한 줄 호출만으로
    # 바로 벡터스토어에 넣을 수 있는 최종 코퍼스가 나온다.
    docs = chunker.chunk("paper.pdf", output="langchain")           # List[LangChainDocument] (표+본문)
    nodes = chunker.chunk("paper.pdf", output="llamaindex")         # List[LlamaIndex TextNode] (표+본문)

    # 표만 필요하거나 직접 다른 텍스트 청커를 쓰고 싶으면 include_text=False
    docs_tables_only = chunker.chunk("paper.pdf", output="langchain", include_text=False)

    # 행 단위 다중 벡터(small-to-big): 표의 특정 셀 값을 묻는 질의에 강함.
    # 벡터스토어에는 이 작은 Document/Node들을 넣고, 검색 후에는
    # metadata["parent_text"](표 전체 맥락)를 LLM에 넘길 것.
    docs = chunker.chunk("paper.pdf", output="langchain_units")     # List[LangChainDocument]
    nodes = chunker.chunk("paper.pdf", output="llamaindex_units")   # List[LlamaIndex TextNode]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from .unit import EvidenceUnit


# ---------------------------------------------------------------------------
# SmartChunker
# ---------------------------------------------------------------------------

class SmartChunker:
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
        output: Literal[
            "eu", "langchain", "llamaindex", "langchain_units", "llamaindex_units"
        ] = "eu",
        include_text: bool = True,
    ) -> list:
        """
        PDF를 Evidence Unit(+ 필요시 일반 본문)으로 청킹.

        Args:
            pdf_path: PDF 파일 경로
            output:   반환 타입
                      "eu"               → List[EvidenceUnit] (표만. include_text 무시)
                      "langchain"        → List[LangChainDocument] (EU 1개 = Document 1개)
                      "llamaindex"       → List[LlamaIndex TextNode] (EU 1개 = TextNode 1개)
                      "langchain_units"  → List[LangChainDocument] (EU 1개 = 행 단위 여러 개,
                                            small-to-big. metadata["parent_text"]가 전체 맥락)
                      "llamaindex_units" → List[LlamaIndex TextNode] (위와 동일, LlamaIndex용)
            include_text: output이 "eu"가 아닐 때, 표와 무관한 일반 본문도
                      Docling HybridChunker로 청킹해서 같이 반환할지 여부.
                      기본 True — `chunker.chunk(pdf_path, output="langchain")` 한 번
                      호출로 표+본문이 이미 중복 없이 정리된 최종 코퍼스를 받을 수
                      있게 하기 위함. EU가 이미 context_before/after로 흡수한 문단은
                      langchain_wrapper.filter_consumed_paragraphs()로 자동 제거되므로
                      같은 문단이 EU와 일반 청크 양쪽에 중복 등장하지 않는다.
                      직접 다른 텍스트 청커를 쓰고 싶거나 표만 필요하면 False로 끌 것.
                      반환 리스트 안에서는 metadata["eu_id"]가 None인 항목이 일반
                      본문 청크(EU 아님)를 뜻한다.

        Returns:
            output 타입에 맞는 리스트. output != "eu"이고 include_text=True이면
            같은 타입(LangChainDocument 등) 안에 EU 유래 항목과 일반 본문 유래
            항목이 함께 섞여 있다 — 리스트 자체의 타입은 항상 균일하다.
        """
        pdf_path = str(pdf_path)
        doc = self._parse(pdf_path)
        eu_list = self._build_eu(doc)  # context_attacher까지 이 안에서 처리됨

        if output == "eu":
            return eu_list

        text_chunks = self._build_text_chunks(doc, eu_list) if include_text else []

        if output == "langchain":
            from .export.langchain import eu_to_langchain, text_chunk_to_langchain
            return [eu_to_langchain(eu) for eu in eu_list] + \
                   [text_chunk_to_langchain(c) for c in text_chunks]

        if output == "llamaindex":
            from .export.llamaindex import eu_to_llamaindex, text_chunk_to_llamaindex
            return [eu_to_llamaindex(eu) for eu in eu_list] + \
                   [text_chunk_to_llamaindex(c) for c in text_chunks]

        if output == "langchain_units":
            from .export.langchain import eu_to_langchain_units, text_chunk_to_langchain
            return [d for eu in eu_list for d in eu_to_langchain_units(eu)] + \
                   [text_chunk_to_langchain(c) for c in text_chunks]

        if output == "llamaindex_units":
            from .export.llamaindex import eu_to_llamaindex_units, text_chunk_to_llamaindex
            return [n for eu in eu_list for n in eu_to_llamaindex_units(eu)] + \
                   [text_chunk_to_llamaindex(c) for c in text_chunks]

        return eu_list

    def _build_text_chunks(self, doc, eu_list: list[EvidenceUnit]) -> list:
        """
        표와 무관한 일반 본문 청크 생성 (Docling HybridChunker).
        EU가 context_before/after로 이미 흡수한 문단과 겹치는 청크는
        langchain_wrapper.filter_consumed_paragraphs()로 제거한다 —
        그렇게 안 하면 같은 문단이 EU 안에도, 여기 일반 청크로도 중복
        등장해서 검색 코퍼스 안에서 서로 경쟁하는 문제(카니발라이제이션)가
        생긴다 (Version01 벤치마크에서 EU 검색 실패의 주요 원인으로 확인됨).
        """
        from docling.chunking import HybridChunker
        from docling_core.types.doc import DocItemLabel
        from .export.langchain import filter_consumed_paragraphs

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

    def _build_eu(self, doc) -> list[EvidenceUnit]:
        """DoclingDocument → List[EvidenceUnit] (context 없는 기본 뼈대)."""
        from .geometry import normalize_bbox
        from .context import attach_context_paragraphs
        from .caption import map_table_caption
        from .flatten import (
            build_col_header_map,
            build_table_abstract,
            group_sentences_by_row,
        )
        from .filters import find_duplicate_tables, is_toc_or_lof_decoy

        page_sizes: dict[int, dict] = {}
        if hasattr(doc, "pages"):
            for pg_key, pg_val in doc.pages.items():
                pg_dict = pg_val.model_dump() if hasattr(pg_val, "model_dump") else {}
                page_sizes[int(pg_key)] = pg_dict.get("size", {})

        # 중복 표 감지 (Docling이 표 1개를 TableItem 2개로 중복 인식하는 경우 대응)
        dup_drop_map = find_duplicate_tables(doc)
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

            if is_toc_or_lof_decoy(doc, table):
                continue  # v03 p3 필터: 목차/그림·표 목록이 표로 오인식된 경우 EU 생성 대상에서 제외

            t_dict = table.model_dump()
            pg, bbox = self._get_prov(t_dict)
            if pg == -1:
                continue

            idx = page_counters.get(pg, 0)
            page_counters[pg] = idx + 1
            eu_id = f"eu-p{pg}-{idx}"

            # 캡션
            cap_mapping = map_table_caption(doc, table, table_index)
            if cap_mapping.confidence == "none" and table_index in dup_donor_caption:
                cap_mapping = dup_donor_caption[table_index]

            # 각주
            footnote_text = self._resolve_footnote(doc, t_dict)

            # 표 HTML
            try:
                table_html = table.export_to_html(doc) or None
            except Exception:
                table_html = None

            # Row Flattening + Abstract
            data = t_dict.get("data", {})
            cells = data.get("table_cells", [])
            num_rows = data.get("num_rows", 0)
            num_cols = data.get("num_cols", 0)
            row_sentence_map = group_sentences_by_row(cells, num_rows, num_cols, footnote_text)
            flattened = [
                s for row in sorted(row_sentence_map) for s in row_sentence_map[row]
            ]
            col_map = build_col_header_map(cells, num_cols)
            abstract = build_table_abstract(cap_mapping.caption_text, col_map, num_rows, None)

            # bbox 정규화
            ps = page_sizes.get(pg, {})
            norm_bbox = normalize_bbox(bbox, ps.get("width", 1.0), ps.get("height", 1.0))

            eu = EvidenceUnit(
                eu_id=eu_id,
                page_no=pg,
                caption_text=cap_mapping.caption_text,
                caption_confidence=cap_mapping.confidence,
                table_html=table_html,
                footnote_text=footnote_text,
                flattened_rows=flattened,
                table_abstract=abstract,
                bbox=norm_bbox,
            )
            eu.row_sentence_map = row_sentence_map

            # 섹션 헤더 + 인접 단락 (bbox 거리 + 임베딩 유사도, context_attacher.py).
            # table_bbox(원본 PDF 포인트 bbox)를 여기서 직접 넘김 — eu_id의
            # 페이지 내 순번은 dedup으로 doc.tables의 스캔 순서와 어긋날 수
            # 있어서, attach_context_paragraphs가 eu_id로 재추정하게 두면
            # 제거된 중복 표의 위치를 잘못 집어올 수 있다.
            attach_context_paragraphs(
                eu, doc, self.bbox_threshold, self.sim_threshold, table_bbox=bbox
            )

            eu_list.append(eu)

        return eu_list

    # ------------------------------------------------------------------
    # 헬퍼
    # ------------------------------------------------------------------

    @staticmethod
    def _get_prov(item_dict: dict) -> tuple[int, dict]:
        prov_list = item_dict.get("prov", [])
        if not prov_list:
            return -1, {}
        p = prov_list[0]
        return p.get("page_no", -1), p.get("bbox", {})

    @staticmethod
    def _resolve_footnote(doc, t_dict: dict) -> str | None:
        fn_refs = t_dict.get("footnotes", [])
        if not fn_refs:
            return None
        ref = fn_refs[0]
        cref = ref.get("cref", "") if isinstance(ref, dict) else getattr(ref, "cref", "")
        if not cref:
            return None
        try:
            parts = cref.strip("#/").split("/")
            obj = doc
            for p in parts:
                obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
            d = obj.model_dump() if hasattr(obj, "model_dump") else {}
            return d.get("text") or None
        except Exception:
            return None
