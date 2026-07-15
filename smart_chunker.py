"""
smart_chunker.py

EvidenceChunker 메인 클래스.

Usage:
    from smart_chunker import SmartChunker

    chunker = SmartChunker()
    eus = chunker.chunk("paper.pdf")                                # List[EvidenceUnit]
    docs = chunker.chunk("paper.pdf", output="langchain")           # List[LangChainDocument]
    nodes = chunker.chunk("paper.pdf", output="llamaindex")         # List[LlamaIndex TextNode]

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

from interfaces import EvidenceUnit


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
    ) -> list:
        """
        PDF를 Evidence Unit으로 청킹.

        Args:
            pdf_path: PDF 파일 경로
            output:   반환 타입
                      "eu"               → List[EvidenceUnit]
                      "langchain"        → List[LangChainDocument] (EU 1개 = Document 1개)
                      "llamaindex"       → List[LlamaIndex TextNode] (EU 1개 = TextNode 1개)
                      "langchain_units"  → List[LangChainDocument] (EU 1개 = 행 단위 여러 개,
                                            small-to-big. metadata["parent_text"]가 전체 맥락)
                      "llamaindex_units" → List[LlamaIndex TextNode] (위와 동일, LlamaIndex용)

        Returns:
            output 타입에 맞는 리스트
        """
        pdf_path = str(pdf_path)
        doc = self._parse(pdf_path)
        eu_list = self._build_eu(doc)  # context_attacher까지 이 안에서 처리됨

        if output == "langchain":
            from langchain_wrapper import eu_to_langchain
            return [eu_to_langchain(eu) for eu in eu_list]

        if output == "llamaindex":
            from langchain_wrapper import eu_to_llamaindex
            return [eu_to_llamaindex(eu) for eu in eu_list]

        if output == "langchain_units":
            from langchain_wrapper import eu_to_langchain_units
            return [doc for eu in eu_list for doc in eu_to_langchain_units(eu)]

        if output == "llamaindex_units":
            from langchain_wrapper import eu_to_llamaindex_units
            return [node for eu in eu_list for node in eu_to_llamaindex_units(eu)]

        return eu_list

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
        from bbox_utils import normalize_bbox
        from context_attacher import attach_context_paragraphs
        from scripts._caption_mapper import map_table_caption
        from scripts._table_utils import (
            build_col_header_map,
            build_table_abstract,
            find_duplicate_tables,
            group_sentences_by_row,
        )

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
