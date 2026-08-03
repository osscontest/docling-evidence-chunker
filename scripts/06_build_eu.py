"""
STEP 6: EvidenceUnit 실제 구성.

포함 기능:
  - caption_text    : _caption_mapper.map_table_caption() (RefItem 직접 연결 + bbox fallback)
  - section_header  : context_attacher.attach_context_paragraphs() (표 위쪽 가장 가까운 섹션 헤더)
  - context_before/after: context_attacher.attach_context_paragraphs() (bbox 거리 + 임베딩 유사도 필터)
  - table_html      : export_to_html(doc)
  - flattened_rows  : 셀 → 자연어 문장 (Row Flattening, _table_utils)
  - table_abstract  : 표 요약 문자열 (multi-granularity 검색)
  - footnote_text   : footnotes RefItem 역참조
  - bbox            : normalize_bbox() 0~1 변환
  - caption_confidence: direct / inferred / none (_caption_mapper.CaptionMapping.confidence)

Usage:
    python scripts/06_build_eu.py [path/to/file.pdf]
"""
import sys
import os
import json

if hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PDF_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "data", "pdfs", "attention_is_all_you_need.pdf"
)


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ---------------------------------------------------------------------------
# Docling 헬퍼
# ---------------------------------------------------------------------------

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


def get_prov(item_dict: dict) -> tuple[int, dict]:
    prov_list = item_dict.get("prov", [])
    if not prov_list:
        return -1, {}
    p = prov_list[0]
    return p.get("page_no", -1), p.get("bbox", {})


# ---------------------------------------------------------------------------
# EU 빌더
# ---------------------------------------------------------------------------

def build_evidence_units(doc) -> list:
    from evidence_chunker.unit import EvidenceUnit
    from evidence_chunker.geometry import normalize_bbox
    from evidence_chunker.caption import map_table_caption
    from evidence_chunker.flatten import (
        build_col_header_map,
        build_row_header_map,
        build_table_abstract,
        group_sentences_by_row,
    )
    from evidence_chunker.filters import find_duplicate_tables, is_toc_or_lof_decoy
    from evidence_chunker.context import attach_context_paragraphs

    page_sizes: dict[int, dict] = {}
    if hasattr(doc, "pages"):
        for pg_key, pg_val in doc.pages.items():
            pg_dict = pg_val.model_dump() if hasattr(pg_val, "model_dump") else {}
            page_sizes[int(pg_key)] = pg_dict.get("size", {})

    # ── 중복 표 감지: Docling이 표 1개를 TableItem 2개로 중복 인식하는 경우
    #    제거되는 쪽(loser)의 캡션이 direct로 잡혀 있으면 남는 쪽(winner)에 물려줌 ──
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
        pg, bbox = get_prov(t_dict)
        if pg == -1:
            continue

        idx = page_counters.get(pg, 0)
        page_counters[pg] = idx + 1
        eu_id = f"eu-p{pg}-{idx}"

        # ── 캡션 (RefItem 직접 연결 + bbox fallback, _caption_mapper.py) ──
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
            caption_text=caption_text,
            table_html=table_html,
            footnote_text=footnote_text,
            bbox=norm_bbox,
            caption_confidence=caption_confidence,
        )

        # ── 섹션 헤더 + 인접 단락 (bbox 거리 + 임베딩 유사도, context_attacher.py) ──
        # table_bbox를 직접 넘김: eu_id의 페이지 내 순번은 dedup으로 doc.tables의
        # 스캔 순서와 어긋날 수 있어, eu_id 기반 재추정에 맡기면 안 됨.
        attach_context_paragraphs(eu, doc, table_bbox=bbox)

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


def split_oversized_units(eu_list: list) -> list:
    """
    512토큰(DEFAULT_TOKEN_LIMIT) 초과 EU를 table_splitter.split_eu()로 행 단위 분할.
    한도 이내 EU는 그대로 통과.
    """
    from evidence_chunker.split import split_eu

    result = []
    for eu in eu_list:
        result.extend(split_eu(eu).chunks)
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    from evidence_chunker.parser.docling import make_converter

    print(f"[6] Parsing: {PDF_PATH}")
    converter = make_converter()
    result = converter.convert(PDF_PATH)
    doc = result.document
    pdf_name = os.path.splitext(os.path.basename(PDF_PATH))[0]

    section("EvidenceUnit 구성")
    eu_list = build_evidence_units(doc)
    print(f"  총 {len(eu_list)}개 EvidenceUnit 생성\n")

    section("토큰 한도 분할 (table_splitter)")
    eu_list = split_oversized_units(eu_list)
    n_split = sum(1 for eu in eu_list if eu.is_split)
    print(f"  분할 후 총 {len(eu_list)}개 EvidenceUnit ({n_split}개는 분할 조각)\n")

    conf_tags = {"direct": "[DIRECT]", "inferred": "[INFER] ", "none": "[NONE]  "}
    for eu in eu_list:
        conf_tag = conf_tags[eu.caption_confidence]
        split_tag = f" (split {eu.split_index}/{eu.total_splits})" if eu.is_split else ""
        print(f"  {conf_tag} [{eu.eu_id}] p{eu.page_no}{split_tag}")
        print(f"    section_header  : {(eu.section_header or 'None')[:60]}")
        print(f"    caption_text    : {(eu.caption_text or 'None')[:60]}")
        print(f"    table_abstract  : {(eu.table_abstract or 'None')[:80]}")
        print(f"    table_html      : {len(eu.table_html) if eu.table_html else 0} chars")
        print(f"    flattened_rows  : {len(eu.flattened_rows)}개 문장")
        print(f"    context_before  : {len(eu.context_before)}개 단락")
        print(f"    context_after   : {len(eu.context_after)}개 단락")
        print(f"    footnote_text   : {(eu.footnote_text or 'None')[:40]}")
        print(f"    bbox (norm)     : ({eu.bbox[0]:.3f}, {eu.bbox[1]:.3f}, {eu.bbox[2]:.3f}, {eu.bbox[3]:.3f})")
        print(f"    text() length   : {len(eu.text)} chars")
        if eu.flattened_rows:
            print(f"    flattened 예시  : {eu.flattened_rows[0][:80]}")
        print()

    # 첫 EU의 flattened_rows 전체 출력
    section("flattened_rows 전체 (EU 0)")
    if eu_list:
        for s in eu_list[0].flattened_rows:
            print(f"  - {s}")

    # JSON 저장
    section("JSON 저장")
    out_data = []
    for eu in eu_list:
        out_data.append({
            "eu_id": eu.eu_id,
            "page_no": eu.page_no,
            "caption_confidence": eu.caption_confidence,
            "section_header": eu.section_header,
            "caption_text": eu.caption_text,
            "table_abstract": eu.table_abstract,
            "footnote_text": eu.footnote_text,
            "table_html_len": len(eu.table_html) if eu.table_html else 0,
            "flattened_rows": eu.flattened_rows,
            "context_before_count": len(eu.context_before),
            "context_after_count": len(eu.context_after),
            "context_before": eu.context_before,
            "context_after": eu.context_after,
            "bbox": list(eu.bbox),
            "text_length": len(eu.text),
            "is_split": eu.is_split,
            "split_index": eu.split_index,
            "total_splits": eu.total_splits,
        })

    out_path = os.path.join(OUTPUT_DIR, f"{pdf_name}_evidence_units.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"  저장: {out_path}")
    print("\n[DONE] 06_build_eu.py complete")


if __name__ == "__main__":
    main()
