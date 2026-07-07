"""
STEP 6: EvidenceUnit 실제 구성.

포함 기능:
  - caption_text    : _caption_mapper.map_table_caption() (RefItem 직접 연결 + bbox fallback)
  - section_header  : 표 위쪽 가장 가까운 섹션 헤더
  - context_before/after: 같은 페이지 300pt 이내 단락
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

CONTEXT_WINDOW_PT = 300.0  # 표 위아래 300 PDF 포인트 이내를 컨텍스트로 수집


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


def bbox_center_y(bbox: dict) -> float:
    return (bbox.get("t", 0.0) + bbox.get("b", 0.0)) / 2.0


# ---------------------------------------------------------------------------
# 섹션 헤더 탐색
# ---------------------------------------------------------------------------

def find_section_header(doc, table_page: int, table_top_y: float) -> str | None:
    """
    표 위쪽(읽기 순서 기준 이전)에서 가장 가까운 section_header.
    BOTTOMLEFT: 시각적으로 표 위 = y 값이 table_top_y보다 큰 헤더.
    같은 페이지에 없으면 이전 페이지 중 가장 마지막 헤더.
    """
    same_above: list[tuple[float, str]] = []
    prev_pages: list[tuple[int, float, str]] = []

    for item in doc.texts:
        d = item.model_dump()
        if d.get("label") != "section_header":
            continue
        pg, bbox = get_prov(d)
        if pg == -1:
            continue
        cy = bbox_center_y(bbox)
        text = d.get("text", "")

        if pg == table_page and cy > table_top_y:
            same_above.append((cy - table_top_y, text))
        elif pg < table_page:
            prev_pages.append((pg, cy, text))

    if same_above:
        same_above.sort(key=lambda x: x[0])
        return same_above[0][1]

    if prev_pages:
        prev_pages.sort(key=lambda x: (x[0], x[1]))
        return prev_pages[-1][2]

    return None


# ---------------------------------------------------------------------------
# 컨텍스트 단락 탐색
# ---------------------------------------------------------------------------

def find_context(
    doc, table_page: int, table_top_y: float, table_bot_y: float
) -> tuple[list[str], list[str]]:
    """
    같은 페이지 CONTEXT_WINDOW_PT 범위 내 text/list_item 단락 수집.
    before: 표 위쪽 (cy > table_top_y)
    after:  표 아래쪽 (cy < table_bot_y)
    """
    before: list[tuple[float, str]] = []
    after:  list[tuple[float, str]] = []

    for item in doc.texts:
        d = item.model_dump()
        if d.get("label") not in ("text", "list_item", "paragraph"):
            continue
        pg, bbox = get_prov(d)
        if pg != table_page:
            continue
        cy = bbox_center_y(bbox)
        content = d.get("text", "").strip()
        if not content:
            continue

        if cy > table_top_y:
            dist = cy - table_top_y
            if dist <= CONTEXT_WINDOW_PT:
                before.append((dist, content))
        elif cy < table_bot_y:
            dist = table_bot_y - cy
            if dist <= CONTEXT_WINDOW_PT:
                after.append((dist, content))

    before.sort(key=lambda x: x[0])
    after.sort(key=lambda x: x[0])
    return [t for _, t in before], [t for _, t in after]


# ---------------------------------------------------------------------------
# EU 빌더
# ---------------------------------------------------------------------------

def build_evidence_units(doc) -> list:
    from interfaces import EvidenceUnit
    from bbox_utils import normalize_bbox
    from _caption_mapper import map_table_caption
    from _table_utils import (
        build_col_header_map,
        build_row_header_map,
        flatten_to_sentences,
        build_table_abstract,
    )

    page_sizes: dict[int, dict] = {}
    if hasattr(doc, "pages"):
        for pg_key, pg_val in doc.pages.items():
            pg_dict = pg_val.model_dump() if hasattr(pg_val, "model_dump") else {}
            page_sizes[int(pg_key)] = pg_dict.get("size", {})

    eu_list: list[EvidenceUnit] = []
    page_counters: dict[int, int] = {}

    for table_index, table in enumerate(doc.tables):
        t_dict = table.model_dump()
        pg, bbox = get_prov(t_dict)
        if pg == -1:
            continue

        idx = page_counters.get(pg, 0)
        page_counters[pg] = idx + 1
        eu_id = f"eu-p{pg}-{idx}"

        table_top_y = bbox.get("t", 0.0)
        table_bot_y = bbox.get("b", 0.0)

        # ── 캡션 (RefItem 직접 연결 + bbox fallback, _caption_mapper.py) ──
        cap_mapping = map_table_caption(doc, table, table_index)
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

        # ── 섹션 헤더 ───────────────────────────────────────────────
        section_hdr = find_section_header(doc, pg, table_top_y)

        # ── 컨텍스트 단락 ────────────────────────────────────────────
        ctx_before, ctx_after = find_context(doc, pg, table_top_y, table_bot_y)

        # ── 표 HTML ─────────────────────────────────────────────────
        try:
            table_html = table.export_to_html(doc) or None
        except Exception:
            table_html = None

        # ── Row Flattening + 다단 헤더 처리 ─────────────────────────
        data = t_dict.get("data", {})
        cells = data.get("table_cells", [])
        num_rows = data.get("num_rows", 0)
        num_cols = data.get("num_cols", 0)

        flattened = flatten_to_sentences(cells, num_rows, num_cols, footnote_text)

        # ── Table Abstract ───────────────────────────────────────────
        col_map = build_col_header_map(cells, num_cols)
        abstract = build_table_abstract(caption_text, col_map, num_rows, section_hdr)

        # ── bbox 정규화 ──────────────────────────────────────────────
        ps = page_sizes.get(pg, {})
        norm_bbox = normalize_bbox(bbox, ps.get("width", 1.0), ps.get("height", 1.0))

        eu = EvidenceUnit(
            eu_id=eu_id,
            page_no=pg,
            section_header=section_hdr,
            caption_text=caption_text,
            table_html=table_html,
            footnote_text=footnote_text,
            context_before=ctx_before,
            context_after=ctx_after,
            flattened_rows=flattened,
            table_abstract=abstract,
            bbox=norm_bbox,
            caption_confidence=caption_confidence,
        )
        eu_list.append(eu)

    return eu_list


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    from _converter import make_converter

    print(f"[6] Parsing: {PDF_PATH}")
    converter = make_converter()
    result = converter.convert(PDF_PATH)
    doc = result.document
    pdf_name = os.path.splitext(os.path.basename(PDF_PATH))[0]

    section("EvidenceUnit 구성")
    eu_list = build_evidence_units(doc)
    print(f"  총 {len(eu_list)}개 EvidenceUnit 생성\n")

    conf_tags = {"direct": "[DIRECT]", "inferred": "[INFER] ", "none": "[NONE]  "}
    for eu in eu_list:
        conf_tag = conf_tags[eu.caption_confidence]
        print(f"  {conf_tag} [{eu.eu_id}] p{eu.page_no}")
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
        })

    out_path = os.path.join(OUTPUT_DIR, f"{pdf_name}_evidence_units.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"  저장: {out_path}")
    print("\n[DONE] 06_build_eu.py complete")


if __name__ == "__main__":
    main()
