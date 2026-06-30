"""
STEP 5: 표 셀 내용 탐색 + export_to_html() 확인.
EU의 table_html 필드를 채우려면 이 구조를 먼저 이해해야 함.

Usage:
    python scripts/05_table_cells.py [path/to/file.pdf]
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


def main():
    from _converter import make_converter

    print(f"[5] Parsing: {PDF_PATH}")
    converter = make_converter()
    result = converter.convert(PDF_PATH)
    doc = result.document
    pdf_name = os.path.splitext(os.path.basename(PDF_PATH))[0]

    # -------------------------------------------------------------------------
    # 1. export_to_html() 확인
    # -------------------------------------------------------------------------
    section("1. table.export_to_html(doc) 확인")
    for i, table in enumerate(doc.tables):
        try:
            html = table.export_to_html(doc)
            print(f"\n  Table {i}: {len(html)} chars")
            print(f"  미리보기 (200자):")
            print(f"  {html[:200]}")
        except Exception as e:
            print(f"  Table {i}: export_to_html() 실패 - {e}")

    # -------------------------------------------------------------------------
    # 2. table.data.table_cells 구조
    # -------------------------------------------------------------------------
    section("2. table.data.table_cells 구조 (Table 0 기준)")
    if doc.tables:
        t0 = doc.tables[0]
        t0_dict = t0.model_dump()
        data = t0_dict.get("data", {})
        cells = data.get("table_cells", [])

        print(f"  총 셀 수: {len(cells)}")
        print(f"  data 키: {list(data.keys())}")

        if cells:
            print(f"\n  첫 번째 셀 구조:")
            for k, v in cells[0].items():
                print(f"    [{k}]: {str(v)[:80]}")

            print(f"\n  모든 셀 텍스트 (앞 10개):")
            for cell in cells[:10]:
                row = cell.get("start_row_offset_idx", "?")
                col = cell.get("start_col_offset_idx", "?")
                text = cell.get("text", "")
                is_header = cell.get("column_header", False) or cell.get("row_header", False)
                tag = "[H]" if is_header else "   "
                print(f"    {tag} ({row},{col}): '{text}'")

        # grid dimensions
        grid = data.get("grid", [])
        if grid:
            print(f"\n  grid 차원: {len(grid)} rows x {len(grid[0]) if grid else 0} cols")

    # -------------------------------------------------------------------------
    # 3. 각 표의 HTML 저장
    # -------------------------------------------------------------------------
    section("3. 각 표 HTML 저장")
    html_outputs = []
    for i, table in enumerate(doc.tables):
        try:
            html = table.export_to_html(doc)
            t_dict = table.model_dump()
            prov = t_dict.get("prov", [{}])[0]
            cap_refs = t_dict.get("captions", [])
            caption = ""
            if cap_refs:
                cref = cap_refs[0].get("cref", "") if isinstance(cap_refs[0], dict) else getattr(cap_refs[0], "cref", "")
                parts = cref.strip("#/").split("/")
                obj = doc
                try:
                    for p in parts:
                        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
                    cap_dict = obj.model_dump() if hasattr(obj, "model_dump") else {}
                    caption = cap_dict.get("text", "")
                except Exception:
                    pass

            entry = {
                "table_idx": i,
                "page_no": prov.get("page_no"),
                "caption": caption,
                "html_length": len(html),
                "html_preview": html[:300],
            }
            html_outputs.append(entry)
            print(f"  Table {i} (p{prov.get('page_no')}): {len(html)} chars | cap='{caption[:50]}'")
        except Exception as e:
            print(f"  Table {i}: ERROR - {e}")

    out_path = os.path.join(OUTPUT_DIR, f"{pdf_name}_table_html.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(html_outputs, f, indent=2, ensure_ascii=False)
    print(f"\n  저장: {out_path}")

    # -------------------------------------------------------------------------
    # 4. 헤더/바디 행 구분
    # -------------------------------------------------------------------------
    section("4. 헤더/바디 셀 구분 (Table 0)")
    if doc.tables:
        t0_dict = doc.tables[0].model_dump()
        cells = t0_dict.get("data", {}).get("table_cells", [])
        header_cells = [c for c in cells if c.get("column_header") or c.get("row_header")]
        body_cells   = [c for c in cells if not c.get("column_header") and not c.get("row_header")]
        print(f"  헤더 셀: {len(header_cells)}개")
        print(f"  바디 셀:  {len(body_cells)}개")
        print(f"\n  헤더 셀 텍스트: {[c.get('text','') for c in header_cells[:8]]}")

    print("\n[DONE] 05_table_cells.py complete")


if __name__ == "__main__":
    main()
