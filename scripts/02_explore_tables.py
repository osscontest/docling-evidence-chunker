"""
STEP 2: Table field exploration.
Answers:
  - Are captions always populated? What are missing cases?
  - What coordinate system does bbox use? (pixels / PDF points / normalized?)
  - Do tables and captions ever span different pages?

Usage:
    python scripts/02_explore_tables.py [path/to/file.pdf]
"""
import sys
import os
import json

if hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PDF_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "data", "pdfs", "attention_is_all_you_need.pdf"
)


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def bbox_summary(bbox) -> str:
    if bbox is None:
        return "None"
    b = bbox if isinstance(bbox, dict) else (bbox.model_dump() if hasattr(bbox, 'model_dump') else vars(bbox))
    return f"l={b.get('l',b.get('x0','?')):.2f} t={b.get('t',b.get('y0','?')):.2f} r={b.get('r',b.get('x1','?')):.2f} b={b.get('b',b.get('y1','?')):.2f}"


def resolve_ref(doc, ref_item):
    """RefItem -> actual text element."""
    try:
        cref = ref_item.cref if hasattr(ref_item, 'cref') else ref_item.get('cref', '')
        # cref looks like "#/texts/3"
        parts = cref.strip('#/').split('/')
        obj = doc
        for p in parts:
            if p.isdigit():
                obj = obj[int(p)]
            else:
                obj = getattr(obj, p, None) or obj[p]
        return obj
    except Exception:
        return None


def main():
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(__file__))
    from _converter import make_converter

    print(f"[2] Parsing: {PDF_PATH}")
    converter = make_converter()
    result = converter.convert(PDF_PATH)
    doc = result.document

    pdf_name = os.path.splitext(os.path.basename(PDF_PATH))[0]

    # -------------------------------------------------------------------------
    # Collect page sizes for coordinate system detection
    # -------------------------------------------------------------------------
    page_sizes = {}
    if hasattr(doc, 'pages'):
        for pg_key, pg_val in doc.pages.items():
            pg_dict = pg_val.model_dump() if hasattr(pg_val, 'model_dump') else {}
            size = pg_dict.get('size', {})
            page_sizes[int(pg_key)] = size

    section("1. Page sizes (for bbox coordinate system detection)")
    for pg, size in list(page_sizes.items())[:5]:
        print(f"  Page {pg}: width={size.get('width','?')} height={size.get('height','?')}")

    # -------------------------------------------------------------------------
    # Table-by-table analysis
    # -------------------------------------------------------------------------
    section(f"2. Per-table analysis ({len(doc.tables)} tables found)")

    findings = {
        "total_tables": len(doc.tables),
        "tables_with_caption": 0,
        "tables_without_caption": 0,
        "tables_with_footnotes": 0,
        "cross_page_caption_cases": [],
        "bbox_values_sample": [],
        "caption_texts": [],
        "no_caption_indices": [],
        "no_prov_indices": [],
        "edge_cases": [],
    }

    for i, table in enumerate(doc.tables):
        t_dict = table.model_dump()

        # --- prov (provenance = position info) ---
        prov_list = t_dict.get('prov', [])
        if not prov_list:
            findings["no_prov_indices"].append(i)
            print(f"  Table {i:>3}: NO PROV - cannot get bbox/page")
            continue

        prov = prov_list[0]
        page_no = prov.get('page_no', '?')
        bbox = prov.get('bbox', {})
        bbox_str = bbox_summary(bbox)

        # Collect bbox sample for coordinate system analysis
        if len(findings["bbox_values_sample"]) < 5:
            findings["bbox_values_sample"].append({
                "table_idx": i,
                "page_no": page_no,
                "bbox": bbox,
                "page_size": page_sizes.get(page_no, {}),
            })

        # --- captions ---
        captions_raw = t_dict.get('captions', [])
        has_caption = len(captions_raw) > 0
        caption_texts = []
        caption_page = None

        if has_caption:
            findings["tables_with_caption"] += 1
            for cap_ref in captions_raw:
                resolved = resolve_ref(doc, cap_ref if isinstance(cap_ref, object) else cap_ref)
                if resolved is not None:
                    cap_dict = resolved.model_dump() if hasattr(resolved, 'model_dump') else {}
                    cap_text = cap_dict.get('text', '')[:80]
                    cap_prov = cap_dict.get('prov', [{}])
                    if cap_prov:
                        caption_page = cap_prov[0].get('page_no', None)
                    caption_texts.append(cap_text)
                    if caption_texts:
                        findings["caption_texts"].append(caption_texts[0][:60])
        else:
            findings["tables_without_caption"] += 1
            findings["no_caption_indices"].append(i)

        # --- cross-page check ---
        if caption_page is not None and caption_page != page_no:
            case = {"table_idx": i, "table_page": page_no, "caption_page": caption_page}
            findings["cross_page_caption_cases"].append(case)
            findings["edge_cases"].append(f"Table {i}: table on p{page_no}, caption on p{caption_page}")

        # --- footnotes ---
        footnotes = t_dict.get('footnotes', [])
        if footnotes:
            findings["tables_with_footnotes"] += 1

        cap_display = caption_texts[0][:50] if caption_texts else "*** NONE ***"
        cross = f" [CROSS-PAGE: cap_p={caption_page}]" if caption_page and caption_page != page_no else ""
        print(f"  Table {i:>3}: p{page_no} | bbox({bbox_str}) | cap='{cap_display}'{cross}")

    # -------------------------------------------------------------------------
    # Coordinate system analysis
    # -------------------------------------------------------------------------
    section("3. Coordinate system analysis")
    print("  Checking if bbox coords exceed 1.0 (=> not normalized)")
    for sample in findings["bbox_values_sample"]:
        bbox = sample["bbox"]
        page_size = sample["page_size"]
        coords = [bbox.get('l', bbox.get('x0', 0)),
                  bbox.get('t', bbox.get('y0', 0)),
                  bbox.get('r', bbox.get('x1', 0)),
                  bbox.get('b', bbox.get('y1', 0))]
        max_coord = max(abs(c) for c in coords if isinstance(c, (int, float)))
        w = page_size.get('width', '?')
        h = page_size.get('height', '?')
        coord_type = "PDF points (72 dpi)" if max_coord > 1.5 else "normalized [0,1]"
        print(f"  Table {sample['table_idx']} p{sample['page_no']}: max_coord={max_coord:.2f} | page({w}x{h}) => {coord_type}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    section("4. Summary findings")
    print(f"  Total tables:             {findings['total_tables']}")
    print(f"  With caption:             {findings['tables_with_caption']}")
    print(f"  Without caption:          {findings['tables_without_caption']}")
    print(f"  With footnotes:           {findings['tables_with_footnotes']}")
    print(f"  No prov (no position):    {len(findings['no_prov_indices'])}")
    print(f"  Cross-page (table/cap):   {len(findings['cross_page_caption_cases'])}")

    if findings["no_caption_indices"]:
        print(f"\n  Tables WITHOUT caption: {findings['no_caption_indices']}")
    if findings["cross_page_caption_cases"]:
        print(f"\n  Cross-page cases:")
        for c in findings["cross_page_caption_cases"]:
            print(f"    {c}")
    if findings["edge_cases"]:
        print(f"\n  Edge cases detected:")
        for e in findings["edge_cases"]:
            print(f"    - {e}")

    # Save findings JSON
    out_path = os.path.join(OUTPUT_DIR, f"{pdf_name}_table_findings.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Findings saved: {out_path}")

    print("\n[DONE] 02_explore_tables.py complete")


if __name__ == "__main__":
    main()
