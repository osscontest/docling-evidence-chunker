"""
STEP 1: Basic PDF parsing and full structure dump.
Goal: "What data can we pull from a Docling object?" - verify with your own eyes.

Usage:
    python scripts/01_basic_parse.py [path/to/file.pdf]
"""
import sys
import os
import json
import re

# UTF-8 output for Windows
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


def main():
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(__file__))
    from _converter import make_converter

    print(f"[1] Parsing: {PDF_PATH}")
    converter = make_converter()
    result = converter.convert(PDF_PATH)
    doc = result.document

    # -------------------------------------------------------------------------
    # 1. Top-level attribute inventory
    # -------------------------------------------------------------------------
    section("1. DoclingDocument top-level fields")
    attrs = [a for a in dir(doc) if not a.startswith('_')]
    data_attrs = []
    for a in attrs:
        v = getattr(doc, a, None)
        if v is None or callable(v):
            continue
        type_name = type(v).__name__
        length = ""
        if hasattr(v, '__len__'):
            length = f" (len={len(v)})"
        data_attrs.append(f"  {a:<25} {type_name}{length}")
    for line in data_attrs:
        print(line)

    # -------------------------------------------------------------------------
    # 2. Document-level counts
    # -------------------------------------------------------------------------
    section("2. Element counts")
    counts = {
        "tables":      len(doc.tables),
        "texts":       len(doc.texts),
        "pictures":    len(doc.pictures),
        "groups":      len(doc.groups) if hasattr(doc, 'groups') else 'N/A',
        "key_value_items": len(doc.key_value_items) if hasattr(doc, 'key_value_items') else 'N/A',
    }
    for k, v in counts.items():
        print(f"  {k:<25} {v}")

    # -------------------------------------------------------------------------
    # 3. Save full JSON dump (truncated for readability)
    # -------------------------------------------------------------------------
    section("3. Saving full JSON dump")
    pdf_name = os.path.splitext(os.path.basename(PDF_PATH))[0]
    json_path = os.path.join(OUTPUT_DIR, f"{pdf_name}_full_dump.json")
    full_json = doc.model_dump_json(indent=2)
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(full_json)
    print(f"  Saved: {json_path}")
    print(f"  Size:  {len(full_json):,} chars")

    # -------------------------------------------------------------------------
    # 4. Preview first table structure
    # -------------------------------------------------------------------------
    section("4. First table raw structure preview")
    if doc.tables:
        t = doc.tables[0]
        t_dict = t.model_dump()
        # Show only keys at depth 1-2 to avoid noise
        print("  Keys:", list(t_dict.keys()))
        for k, v in t_dict.items():
            if isinstance(v, (dict, list)):
                preview = str(v)[:120]
            else:
                preview = str(v)[:120]
            print(f"  [{k}]: {preview}")
    else:
        print("  No tables found in this document.")

    # -------------------------------------------------------------------------
    # 5. Preview first text element structure
    # -------------------------------------------------------------------------
    section("5. First text element raw structure preview")
    if doc.texts:
        t = doc.texts[0]
        t_dict = t.model_dump()
        print("  Keys:", list(t_dict.keys()))
        for k, v in t_dict.items():
            print(f"  [{k}]: {str(v)[:120]}")
    else:
        print("  No text elements found.")

    # -------------------------------------------------------------------------
    # 6. Page count
    # -------------------------------------------------------------------------
    section("6. Page info")
    if hasattr(doc, 'pages'):
        print(f"  Total pages: {len(doc.pages)}")
        for pg_key, pg_val in list(doc.pages.items())[:3]:
            pg_dict = pg_val.model_dump() if hasattr(pg_val, 'model_dump') else {}
            size = pg_dict.get('size', {})
            print(f"  Page {pg_key}: size={size}")
    else:
        print("  doc.pages not available")

    print("\n[DONE] 01_basic_parse.py complete")
    print(f"       Full dump -> {json_path}")


if __name__ == "__main__":
    main()
