"""
STEP 3: Text element field exploration.
Answers:
  - What label types exist? (paragraph, section_header, caption, ...)
  - Does bbox allow distance calculation to nearby tables?
  - How are section headers structured?

Usage:
    python scripts/03_explore_texts.py [path/to/file.pdf]
"""
import sys
import os
import json
from collections import Counter

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


def get_bbox(prov_list):
    if not prov_list:
        return None, None
    p = prov_list[0]
    return p.get('bbox', {}), p.get('page_no', None)


def bbox_vertical_center(bbox: dict) -> float | None:
    """Return vertical center of bbox (for distance calculation)."""
    t = bbox.get('t', bbox.get('y0', None))
    b = bbox.get('b', bbox.get('y1', None))
    if t is None or b is None:
        return None
    return (t + b) / 2.0


def main():
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(__file__))
    from _converter import make_converter

    print(f"[3] Parsing: {PDF_PATH}")
    converter = make_converter()
    result = converter.convert(PDF_PATH)
    doc = result.document

    pdf_name = os.path.splitext(os.path.basename(PDF_PATH))[0]

    # -------------------------------------------------------------------------
    # 1. Label inventory
    # -------------------------------------------------------------------------
    section(f"1. Label types in doc.texts ({len(doc.texts)} elements)")
    label_counter = Counter()
    label_examples = {}

    for text in doc.texts:
        t_dict = text.model_dump()
        label = t_dict.get('label', 'UNKNOWN')
        label_counter[label] += 1
        if label not in label_examples:
            label_examples[label] = {
                "text_preview": t_dict.get('text', '')[:80],
                "keys": list(t_dict.keys()),
            }

    for label, count in label_counter.most_common():
        ex = label_examples[label]["text_preview"]
        print(f"  {label:<30} count={count:>4}  ex: '{ex[:50]}'")

    # -------------------------------------------------------------------------
    # 2. Detailed per-element structure (first 2 of each label)
    # -------------------------------------------------------------------------
    section("2. Field structure per label (first occurrence)")
    seen_labels = set()
    for text in doc.texts:
        t_dict = text.model_dump()
        label = t_dict.get('label', 'UNKNOWN')
        if label in seen_labels:
            continue
        seen_labels.add(label)
        print(f"\n  --- {label} ---")
        for k, v in t_dict.items():
            print(f"    [{k}]: {str(v)[:100]}")

    # -------------------------------------------------------------------------
    # 3. Section header analysis (for EU context boundary detection)
    # -------------------------------------------------------------------------
    section("3. Section headers (candidate boundary markers)")
    headers = [t for t in doc.texts if t.model_dump().get('label') in
               ('section_header', 'title', 'subtitle', 'heading')]
    print(f"  Found {len(headers)} section header elements")
    for h in headers[:20]:
        h_dict = h.model_dump()
        bbox, page_no = get_bbox(h_dict.get('prov', []))
        level = h_dict.get('level', '?')  # heading level if available
        text_preview = h_dict.get('text', '')[:70]
        center_y = bbox_vertical_center(bbox) if bbox else None
        cy_str = f"{center_y:.1f}" if center_y else "N/A"
        print(f"  p{page_no} cy={cy_str:>7} lv={level} | '{text_preview}'")

    # -------------------------------------------------------------------------
    # 4. Caption elements in doc.texts
    # -------------------------------------------------------------------------
    section("4. Caption elements in doc.texts")
    captions = [t for t in doc.texts if 'caption' in t.model_dump().get('label', '').lower()]
    print(f"  Found {len(captions)} caption elements in doc.texts")
    for cap in captions[:10]:
        c_dict = cap.model_dump()
        bbox, page_no = get_bbox(c_dict.get('prov', []))
        print(f"  p{page_no} | label={c_dict.get('label')} | '{c_dict.get('text','')[:70]}'")

    # -------------------------------------------------------------------------
    # 5. Distance calculation demo: text elements near first table
    # -------------------------------------------------------------------------
    section("5. Distance demo: text elements near Table 0")
    if doc.tables:
        t0 = doc.tables[0].model_dump()
        t0_prov = t0.get('prov', [{}])[0]
        t0_bbox = t0_prov.get('bbox', {})
        t0_page = t0_prov.get('page_no', -1)
        t0_cy = bbox_vertical_center(t0_bbox)

        print(f"  Table 0 on page {t0_page}, center_y={t0_cy:.1f}")
        print(f"  Texts on same page sorted by distance to table:")

        same_page_texts = []
        for text in doc.texts:
            t_dict = text.model_dump()
            bbox, page_no = get_bbox(t_dict.get('prov', []))
            if page_no != t0_page or bbox is None:
                continue
            cy = bbox_vertical_center(bbox)
            if cy is None or t0_cy is None:
                continue
            dist = abs(cy - t0_cy)
            same_page_texts.append({
                "dist": dist,
                "label": t_dict.get('label', '?'),
                "text": t_dict.get('text', '')[:60],
                "center_y": cy,
            })

        same_page_texts.sort(key=lambda x: x["dist"])
        for item in same_page_texts[:10]:
            print(f"    dist={item['dist']:>7.1f} | {item['label']:<20} | '{item['text']}'")
    else:
        print("  No tables in document.")

    # -------------------------------------------------------------------------
    # 6. Elements without prov (edge case)
    # -------------------------------------------------------------------------
    section("6. Edge case: text elements without prov (no position)")
    no_prov = [t for t in doc.texts if not t.model_dump().get('prov')]
    print(f"  Count: {len(no_prov)}")
    for item in no_prov[:5]:
        d = item.model_dump()
        print(f"    label={d.get('label')} text='{d.get('text','')[:60]}'")

    # -------------------------------------------------------------------------
    # Save findings
    # -------------------------------------------------------------------------
    findings = {
        "total_texts": len(doc.texts),
        "label_counts": dict(label_counter),
        "label_list": list(label_counter.keys()),
        "caption_elements_in_texts": len(captions),
        "section_headers_count": len(headers),
        "no_prov_count": len(no_prov),
        "label_examples": {k: v["text_preview"] for k, v in label_examples.items()},
    }
    out_path = os.path.join(OUTPUT_DIR, f"{pdf_name}_text_findings.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2, ensure_ascii=False)
    print(f"\n  Findings saved: {out_path}")

    print("\n[DONE] 03_explore_texts.py complete")


if __name__ == "__main__":
    main()
