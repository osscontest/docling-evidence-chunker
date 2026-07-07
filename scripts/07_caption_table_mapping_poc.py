"""
STEP 7 (W2 PoC): 캡션 ↔ 표 매핑 알고리즘 검증.

담당: 팀원 1 (캡션↔표 연결, EU 핵심 anchor 로직)

목적:
  - Docling TableItem.captions RefItem 파싱 -> 캡션 텍스트 역참조
  - 표-캡션 1:1 매핑 알고리즘 프로토타입 (_caption_mapper.py)
  - 매핑률, 복수 캡션, cross-page, 충돌(collision) 케이스 검증/리포트

W3에서 예정된 작업 (여기서는 감지만 하고 실제 처리는 하지 않음):
  - 캡션 없는 표 fallback (bbox 거리 매칭)
  - 다음 페이지 캡션 케이스 연결
  - 복수 캡션 병합 처리

Usage:
    python scripts/07_caption_table_mapping_poc.py [path/to/file.pdf]
    python scripts/07_caption_table_mapping_poc.py --all   # data/pdfs 전체 순회
"""
import sys
import os
import json
import glob

if hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(__file__))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PDF_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "pdfs")


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def run_on_pdf(converter, pdf_path: str) -> dict:
    from _caption_mapper import map_all_captions, validate_mapping

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"\n[7] Parsing: {pdf_path}")
    result = converter.convert(pdf_path)
    doc = result.document

    mappings = map_all_captions(doc)
    stats = validate_mapping(mappings)

    section(f"{pdf_name} — 표 {stats['total_tables']}개 매핑 결과")
    tags = {"direct": "[DIRECT]", "inferred": "[INFER] ", "none": "[NONE]  "}
    for m in mappings:
        tag = tags[m.confidence]
        flags = []
        if m.multi_caption:
            flags.append("MULTI-CAPTION")
        if m.cross_page:
            flags.append("CROSS-PAGE")
        flag_str = f" ({', '.join(flags)})" if flags else ""
        cap_display = (m.caption_text or "*** 캡션 없음 ***")[:70]
        print(f"  {tag} table[{m.table_index}] p{m.page_no}: {cap_display}{flag_str}")

    print(f"\n  매핑률: {stats['mapped']}/{stats['total_tables']} ({stats['mapping_rate']*100:.1f}%)")
    bc = stats["by_confidence"]
    print(f"  confidence 분포: direct={bc['direct']}, inferred={bc['inferred']}, none={bc['none']}")
    if stats["collisions"]:
        print(f"  [WARN] 캡션 충돌 감지: {stats['collisions']}")
    if stats["multi_caption_tables"]:
        print(f"  [INFO] 복수 캡션 표: {stats['multi_caption_tables']}")
    if stats["cross_page_tables"]:
        print(f"  [INFO] cross-page 캡션 표: {stats['cross_page_tables']}")

    out = {
        "pdf": pdf_name,
        "stats": stats,
        "mappings": [
            {
                "table_index": m.table_index,
                "page_no": m.page_no,
                "caption_text": m.caption_text,
                "caption_ref": m.caption_ref,
                "confidence": m.confidence,
                "multi_caption": m.multi_caption,
                "cross_page": m.cross_page,
            }
            for m in mappings
        ],
    }
    out_path = os.path.join(OUTPUT_DIR, f"{pdf_name}_caption_mapping_poc.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  저장: {out_path}")
    return stats


def main():
    from _converter import make_converter

    if "--all" in sys.argv:
        pdf_paths = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    else:
        arg_paths = [a for a in sys.argv[1:] if not a.startswith("--")]
        pdf_paths = [arg_paths[0]] if arg_paths else [
            os.path.join(PDF_DIR, "attention_is_all_you_need.pdf")
        ]

    converter = make_converter()

    section("캡션 ↔ 표 매핑 PoC (W2, 팀원 1)")
    all_stats = [run_on_pdf(converter, p) for p in pdf_paths]

    if len(all_stats) > 1:
        section("전체 요약")
        total = sum(s["total_tables"] for s in all_stats)
        mapped = sum(s["mapped"] for s in all_stats)
        if total:
            print(f"  전체 표: {total}개, 매핑 성공: {mapped}개 ({mapped/total*100:.1f}%)")
        else:
            print("  표 없음")

    print("\n[DONE] 07_caption_table_mapping_poc.py complete")


if __name__ == "__main__":
    main()
