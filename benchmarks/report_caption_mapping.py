"""
report_caption_mapping.py

실제 PDF에서 캡션 ↔ 표 매핑(caption.py)이 어떻게 붙는지 리포트.

확인 항목:
  - 매핑률과 confidence 분포(direct / inferred / none)
  - 복수 캡션(multi_caption), 다른 페이지 캡션(cross_page)
  - 같은 캡션을 두 표가 물고 있는 충돌(collision)

caption.py의 fallback 경로(캡션 없는 표의 bbox 거리 매칭, 인접 페이지 탐색,
복수 캡션 병합)는 실제 PDF에서 잘 발동하지 않는 분기까지 포함하므로,
합성 데이터로 하는 검증은 tests/test_caption_exceptions.py가 담당한다.

Usage:
    python benchmarks/report_caption_mapping.py [path/to/file.pdf]
    python benchmarks/report_caption_mapping.py --all   # data/pdfs 전체 순회
"""
import sys
import os
import json
import glob

if hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PDF_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "pdfs")


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def run_on_pdf(converter, pdf_path: str) -> dict:
    from evidence_chunker.caption import map_all_captions, validate_mapping
    from evidence_chunker.parser.docling import DoclingParser

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"\n[PARSE] {pdf_path}")
    result = converter.convert(pdf_path)
    doc = result.document

    # caption.py는 DoclingDocument가 아니라 ParsedDoc만 받는다 — 파서 경계를
    # 거쳐야 하며, 좌표계도 여기서 TOPLEFT로 정규화된다.
    parsed = DoclingParser().from_doc(doc)
    mappings = map_all_captions(parsed)
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
    from evidence_chunker.parser.docling import make_converter

    if "--all" in sys.argv:
        pdf_paths = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    else:
        arg_paths = [a for a in sys.argv[1:] if not a.startswith("--")]
        pdf_paths = [arg_paths[0]] if arg_paths else [
            os.path.join(PDF_DIR, "attention_is_all_you_need.pdf")
        ]

    converter = make_converter()

    section("캡션 ↔ 표 매핑 리포트")
    all_stats = [run_on_pdf(converter, p) for p in pdf_paths]

    if len(all_stats) > 1:
        section("전체 요약")
        total = sum(s["total_tables"] for s in all_stats)
        mapped = sum(s["mapped"] for s in all_stats)
        if total:
            print(f"  전체 표: {total}개, 매핑 성공: {mapped}개 ({mapped/total*100:.1f}%)")
        else:
            print("  표 없음")

    print("\n[DONE] 캡션 매핑 리포트 완료")


if __name__ == "__main__":
    main()
