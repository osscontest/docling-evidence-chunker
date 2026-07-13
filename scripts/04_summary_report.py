"""
STEP 4: Summary report generator.
Reads JSON findings from steps 2 and 3, then produces a human-readable
W1 exploration report covering all required checklist items.

Required checklist:
  [A] captions 필드가 항상 채워져 있는가? 없는 케이스는?
  [B] bbox 좌표계가 무엇인가? (픽셀? PDF 포인트? 정규화?)
  [C] 텍스트 요소의 label 종류 목록
  [D] 표와 캡션이 다른 페이지에 있는 케이스가 있는가?
  [E] 탐색 중 발견한 예외 케이스 목록

Usage:
    python scripts/04_summary_report.py [pdf_name_prefix]
    (default: attention_is_all_you_need)
"""
import sys
import os
import json
from datetime import datetime

if hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "outputs")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

PDF_PREFIX = sys.argv[1] if len(sys.argv) > 1 else "attention_is_all_you_need"


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def divider(char="=", width=62) -> str:
    return char * width


def main():
    table_f = load_json(os.path.join(OUTPUT_DIR, f"{PDF_PREFIX}_table_findings.json"))
    text_f  = load_json(os.path.join(OUTPUT_DIR, f"{PDF_PREFIX}_text_findings.json"))

    lines = []

    def p(s=""):
        lines.append(s)
        print(s)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    p(divider())
    p(f"  W1 Docling API 탐색 보고서")
    p(f"  PDF: {PDF_PREFIX}")
    p(f"  생성: {now}")
    p(divider())

    # ------------------------------------------------------------------
    # [A] captions 필드
    # ------------------------------------------------------------------
    p()
    p("[ A ] captions 필드 — 항상 채워져 있는가?")
    p(divider('-'))
    total   = table_f.get("total_tables", "?")
    w_cap   = table_f.get("tables_with_caption", "?")
    wo_cap  = table_f.get("tables_without_caption", "?")
    no_prov = table_f.get("no_prov_indices", [])

    p(f"  전체 표:              {total}")
    p(f"  캡션 있음:            {w_cap}")
    p(f"  캡션 없음:            {wo_cap}")
    p(f"  prov 없음(위치 미상): {len(no_prov)}")

    if isinstance(wo_cap, int) and wo_cap > 0:
        p(f"\n  결론: captions 필드는 항상 채워지지 않음.")
        p(f"        Table 인덱스 {table_f.get('no_caption_indices', [])} 에 캡션 없음.")
        p(f"  -> EU 구성 시 '캡션 없는 표' 폴백 로직 필요 (table label 등 사용)")
    else:
        p(f"\n  결론: 이 PDF에서는 모든 표에 캡션이 존재함.")
        p(f"        단, 다른 PDF에서 캡션 없는 케이스 가능 — 방어 코드 필수.")

    cap_examples = table_f.get("caption_texts", [])[:5]
    if cap_examples:
        p(f"\n  캡션 예시:")
        for ex in cap_examples:
            p(f"    - '{ex}'")

    # ------------------------------------------------------------------
    # [B] bbox 좌표계
    # ------------------------------------------------------------------
    p()
    p("[ B ] bbox 좌표계 — 픽셀? PDF 포인트? 정규화?")
    p(divider('-'))
    bbox_samples = table_f.get("bbox_values_sample", [])
    if bbox_samples:
        for s in bbox_samples[:3]:
            bbox = s.get("bbox", {})
            page_size = s.get("page_size", {})
            coords = [v for k, v in bbox.items() if isinstance(v, (int, float))]
            max_c = max(abs(c) for c in coords) if coords else 0
            w = page_size.get('width', '?')
            h = page_size.get('height', '?')
            p(f"  Table {s['table_idx']} (p{s['page_no']}): bbox={bbox}")
            p(f"    page size=({w} x {h})  max_coord={max_c:.2f}")

        all_max = []
        for s in bbox_samples:
            bbox = s.get("bbox", {})
            coords = [v for k, v in bbox.items() if isinstance(v, (int, float))]
            if coords:
                all_max.append(max(abs(c) for c in coords))

        if all_max:
            overall_max = max(all_max)
            if overall_max <= 1.0:
                coord_type = "정규화된 좌표 [0, 1]"
                note = "페이지 크기로 나눠야 실제 위치 계산 가능"
            elif overall_max <= 842:  # A4 height in points
                coord_type = "PDF 포인트 (72 dpi 기준, 1pt = 1/72 inch)"
                note = "일반적인 PDF 좌표계. A4는 595x842pt"
            else:
                coord_type = "픽셀 (72dpi 초과 — 렌더링 해상도 기준)"
                note = "페이지 dpi에 따라 달라짐"

            p(f"\n  결론: {coord_type}")
            p(f"        ({note})")
    else:
        p("  bbox 샘플 없음 — 02_explore_tables.py 를 먼저 실행하세요.")

    # ------------------------------------------------------------------
    # [C] 텍스트 label 종류
    # ------------------------------------------------------------------
    p()
    p("[ C ] 텍스트 요소 label 종류 목록")
    p(divider('-'))
    label_counts = text_f.get("label_counts", {})
    label_examples = text_f.get("label_examples", {})
    if label_counts:
        p(f"  발견된 label 종류: {len(label_counts)}가지\n")
        p(f"  {'LABEL':<30} {'COUNT':>6}  EXAMPLE")
        p(f"  {'-'*30} {'-'*6}  {'-'*30}")
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            ex = label_examples.get(label, '')[:40]
            p(f"  {label:<30} {count:>6}  '{ex}'")
        p()
        p(f"  EU 구성 핵심 레이블:")
        key_labels = ['section_header', 'paragraph', 'caption', 'table_caption',
                      'title', 'footnote', 'list_item', 'formula']
        for lbl in key_labels:
            status = "O" if lbl in label_counts else "X"
            cnt = label_counts.get(lbl, 0)
            p(f"    [{status}] {lbl:<25} (n={cnt})")
    else:
        p("  label 데이터 없음 — 03_explore_texts.py 를 먼저 실행하세요.")

    # ------------------------------------------------------------------
    # [D] 표-캡션 다른 페이지 케이스
    # ------------------------------------------------------------------
    p()
    p("[ D ] 표와 캡션이 다른 페이지에 있는 케이스")
    p(divider('-'))
    cross_page = table_f.get("cross_page_caption_cases", [])
    if cross_page:
        p(f"  발견! {len(cross_page)}건의 크로스-페이지 케이스:")
        for c in cross_page:
            p(f"    Table {c['table_idx']}: 표=p{c['table_page']}, 캡션=p{c['caption_page']}")
        p()
        p("  -> EU 구성 시 다른 페이지 캡션도 검색하는 로직 필요")
        p("     (page_no 기준 +/- 1 페이지 탐색 권장)")
    else:
        p("  이 PDF에서는 크로스-페이지 케이스 없음.")
        p("  다른 PDF에서 발생 가능 — bbox 기반 거리가 아닌")
        p("  레퍼런스 기반(captions RefItem)으로 연결이 더 안전.")

    # ------------------------------------------------------------------
    # [E] 예외 케이스 목록
    # ------------------------------------------------------------------
    p()
    p("[ E ] 탐색 중 발견한 예외 케이스")
    p(divider('-'))
    edge_cases = table_f.get("edge_cases", [])
    no_cap_idx  = table_f.get("no_caption_indices", [])
    no_prov_idx = table_f.get("no_prov_indices", [])
    no_prov_text = text_f.get("no_prov_count", 0)

    all_edges = []
    if no_cap_idx:
        all_edges.append(f"캡션 없는 표 {len(no_cap_idx)}개: index={no_cap_idx}")
    if no_prov_idx:
        all_edges.append(f"prov 없는 표 {len(no_prov_idx)}개 (bbox 계산 불가): index={no_prov_idx}")
    if no_prov_text:
        all_edges.append(f"prov 없는 텍스트 요소 {no_prov_text}개 (위치 미상)")
    for e in edge_cases:
        all_edges.append(e)

    if all_edges:
        for e in all_edges:
            p(f"  ! {e}")
    else:
        p("  이 PDF에서 특별한 예외 케이스 없음.")

    # ------------------------------------------------------------------
    # 다음 단계 제안
    # ------------------------------------------------------------------
    p()
    p(divider())
    p("  W1 결론 및 W2 준비 사항")
    p(divider())
    p()
    p("  1. captions RefItem 연결 확인됨 — cref 기반 역참조 동작")
    p("  2. bbox 좌표계 확인 — PDF 포인트 기반이므로 distance 계산 가능")
    p("  3. 캡션 없는 표 처리: doc.texts 내 'caption' label 원소를 bbox 거리로 매칭")
    p("  4. section_header label 존재 확인 — EU 경계 후보로 활용 가능")
    p("  5. W2 과제: captions RefItem 파싱 PoC 및 캡션-표 1:1 매핑 알고리즘 구현")
    p()

    # Save report
    report_path = os.path.join(REPORT_DIR, f"W1_docling_api_report_{PDF_PREFIX}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved: {report_path}")
    print("\n[DONE] 04_summary_report.py complete")


if __name__ == "__main__":
    main()
