"""
context_attacher.py

인접 단락 탐지 모듈.

attach_context_paragraphs(eu, doc) 가 메인 진입점.

* bbox 거리 기반 단락 수집

TODO: 코사인 유사도 필터 추가
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from interfaces import EvidenceUnit

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

CONTEXT_WINDOW_PT = 300.0   # 표 위아래 300 PDF 포인트 이내
SKIP_LABELS = {"page_header", "page_footer", "formula"}
CONTEXT_LABELS = {"text", "list_item", "paragraph"}


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _get_prov(item_dict: dict) -> tuple[int, dict]:
    """model_dump() dict → (page_no, bbox dict)."""
    prov_list = item_dict.get("prov", [])
    if not prov_list:
        return -1, {}
    p = prov_list[0]
    return p.get("page_no", -1), p.get("bbox", {})


def _center_y(bbox: dict) -> float:
    return (bbox.get("t", 0.0) + bbox.get("b", 0.0)) / 2.0


# ---------------------------------------------------------------------------
# 섹션 헤더 탐색
# ---------------------------------------------------------------------------

def find_section_header(doc, table_page: int, table_top_y: float) -> str | None:
    """
    표 위쪽에서 가장 가까운 section_header 반환.

    BOTTOMLEFT 좌표계: 시각적으로 표 위 = cy > table_top_y.
    같은 페이지에 없으면 이전 페이지 중 마지막 헤더.
    """
    same_above: list[tuple[float, str]] = []
    prev_pages: list[tuple[int, float, str]] = []

    for item in doc.texts:
        d = item.model_dump()
        if d.get("label") != "section_header":
            continue
        pg, bbox = _get_prov(d)
        if pg == -1:
            continue
        cy = _center_y(bbox)
        text = d.get("text", "").strip()
        if not text:
            continue

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
# bbox 거리 기반 단락 수집
# ---------------------------------------------------------------------------

def _collect_by_bbox(
    doc,
    table_page: int,
    table_top_y: float,
    table_bot_y: float,
    bbox_threshold: float,
) -> tuple[list[tuple[float, str]], list[tuple[float, str]]]:
    """
    같은 페이지에서 bbox 거리 기준 단락 수집.

    Returns:
        before: [(거리, 텍스트), ...] — 표 위쪽, 거리 오름차순
        after:  [(거리, 텍스트), ...] — 표 아래쪽, 거리 오름차순
    """
    before: list[tuple[float, str]] = []
    after: list[tuple[float, str]] = []

    for item in doc.texts:
        d = item.model_dump()
        if d.get("label") not in CONTEXT_LABELS:
            continue
        pg, bbox = _get_prov(d)
        if pg != table_page:
            continue
        cy = _center_y(bbox)
        text = d.get("text", "").strip()
        if not text:
            continue

        if cy > table_top_y:
            dist = cy - table_top_y
            if dist <= bbox_threshold:
                before.append((dist, text))
        elif cy < table_bot_y:
            dist = table_bot_y - cy
            if dist <= bbox_threshold:
                after.append((dist, text))

    before.sort(key=lambda x: x[0])
    after.sort(key=lambda x: x[0])
    return before, after


# ---------------------------------------------------------------------------
# 임베딩 유사도 필터 (W3)
# ---------------------------------------------------------------------------

def _embedding_filter(
    candidates: list[tuple[float, str]],
    reference_text: str,
    sim_threshold: float,
) -> list[str]:
    """
    TODO W3: 코사인 유사도 >= sim_threshold 인 단락만 통과.

    현재는 모든 단락 통과 (bbox 필터만 적용된 상태).
    """
    # W3 구현 전까지 bbox 통과한 단락 전부 반환
    return [text for _, text in candidates]


# ---------------------------------------------------------------------------
# 메인 함수
# ---------------------------------------------------------------------------

def attach_context_paragraphs(
    eu: "EvidenceUnit",
    doc,
    bbox_threshold: float = CONTEXT_WINDOW_PT,
    sim_threshold: float = 0.40,    # W3에서 활성화
) -> "EvidenceUnit":
    """
    EU에 인접 설명 단락 + 섹션 헤더 추가.

    W2: bbox 거리 기반 수집
    W3: 코사인 유사도 필터 추가 예정

    Args:
        eu:             팀원1이 만든 EU 기본 뼈대
        doc:            DoclingDocument
        bbox_threshold: 표 위아래 수집 범위 (PDF 포인트). 기본 300pt
        sim_threshold:  임베딩 유사도 임계값 (W3 활성화 예정)

    Returns:
        context_before / context_after / section_header 가 채워진 EU
    """
    t_dict = None
    table_top_y = 0.0
    table_bot_y = 0.0

    # EU bbox는 0~1 normalized → 원본 pt 단위가 필요하므로
    # doc.tables에서 해당 EU의 원본 bbox를 다시 조회
    for table in doc.tables:
        d = table.model_dump()
        prov_list = d.get("prov", [])
        if not prov_list:
            continue
        p = prov_list[0]
        if p.get("page_no", -1) != eu.page_no:
            continue
        # eu_id 인덱스로 매칭 (eu-p{page}-{idx} 형식)
        t_dict = d
        table_top_y = p.get("bbox", {}).get("t", 0.0)
        table_bot_y = p.get("bbox", {}).get("b", 0.0)
        break   # 같은 페이지 첫 번째 표 — 다중 표 케이스는 추후 개선

    if t_dict is None:
        return eu

    # 섹션 헤더
    eu.section_header = find_section_header(doc, eu.page_no, table_top_y)

    # bbox 수집
    before_candidates, after_candidates = _collect_by_bbox(
        doc, eu.page_no, table_top_y, table_bot_y, bbox_threshold
    )

    # 임베딩 필터 (W3 전까지 전부 통과)
    reference = eu.caption_text or eu.table_abstract or ""
    eu.context_before = _embedding_filter(before_candidates, reference, sim_threshold)
    eu.context_after  = _embedding_filter(after_candidates,  reference, sim_threshold)

    return eu