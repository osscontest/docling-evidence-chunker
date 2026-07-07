"""
context_attacher.py

인접 단락 탐지 모듈.

attach_context_paragraphs(eu, doc) 가 메인 진입점.

* bbox 거리 기반 단락 수집
* 코사인 유사도 필터, 섹션 헤딩 경계 감지, 멀티컬럼 예외처리
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from interfaces import EvidenceUnit

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

CONTEXT_WINDOW_PT = 300.0  # 표 위아래 300 PDF 포인트 이내
CONTEXT_LABELS = {"text", "list_item", "paragraph"}
COLUMN_OVERLAP_TOL = 20.0  # 멀티컬럼 컬럼 판별 허용 오차 (pt)


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


def _same_column(table_bbox: dict, para_bbox: dict) -> bool:
    """
    표와 단락의 x 범위가 겹치면 같은 컬럼으로 판단.
    단일 컬럼 PDF에서는 항상 True.
    """
    t_l = table_bbox.get("l", 0.0)
    t_r = table_bbox.get("r", 0.0)
    p_l = para_bbox.get("l", 0.0)
    p_r = para_bbox.get("r", 0.0)
    return p_r >= t_l - COLUMN_OVERLAP_TOL and p_l <= t_r + COLUMN_OVERLAP_TOL


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
# 섹션 헤딩 경계 감지
# ---------------------------------------------------------------------------

def _has_section_boundary(
    doc,
    table_page: int,
    table_edge_y: float,
    para_cy: float,
    direction: str,  # "above" | "below"
) -> bool:
    """
    표와 단락 사이에 section_header가 있으면 True (과포함 방지).

    above: para_cy < section_cy < table_edge_y
    below: table_edge_y < section_cy < para_cy
    """
    for item in doc.texts:
        d = item.model_dump()
        if d.get("label") != "section_header":
            continue
        pg, bbox = _get_prov(d)
        if pg != table_page:
            continue
        cy = _center_y(bbox)

        if direction == "above" and table_edge_y < cy < para_cy:
            return True
        if direction == "below" and para_cy < cy < table_edge_y:
            return True

    return False


# ---------------------------------------------------------------------------
# bbox 거리 기반 단락 수집
# ---------------------------------------------------------------------------

def _collect_by_bbox(
    doc,
    table_page: int,
    table_bbox: dict,
    bbox_threshold: float,
) -> tuple[list[tuple[float, str]], list[tuple[float, str]]]:
    """
    같은 페이지에서 bbox 거리 기준 단락 수집.
    다른 컬럼 단락 및 섹션 경계를 넘는 단락은 제외.

    Returns:
        before: [(거리, 텍스트), ...] — 표 위쪽, 거리 오름차순
        after:  [(거리, 텍스트), ...] — 표 아래쪽, 거리 오름차순
    """
    table_top_y = table_bbox.get("t", 0.0)
    table_bot_y = table_bbox.get("b", 0.0)

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

        if not _same_column(table_bbox, bbox):
            continue

        if cy > table_top_y:
            dist = cy - table_top_y
            if dist > bbox_threshold:
                continue
            if _has_section_boundary(doc, table_page, table_top_y, cy, "above"):
                continue
            before.append((dist, text))

        elif cy < table_bot_y:
            dist = table_bot_y - cy
            if dist > bbox_threshold:
                continue
            if _has_section_boundary(doc, table_page, table_bot_y, cy, "below"):
                continue
            after.append((dist, text))

    before.sort(key=lambda x: x[0])
    after.sort(key=lambda x: x[0])
    return before, after


# ---------------------------------------------------------------------------
# 임베딩 유사도 필터
# ---------------------------------------------------------------------------

_EMBEDDING_MODEL = None  # 모듈 레벨 캐시


def _get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDING_MODEL


def _embedding_filter(
    candidates: list[tuple[float, str]],
    reference_text: str,
    sim_threshold: float,
) -> list[str]:
    """
    코사인 유사도 >= sim_threshold 인 단락만 통과.

    reference_text 없으면 전부 통과 (캡션 없는 표 방어).
    sentence-transformers 미설치 시 전부 통과 (graceful degradation).
    """
    if not reference_text:
        return [text for _, text in candidates]

    try:
        from sentence_transformers import util
        model = _get_embedding_model()
    except ImportError:
        return [text for _, text in candidates]

    if not candidates:
        return []

    ref_emb = model.encode(reference_text, convert_to_tensor=True)
    texts = [text for _, text in candidates]
    cand_embs = model.encode(texts, convert_to_tensor=True)
    scores = util.cos_sim(ref_emb, cand_embs)[0]

    return [text for text, score in zip(texts, scores) if score.item() >= sim_threshold]


# ---------------------------------------------------------------------------
# 메인 함수
# ---------------------------------------------------------------------------

def attach_context_paragraphs(
    eu: "EvidenceUnit",
    doc,
    bbox_threshold: float = CONTEXT_WINDOW_PT,
    sim_threshold: float = 0.40,
) -> "EvidenceUnit":
    """
    EU에 인접 설명 단락 + 섹션 헤더 추가.

    Args:
        eu:             EU 기본 뼈대
        doc:            DoclingDocument
        bbox_threshold: 표 위아래 수집 범위 (PDF 포인트). 기본 300pt
        sim_threshold:  임베딩 유사도 임계값. 기본 0.40

    Returns:
        context_before / context_after / section_header 가 채워진 EU
    """
    table_bbox: dict = {}
    table_top_y = 0.0

    for table in doc.tables:
        d = table.model_dump()
        prov_list = d.get("prov", [])
        if not prov_list:
            continue
        p = prov_list[0]
        if p.get("page_no", -1) != eu.page_no:
            continue
        table_bbox = p.get("bbox", {})
        table_top_y = table_bbox.get("t", 0.0)
        break  # 다중 표 케이스는 추후 eu_id 인덱스 매칭으로 개선

    if not table_bbox:
        return eu

    eu.section_header = find_section_header(doc, eu.page_no, table_top_y)

    before_candidates, after_candidates = _collect_by_bbox(
        doc, eu.page_no, table_bbox, bbox_threshold
    )

    reference = eu.caption_text or eu.table_abstract or ""
    eu.context_before = _embedding_filter(before_candidates, reference, sim_threshold)
    eu.context_after  = _embedding_filter(after_candidates,  reference, sim_threshold)

    return eu