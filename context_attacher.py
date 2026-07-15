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

CONTEXT_WINDOW_PT = 300.0   # 표 위아래 300 PDF 포인트 이내
CONTEXT_LABELS = {"text", "list_item", "paragraph"}
COLUMN_OVERLAP_TOL = 20.0   # 멀티컬럼 컬럼 판별 허용 오차 (pt)
ADJACENT_PAGE_EDGE_RATIO = 0.15  # 페이지 상하단 15% 이내일 때 인접 페이지 탐색


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


def _get_page_height(doc, page_no: int) -> float:
    pages = getattr(doc, "pages", {})
    pg = pages.get(page_no) if hasattr(pages, "get") else None
    if pg is None:
        return 0.0
    d = pg.model_dump() if hasattr(pg, "model_dump") else {}
    return (d.get("size") or {}).get("height", 0.0)


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
    bbox 거리 기준 단락 수집. 같은 페이지 + 인접 페이지 경계 케이스 포함.
    섹션 경계를 넘는 단락은 항상 제외.

    같은 컬럼 단락을 우선하되, 표가 본문 컬럼 그리드에 딱 맞지 않게
    배치된 경우(넓은 표가 옆으로 살짝 걸치는 등, 실사례: docling 기술
    보고서 8페이지 — 표와 도입 문단의 x축 간격이 28pt로 COLUMN_OVERLAP_TOL
    (20pt)을 살짝 넘어서 매칭이 안 됨) 같은 컬럼에서 아무것도 못 찾으면
    컬럼 제한 없이(같은 페이지 전체에서) 재탐색한다. 이미 section_boundary
    체크가 다른 섹션 내용이 섞이는 걸 막고 있으므로, 이 폴백을 추가해도
    무관한 컬럼의 내용을 잘못 끌어올 위험은 낮다.

    Returns:
        before: [(거리, 텍스트), ...] — 표 위쪽, 거리 오름차순
        after:  [(거리, 텍스트), ...] — 표 아래쪽, 거리 오름차순
    """
    table_top_y = table_bbox.get("t", 0.0)
    table_bot_y = table_bbox.get("b", 0.0)

    before_same_col: list[tuple[float, str]] = []
    before_any_col: list[tuple[float, str]] = []
    after_same_col: list[tuple[float, str]] = []
    after_any_col: list[tuple[float, str]] = []

    # 인접 페이지 탐색 여부 판단
    page_height = _get_page_height(doc, table_page)
    check_prev = page_height > 0 and table_top_y >= page_height * (1 - ADJACENT_PAGE_EDGE_RATIO)
    check_next = page_height > 0 and table_bot_y <= page_height * ADJACENT_PAGE_EDGE_RATIO

    before: list[tuple[float, str]] = []
    after: list[tuple[float, str]] = []

    for item in doc.texts:
        d = item.model_dump()
        if d.get("label") not in CONTEXT_LABELS:
            continue
        pg, bbox = _get_prov(d)
        cy = _center_y(bbox)
        text = d.get("text", "").strip()
        if not text:
            continue

        if pg == table_page:
            same_col = _same_column(table_bbox, bbox)

            if cy > table_top_y:
                dist = cy - table_top_y
                if dist > bbox_threshold:
                    continue
                if _has_section_boundary(doc, table_page, table_top_y, cy, "above"):
                    continue
                (before_same_col if same_col else before_any_col).append((dist, text))

            elif cy < table_bot_y:
                dist = table_bot_y - cy
                if dist > bbox_threshold:
                    continue
                if _has_section_boundary(doc, table_page, table_bot_y, cy, "below"):
                    continue
                (after_same_col if same_col else after_any_col).append((dist, text))

        # 이전 페이지 하단 단락 (표가 현재 페이지 맨 위 근처일 때)
        # BOTTOMLEFT: 페이지 하단 = cy 작은 값
        elif check_prev and pg == table_page - 1:
            if cy <= bbox_threshold:
                before.append((cy + 1.0, text))  # cy 작을수록 하단 = 표에 가까움

        # 다음 페이지 상단 단락 (표가 현재 페이지 맨 아래 근처일 때)
        # BOTTOMLEFT: 페이지 상단 = cy 큰 값
        elif check_next and pg == table_page + 1:
            next_page_height = _get_page_height(doc, table_page + 1)
            if next_page_height > 0 and cy >= next_page_height - bbox_threshold:
                after.append((next_page_height - cy + 1.0, text))

    before.extend(before_same_col or before_any_col)
    after.extend(after_same_col or after_any_col)

    before.sort(key=lambda x: x[0])
    after.sort(key=lambda x: x[0])
    return before, after


# ---------------------------------------------------------------------------
# 임베딩 유사도 필터
# ---------------------------------------------------------------------------

_EMBEDDING_MODEL = None  # 모듈 레벨 캐시 — EU마다 재로드 방지


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
    table_bbox: dict | None = None,
) -> "EvidenceUnit":
    """
    EU에 인접 설명 단락 + 섹션 헤더 추가.

    Args:
        eu:             EU 기본 뼈대
        doc:            DoclingDocument
        bbox_threshold: 표 위아래 수집 범위 (PDF 포인트). 기본 300pt
        sim_threshold:  임베딩 유사도 임계값. 기본 0.40
        table_bbox:     이 EU가 속한 표의 원본(PDF 포인트) bbox. 호출자가
                         doc.tables를 순회하며 이미 알고 있는 값을 직접
                         넘겨주는 게 정확하다 — find_duplicate_tables()로
                         일부 표가 제거되면 eu_id의 페이지 내 순번과
                         doc.tables의 페이지 내 스캔 순서가 어긋나서,
                         아래 eu_id 기반 fallback으로는 엉뚱한(제거된
                         중복 표의) bbox를 집어올 수 있다.
                         (W4 후속 회귀: 8페이지 표 dedup 후에도
                         attach_context_paragraphs가 여전히 제거된
                         중복 표의 위치로 컨텍스트를 찾고 있었음)
                         None이면 eu_id 기반 추정으로 fallback.

    Returns:
        context_before / context_after / section_header 가 채워진 EU
    """
    if table_bbox is None:
        # eu_id = "eu-p{page}-{idx}" — 같은 페이지 내 표 순서 (0-based) 추정.
        # 호출자가 table_bbox를 직접 넘기지 못하는 경우의 fallback이며,
        # 표 dedup이 있었다면 부정확할 수 있다.
        try:
            eu_idx = int(eu.eu_id.split("-")[-1])
        except (ValueError, IndexError):
            eu_idx = 0

        page_tables: list[dict] = []
        for table in doc.tables:
            d = table.model_dump()
            prov_list = d.get("prov", [])
            if not prov_list:
                continue
            p = prov_list[0]
            if p.get("page_no", -1) != eu.page_no:
                continue
            page_tables.append(p.get("bbox", {}))

        if eu_idx >= len(page_tables):
            return eu

        table_bbox = page_tables[eu_idx]

    table_top_y = table_bbox.get("t", 0.0)

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