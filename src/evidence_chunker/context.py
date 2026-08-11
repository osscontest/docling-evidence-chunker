"""
context.py

인접 단락 탐지 모듈.

attach_context_paragraphs(eu, parsed) 가 메인 진입점.

* bbox 거리 기반 단락 수집
* 코사인 유사도 필터(기본 비활성 — sim_threshold=0.0), 섹션 헤딩 경계 감지, 멀티컬럼 예외처리

sim_threshold 기본값이 0.0(Roadmap 7, Benchmark.md §11)으로 바뀐 뒤로는
_embedding_filter()가 model.encode() 호출 자체를 스킵한다(아래 참고) — 즉
기본 설정으로 쓰면 sentence-transformers/torch가 설치돼 있지 않아도 청킹이
그대로 동작한다. sim_threshold를 0 초과로 직접 올려서 넘기는 호출자만 그
무거운 의존성이 필요하다(pyproject.toml의 optional extra로 분리됨).

Stage 4 파서 추상화: parser.base.ParsedDoc/TextBlock/TableBlock 기반 —
Docling DoclingDocument를 직접 만지지 않는다. 좌표는 TOPLEFT(파서 경계에서
정규화, parser/base.py 참고) 기준이라 이 모듈의 모든 상/하 부등호가 원본
BOTTOMLEFT 버전과 반대다. bbox flip 자체는 tests/test_parser_docling.py가
별도로 잠갔다 — 여기서는 그 위에서 도는 알고리즘의 부등호만 재도출했다.

페이지 경계(이전/다음 페이지 탐색)의 "위/아래"도 뒤집힌다: BOTTOMLEFT에서는
페이지 하단이 y=0(페이지 높이 무관)이라 이전 페이지 쪽 거리 계산에 그
페이지의 높이가 필요 없었는데, TOPLEFT에서는 페이지 하단이 y=page_height
(페이지마다 다를 수 있음)라 이전 페이지 높이 조회가 새로 필요해졌다.
반대로 다음 페이지 쪽은 원래 필요하던 페이지 높이 조회가 필요 없어졌다
(TOPLEFT 상단은 항상 y=0).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .parser.base import BBox, ParsedDoc, TableBlock
    from .unit import EvidenceUnit

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

CONTEXT_WINDOW_PT = 300.0   # 표 위아래 300 PDF 포인트 이내
CONTEXT_LABEL_NAMES = {"text", "list_item", "paragraph"}
COLUMN_OVERLAP_TOL = 20.0   # 멀티컬럼 컬럼 판별 허용 오차 (pt)
ADJACENT_PAGE_EDGE_RATIO = 0.15  # 페이지 상하단 15% 이내일 때 인접 페이지 탐색


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _center_y(bbox: "BBox") -> float:
    return (bbox.t + bbox.b) / 2.0


def _same_column(table_bbox: "BBox", para_bbox: "BBox") -> bool:
    """
    표와 단락의 x 범위가 겹치면 같은 컬럼으로 판단.
    단일 컬럼 PDF에서는 항상 True.
    """
    return (
        para_bbox.r >= table_bbox.l - COLUMN_OVERLAP_TOL
        and para_bbox.l <= table_bbox.r + COLUMN_OVERLAP_TOL
    )


# ---------------------------------------------------------------------------
# 섹션 헤더 탐색
# ---------------------------------------------------------------------------

def find_section_header(parsed: "ParsedDoc", table_page: int, table_top_y: float) -> str | None:
    """
    표 위쪽에서 가장 가까운 section_header 반환.

    TOPLEFT 좌표계: 시각적으로 표 위 = cy < table_top_y.
    같은 페이지에 없으면 이전 페이지 중 마지막 헤더.
    """
    from .parser.base import BlockLabel

    same_above: list[tuple[float, str]] = []
    prev_pages: list[tuple[int, float, str]] = []

    for item in parsed.texts:
        if item.label != BlockLabel.SECTION_HEADER:
            continue
        if item.page_no == -1:
            continue
        cy = _center_y(item.bbox)
        text = item.text.strip()
        if not text:
            continue

        if item.page_no == table_page and cy < table_top_y:
            same_above.append((table_top_y - cy, text))
        elif item.page_no < table_page:
            prev_pages.append((item.page_no, cy, text))

    if same_above:
        same_above.sort(key=lambda x: x[0])
        return same_above[0][1]

    if prev_pages:
        # 가장 가까운 이전 페이지(pg 최댓값) 중, 그 페이지에서 가장 나중에
        # 등장한(원본 BOTTOMLEFT 기준 cy 최댓값 == TOPLEFT cy 최솟값) 헤더.
        prev_pages.sort(key=lambda x: (x[0], -x[1]))
        return prev_pages[-1][2]

    return None


# ---------------------------------------------------------------------------
# 섹션 헤딩 경계 감지
# ---------------------------------------------------------------------------

def _has_section_boundary(
    parsed: "ParsedDoc",
    table_page: int,
    table_edge_y: float,
    para_cy: float,
    direction: str,  # "above" | "below"
) -> bool:
    """
    표와 단락 사이에 section_header가 있으면 True (과포함 방지).

    TOPLEFT: above는 para가 표보다 위(작은 y)에 있는 경우이므로
    para_cy < section_cy < table_edge_y, below는 그 반대
    (table_edge_y < section_cy < para_cy) — BOTTOMLEFT 원본과 두 분기의
    부등호 자체가 통째로 뒤바뀐다.
    """
    from .parser.base import BlockLabel

    for item in parsed.texts:
        if item.label != BlockLabel.SECTION_HEADER:
            continue
        if item.page_no != table_page:
            continue
        cy = _center_y(item.bbox)

        if direction == "above" and para_cy < cy < table_edge_y:
            return True
        if direction == "below" and table_edge_y < cy < para_cy:
            return True

    return False


# ---------------------------------------------------------------------------
# bbox 거리 기반 단락 수집
# ---------------------------------------------------------------------------

def _collect_by_bbox(
    parsed: "ParsedDoc",
    table_page: int,
    table_bbox: "BBox",
    bbox_threshold: float,
) -> tuple[list[tuple[float, str, int]], list[tuple[float, str, int]]]:
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
        before: [(거리, 텍스트, page_no), ...] — 표 위쪽, 거리 오름차순
        after:  [(거리, 텍스트, page_no), ...] — 표 아래쪽, 거리 오름차순

    [fix] page_no 를 3번째 원소로 추가 — 인접 페이지(table_page±1)에서 끌어온
    단락이 있으면 attach_context_paragraphs()가 eu.page_span 에 반영한다.
    """
    from .parser.base import BlockLabel

    table_top_y = table_bbox.t
    table_bot_y = table_bbox.b

    before_same_col: list[tuple[float, str, int]] = []
    before_any_col: list[tuple[float, str, int]] = []
    after_same_col: list[tuple[float, str, int]] = []
    after_any_col: list[tuple[float, str, int]] = []

    # 인접 페이지 탐색 여부 판단. TOPLEFT: 페이지 위쪽 = y가 0에 가까움,
    # 페이지 아래쪽 = y가 page_height에 가까움.
    page_height = parsed.page_sizes.get(table_page, (0.0, 0.0))[1]
    check_prev = page_height > 0 and table_top_y <= page_height * ADJACENT_PAGE_EDGE_RATIO
    check_next = page_height > 0 and table_bot_y >= page_height * (1 - ADJACENT_PAGE_EDGE_RATIO)

    before: list[tuple[float, str, int]] = []
    after: list[tuple[float, str, int]] = []

    for item in parsed.texts:
        if item.label.value not in CONTEXT_LABEL_NAMES:
            continue
        text = item.text.strip()
        if not text:
            continue
        cy = _center_y(item.bbox)

        if item.page_no == table_page:
            same_col = _same_column(table_bbox, item.bbox)

            if cy < table_top_y:  # TOPLEFT: 표 위
                dist = table_top_y - cy
                if dist > bbox_threshold:
                    continue
                if _has_section_boundary(parsed, table_page, table_top_y, cy, "above"):
                    continue
                (before_same_col if same_col else before_any_col).append((dist, text, table_page))

            elif cy > table_bot_y:  # TOPLEFT: 표 아래
                dist = cy - table_bot_y
                if dist > bbox_threshold:
                    continue
                if _has_section_boundary(parsed, table_page, table_bot_y, cy, "below"):
                    continue
                (after_same_col if same_col else after_any_col).append((dist, text, table_page))

        # 이전 페이지 하단 단락 (표가 현재 페이지 맨 위 근처일 때)
        # TOPLEFT: 페이지 하단 = cy가 그 페이지 높이에 가까운 값 — BOTTOMLEFT와
        # 달리 페이지마다 다를 수 있는 그 페이지 자신의 높이가 필요하다.
        elif check_prev and item.page_no == table_page - 1:
            prev_page_height = parsed.page_sizes.get(table_page - 1, (0.0, 0.0))[1]
            if prev_page_height > 0 and cy >= prev_page_height - bbox_threshold:
                before.append((prev_page_height - cy + 1.0, text, table_page - 1))

        # 다음 페이지 상단 단락 (표가 현재 페이지 맨 아래 근처일 때)
        # TOPLEFT: 페이지 상단은 항상 y=0이라 그 페이지 높이 조회가 필요 없다
        # (원본 BOTTOMLEFT는 페이지 상단이 page_height라 조회가 필요했음).
        elif check_next and item.page_no == table_page + 1:
            if cy <= bbox_threshold:
                after.append((cy + 1.0, text, table_page + 1))

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
    candidates: list[tuple[float, str, int]],
    reference_text: str,
    sim_threshold: float,
) -> list[tuple[str, int]]:
    """
    코사인 유사도 >= sim_threshold 인 단락만 통과.

    reference_text 없으면 전부 통과 (캡션 없는 표 방어).
    sentence-transformers 미설치 시 전부 통과 (graceful degradation).
    sim_threshold <= 0 이면 전부 통과 — 아래 short-circuit 참고.

    [fix] 반환을 (텍스트, page_no) 튜플로 바꿈 — 페이지 정보를 여기서
    버리면 attach_context_paragraphs()가 page_span 을 못 채운다.
    """
    if not reference_text:
        return [(text, page) for _, text, page in candidates]

    # [최적화] sim_threshold <= 0 이면 결과적으로 모든 후보가 통과한다
    # (실측 코사인 유사도는 사실상 항상 0 이상 — Benchmark.md §11).
    # 라이브러리 기본값이 0.0으로 바뀐 뒤(Roadmap 7) 이 분기가 사실상 항상
    # 타므로, 결과가 뻔한 model.encode() 호출(문서당 가장 무거운 연산 — 임베딩
    # 모델 로드 + 추론)을 아예 스킵한다. sentence-transformers/torch import도
    # 여기서 걸리지 않으므로, 사용자가 sim_threshold를 0 이하로 두면(기본값)
    # 그 무거운 패키지가 설치돼 있지 않아도 청킹이 그대로 동작한다
    # (pyproject.toml에서 sentence-transformers를 optional extra로 내린 것과 짝).
    if sim_threshold <= 0:
        return [(text, page) for _, text, page in candidates]

    try:
        from sentence_transformers import util
        model = _get_embedding_model()
    except ImportError:
        return [(text, page) for _, text, page in candidates]

    if not candidates:
        return []

    ref_emb = model.encode(reference_text, convert_to_tensor=True)
    texts = [text for _, text, _ in candidates]
    pages = [page for _, _, page in candidates]
    cand_embs = model.encode(texts, convert_to_tensor=True)
    scores = util.cos_sim(ref_emb, cand_embs)[0]

    return [(text, page) for text, page, score in zip(texts, pages, scores)
            if score.item() >= sim_threshold]


# ---------------------------------------------------------------------------
# 메인 함수
# ---------------------------------------------------------------------------

def attach_context_paragraphs(
    eu: "EvidenceUnit",
    parsed: "ParsedDoc",
    bbox_threshold: float = CONTEXT_WINDOW_PT,
    sim_threshold: float = 0.0,
    table_bbox: "BBox | None" = None,
) -> "EvidenceUnit":
    """
    EU에 인접 설명 단락 + 섹션 헤더 추가.

    Args:
        eu:             EU 기본 뼈대
        parsed:         파서 추상화 문서 (parser.base.ParsedDoc)
        bbox_threshold: 표 위아래 수집 범위 (PDF 포인트). 기본 300pt
        sim_threshold:  임베딩 유사도 임계값. 기본 0.0(사실상 무필터).
                        dev20 스윕(Benchmark.md §11)에서 0.40 대비
                        context_dependent EM +12.5pp, 감시 지표 저해 없음
                        확인 후 변경. 더 엄격한 필터가 필요하면 호출자가
                        직접 값을 올려서 넘기면 됨(파라미터는 유지).
        table_bbox:     이 EU가 속한 표의 원본(TOPLEFT 정규화) bbox. 호출자가
                         parsed.tables를 순회하며 이미 알고 있는 값을 직접
                         넘겨주는 게 정확하다 — find_duplicate_tables()로
                         일부 표가 제거되면 eu_id의 페이지 내 순번과
                         parsed.tables의 페이지 내 스캔 순서가 어긋나서,
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
        # eu_id = "{doc_id}-p{page}-{idx}" — 같은 페이지 내 표 순서 (0-based) 추정.
        # 호출자가 table_bbox를 직접 넘기지 못하는 경우의 fallback이며,
        # 표 dedup이 있었다면 부정확할 수 있다. 실제 파이프라인
        # (build_evidence_units)은 항상 table_bbox를 직접 넘기므로 이
        # 경로는 안 탄다 — doc_id에 하이픈이 있으면 eu_idx 파싱이 깨지는
        # 알려진 갭이 있음(doc_id 도입 커밋 참고).
        try:
            eu_idx = int(eu.eu_id.split("-")[-1])
        except (ValueError, IndexError):
            eu_idx = 0

        page_tables: list["BBox"] = [
            t.bbox for t in parsed.tables if t.page_no == eu.page_no
        ]

        if eu_idx >= len(page_tables):
            return eu

        table_bbox = page_tables[eu_idx]

    if not table_bbox:
        return eu

    table_top_y = table_bbox.t

    eu.section_header = find_section_header(parsed, eu.page_no, table_top_y)

    before_candidates, after_candidates = _collect_by_bbox(
        parsed, eu.page_no, table_bbox, bbox_threshold
    )

    reference = eu.caption_text or eu.table_abstract or ""
    before_kept = _embedding_filter(before_candidates, reference, sim_threshold)
    after_kept  = _embedding_filter(after_candidates,  reference, sim_threshold)

    eu.context_before = [text for text, _ in before_kept]
    eu.context_after  = [text for text, _ in after_kept]

    # [fix] 표 자신의 페이지 + 실제로 채택된 문맥 단락이 걸친 페이지 전부.
    # bbox_threshold 조건상 인접 페이지는 table_page±1 만 가능하므로 보통
    # {page_no} 또는 {page_no, page_no±1} 정도의 작은 집합이 된다.
    eu.page_span = {eu.page_no} | {p for _, p in before_kept} | {p for _, p in after_kept}

    return eu
