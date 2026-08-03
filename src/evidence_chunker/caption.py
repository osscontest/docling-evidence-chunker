"""
_caption_mapper.py

캡션 ↔ 표 매핑 알고리즘 (W2 PoC, 팀원 1 담당).

역할:
    Docling TableItem.captions RefItem 필드를 파싱해 표와 캡션 텍스트를 1:1로 연결.
    복수 캡션 참조, 캡션 충돌(같은 캡션을 여러 표가 참조), cross-page 캡션 여부를
    감지해 validate_mapping()의 통계로 노출한다.

    실제 PDF(한국어 보고서) 검증 중 발견한 Docling 데이터 품질 이슈 2건에 대한
    bbox 거리 기반 fallback 포함:
      1. captions RefItem이 비어있는데 caption 라벨 텍스트는 실제로 존재하는 경우
      2. captions RefItem은 있는데 캡션이 아닌 파편(단위 표기 등)을 가리키는 경우
         (진짜 캡션이 section_header로 잘못 라벨링된 채 근처에 있는 경우 포함)

    두 경우 모두 caption_confidence="inferred"로 표시 (interfaces.EvidenceUnit.
    caption_confidence의 direct/inferred/none 표기와 통일).

W3 예외처리 (본 파일에서 처리):
    - 캡션 없는 표: 아래 fallback을 모두 거치고도 못 찾으면 confidence="none"
    - 복수 캡션 병합: captions RefItem이 2개 이상이면 캡션 패턴에 맞는 텍스트를
      모두 이어붙여 caption_text로 사용 (multi_caption=True로 표시)
    - 다음/이전 페이지 캡션: 같은 페이지 bbox fallback도 실패하면 표 바로 앞/뒤
      페이지에서 캡션 패턴 텍스트를 재탐색 (_find_caption_adjacent_page,
      cross_page=True로 표시)

    실제 테스트 PDF(영어 논문 2 + 한국어 보고서 + GPT-3) 어디에서도 복수 캡션·
    cross-page 케이스가 실제로 발동한 적은 없었음 (README 참고). 그래도 향후
    입력 문서에서 발생할 수 있는 케이스라 합성 데이터로 검증함
    (scripts/08_caption_exception_tests.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

# 캡션 fallback 탐색 범위 (표 위/아래, PDF 포인트).
# 일반 컨텍스트 단락(300pt)보다 좁게 잡음 — 캡션은 보통 표 바로 옆에 있음.
CAPTION_SEARCH_PT = 200.0

# fallback 후보로 볼 라벨. caption이 section_header로 잘못 라벨링되는
# 실사례(한국어 보고서 "<표 3>...")가 있어 section_header도 포함.
_CANDIDATE_LABELS = {"caption", "section_header", "text"}

# "Table 1", "Figure 2", "표1", "<표 3>", "그림 2" 등 번호가 붙은 캡션 패턴.
# 부록 전용 문자.숫자 번호 체계("Table C.1", "Figure G.4")도 포함 — GPT-3 논문
# 부록에서 실제로 이 형식을 쓰는데 인식을 못 해 캡션 있는 표 35개가 전부
# "none"으로 빠지는 문제가 있었음 (숫자 앞 문자 하나를 허용하지 않았던 게 원인).
# 실제 버그 사례("(단위: 1), %)" 같은 파편)를 걸러내는 핵심 필터.
_CAPTION_PATTERN = re.compile(r"(table|figure|fig|표|그림)\s*\.?\s*[A-Z]?\.?\s*\d+", re.IGNORECASE)
_MIN_CAPTION_LEN = 8

# 인접 페이지 캡션 탐색을 허용할 "페이지 경계 근처" 범위 (페이지 높이 대비 비율).
# 표가 페이지 위/아래 15% 안쪽에 걸쳐 있을 때만 인접 페이지를 본다.
# 실제 버그 사례: GPT-3 논문에서 목차가 표로 오인식된 케이스(표가 페이지 중간을
# 거의 다 차지) -- 게이팅 없이는 다음 페이지의 무관한 Figure 캡션을 잘못 채택함.
_ADJACENT_PAGE_EDGE_RATIO = 0.15


@dataclass
class CaptionMapping:
    table_index: int
    page_no: int
    caption_text: Optional[str]
    caption_ref: Optional[str]        # 예: "#/texts/12" (fallback으로 찾은 경우 None)
    confidence: Literal["direct", "inferred", "none"]
    multi_caption: bool = False       # 표 하나가 captions RefItem 2개 이상 보유
    cross_page: bool = False          # 캡션이 표와 다른 페이지에 위치


def resolve_ref(doc, cref: str) -> dict:
    """'#/texts/3' 형태의 cref -> model_dump() dict."""
    try:
        parts = cref.strip("#/").split("/")
        obj = doc
        for p in parts:
            obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
        return obj.model_dump() if hasattr(obj, "model_dump") else {}
    except Exception:
        return {}


def _get_prov_page(item_dict: dict) -> int:
    prov = item_dict.get("prov", [])
    return prov[0].get("page_no", -1) if prov else -1


def _cref_of(ref) -> str:
    return ref.get("cref", "") if isinstance(ref, dict) else getattr(ref, "cref", "")


# ---------------------------------------------------------------------------
# [8/3] 구조적 그림 캡션 판별 (텍스트 패턴 대신 doc 구조로 판단)
# ---------------------------------------------------------------------------
# 3-13은 "fig/figure/그림으로 시작하는 텍스트"라는 표면 패턴으로 그림 캡션을
# 걸러냈다 (interfaces.EvidenceUnit.safe_caption과 동일한 방식). 이건 ieee1.pdf에서
# 실제로 관찰된 한 가지 실패 패턴(영어/한국어 "Figure/Fig/그림" 접두어)만 잡고,
# Docling이 캡션을 다른 방식으로 잘못 연결하는 경우(다른 언어, 다른 표현)는
# 여전히 못 잡는 whack-a-mole 방식이라는 지적이 있었음 (8/3 리뷰).
#
# Docling의 DoclingDocument 스키마상 TableItem과 PictureItem은 둘 다
# FloatingItem을 상속하고, 둘 다 동일한 captions: list[RefItem] 필드를 갖는다
# (docling-core doc_items.py 확인: FloatingItem.captions, PictureItem(FloatingItem),
# TableItem(FloatingItem)). 즉 "이 텍스트가 어느 PictureItem의 captions에도
# 참조되어 있는가"는 텍스트 내용을 전혀 안 보고도 "이 텍스트는 그림 캡션으로
# 저자가 직접 지정한 것"임을 구조적으로 확인할 수 있는 훨씬 강한 신호다.
# 텍스트 패턴 매칭보다 이걸 1차 방어선으로 쓰고, 텍스트 패턴(3-11 safe_caption)은
# 이 구조적 체크가 못 잡는 경우를 위한 2차 안전망으로 남겨둔다.
def _get_picture_caption_refs(doc) -> set[str]:
    """doc.pictures[*].captions[*].cref 전체 집합. 실패해도 빈 집합(안전하게 무시)."""
    refs: set[str] = set()
    try:
        for pic in getattr(doc, "pictures", []) or []:
            pic_dict = pic.model_dump() if hasattr(pic, "model_dump") else {}
            for cap in pic_dict.get("captions", []) or []:
                cref = _cref_of(cap)
                if cref:
                    refs.add(cref)
    except Exception:
        return set()
    return refs


def _center_y(bbox: dict) -> float:
    return (bbox.get("t", 0.0) + bbox.get("b", 0.0)) / 2.0


# 캡션 패턴은 텍스트 맨 앞부분에서만 확인 (아래 CAPTION_PREFIX_CHARS 참고).
CAPTION_PREFIX_CHARS = 20


def _looks_like_caption(text: Optional[str]) -> bool:
    """
    "Table 1", "<표 3>" 같은 번호 붙은 캡션 패턴인지 확인.

    패턴은 텍스트 맨 앞부분(CAPTION_PREFIX_CHARS자)에서만 찾는다. 전체 텍스트에서
    찾으면, 본문 중간에 "Figure 2.2" 같은 참조가 우연히 들어간 긴 서술형 문단도
    캡션으로 오탐하는 실사례가 있었음 (GPT-3 논문 부록: "This appendix contains
    the calculations ... Figure 2.2. As a simplifying ..." 전체가 캡션으로 잘못
    채택됨). 실제 캡션은 항상 "Table N", "<표 N>"으로 시작하므로 앞부분만 봐도 충분.

    실사례: RefItem이 "(단위: 1), %)" 같은 각주 파편을 가리키는 버그를 걸러내기 위함.
    """
    if not text or len(text.strip()) < _MIN_CAPTION_LEN:
        return False
    prefix = text.strip()[:CAPTION_PREFIX_CHARS]
    return bool(_CAPTION_PATTERN.search(prefix))


# [3-13] direct RefItem 검증 전용 — table.captions가 그림(Figure) 캡션을
# 가리키는 경우 "direct"로 잘못 신뢰하지 않기 위함. ieee1.pdf 실사례:
# 표의 captions RefItem이 "Fig. 3. Framework of ISAC technologies..."를
# 가리키는데도 confidence="direct"로 확정돼버려서, interfaces.py 쪽에서
# 이 잘못된 캡션이 EU의 모든 하위 유닛에 접두사로 증폭되는 사고로 이어졌음
# (3-11/3-12 참고). caption_confidence는 "출처의 신뢰도"를 뜻하는데,
# 그림 캡션을 direct로 확정하는 건 그 신뢰도 자체가 거짓이므로 여기서 막는다.
#
# bbox/인접 페이지 fallback(_find_caption_by_bbox, _find_caption_adjacent_page)은
# 기존 _CAPTION_PATTERN(figure/fig/그림 포함)을 그대로 쓴다. 표 캡션이 정말
# 없는 페이지에서는 근처 텍스트라도 "inferred"로 남겨야, interfaces.py의
# safe_caption/context_before_with_fallback이 여전히 그 텍스트를 문맥으로
# 활용할 수 있다 — direct 검증만 좁히고 fallback까지 좁히면 회수 가능한
# 정보(ieee5.pdf의 "NWPU-RESISC45" 키워드)까지 원천 봉쇄되는 부작용이 있음.
_TABLE_CAPTION_PATTERN = re.compile(r"(table|표)\s*\.?\s*[A-Z]?\.?\s*\d+", re.IGNORECASE)


def _looks_like_table_caption(text: Optional[str]) -> bool:
    """direct RefItem 검증 전용: table/표 패턴만 인정(그림 캡션 오매핑 방지)."""
    if not text or len(text.strip()) < _MIN_CAPTION_LEN:
        return False
    prefix = text.strip()[:CAPTION_PREFIX_CHARS]
    return bool(_TABLE_CAPTION_PATTERN.search(prefix))


def _find_caption_by_bbox(
    doc, table_page: int, table_top_y: float, table_bot_y: float,
    picture_caption_refs: set[str] | None = None,
) -> Optional[str]:
    """
    captions RefItem이 없거나 품질 미달일 때, 같은 페이지에서 bbox 거리 기준으로
    캡션처럼 보이는 텍스트를 찾는 fallback.

    [8/3] picture_caption_refs가 주어지면, 이미 어느 PictureItem의 캡션으로
    구조적으로 확인된 텍스트(self_ref가 그 집합에 있음)는 후보에서 제외한다.
    """
    candidates: list[tuple[float, str]] = []
    picture_caption_refs = picture_caption_refs or set()

    for item in doc.texts:
        d = item.model_dump()
        if d.get("label") not in _CANDIDATE_LABELS:
            continue
        prov_list = d.get("prov", [])
        if not prov_list or prov_list[0].get("page_no", -1) != table_page:
            continue
        if d.get("self_ref") in picture_caption_refs:
            continue  # [8/3] 그림 캡션으로 구조적으로 확인됨 -> 표 캡션 후보에서 제외

        text = d.get("text", "").strip()
        if not _looks_like_caption(text):
            continue

        cy = _center_y(prov_list[0].get("bbox", {}))
        if cy > table_top_y:
            dist = cy - table_top_y
        elif cy < table_bot_y:
            dist = table_bot_y - cy
        else:
            dist = 0.0  # 표 bbox 범위 안 (드묾)

        if dist <= CAPTION_SEARCH_PT:
            candidates.append((dist, text))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _get_page_height(doc, page_no: int) -> Optional[float]:
    """doc.pages[page_no].size.height 조회. 페이지 정보 없으면 None."""
    pages = getattr(doc, "pages", None)
    if not pages:
        return None
    pg = pages.get(page_no) if hasattr(pages, "get") else None
    if pg is None:
        return None
    d = pg.model_dump() if hasattr(pg, "model_dump") else pg
    return (d.get("size") or {}).get("height")


def _find_caption_adjacent_page(
    doc, table_page: int, table_top_y: float, table_bot_y: float,
    picture_caption_refs: set[str] | None = None,
) -> tuple[Optional[str], bool]:
    """
    같은 페이지 bbox fallback도 실패했을 때, 표 바로 앞/뒤 페이지에서 캡션 재탐색.

    실사례: 표가 페이지 최상단에서 시작하면 캡션은 이전 페이지 맨 아래에,
            표가 페이지 최하단에서 끝나면 캡션은 다음 페이지 맨 위에 남을 수 있음.

    표가 페이지 경계(위/아래 _ADJACENT_PAGE_EDGE_RATIO 이내)에 걸쳐 있을 때만
    탐색한다. 게이팅 없이 무조건 인접 페이지를 보면, 표가 페이지 중간에 있을 때도
    우연히 옆 페이지에 있는 무관한 그림/표의 캡션을 잘못 채택할 수 있음
    (실사례: GPT-3 논문에서 목차가 표로 오인식된 케이스).

    페이지가 다르면 좌표계(페이지별 y 원점)가 서로 달라 bbox 거리 비교가
    무의미하므로, "이전 페이지에서 가장 아래" / "다음 페이지에서 가장 위"에
    있는 캡션 패턴 텍스트를 채택한다.

    Returns:
        (caption_text, found) — 못 찾거나 게이팅에 걸리면 (None, False)
    """
    page_height = _get_page_height(doc, table_page)
    if not page_height:
        return None, False

    near_top = table_top_y >= page_height * (1 - _ADJACENT_PAGE_EDGE_RATIO)
    near_bottom = table_bot_y <= page_height * _ADJACENT_PAGE_EDGE_RATIO
    if not near_top and not near_bottom:
        return None, False

    picture_caption_refs = picture_caption_refs or set()
    prev_candidates: list[tuple[float, str]] = []
    next_candidates: list[tuple[float, str]] = []

    for item in doc.texts:
        d = item.model_dump()
        if d.get("label") not in _CANDIDATE_LABELS:
            continue
        prov_list = d.get("prov", [])
        if not prov_list:
            continue
        if d.get("self_ref") in picture_caption_refs:
            continue  # [8/3] 그림 캡션으로 구조적으로 확인됨 -> 표 캡션 후보에서 제외
        pg = prov_list[0].get("page_no", -1)
        text = d.get("text", "").strip()
        if not _looks_like_caption(text):
            continue
        cy = _center_y(prov_list[0].get("bbox", {}))

        if near_top and pg == table_page - 1:
            prev_candidates.append((cy, text))  # 페이지 맨 아래 = cy 최솟값
        elif near_bottom and pg == table_page + 1:
            next_candidates.append((cy, text))  # 페이지 맨 위 = cy 최댓값

    if prev_candidates:
        prev_candidates.sort(key=lambda x: x[0])
        return prev_candidates[0][1], True

    if next_candidates:
        next_candidates.sort(key=lambda x: -x[0])
        return next_candidates[0][1], True

    return None, False


def map_table_caption(doc, table, table_index: int) -> CaptionMapping:
    """
    단일 표에 대한 캡션 매핑 (1:1).

    우선순위:
      1. captions RefItem이 가리키는 텍스트가 캡션답게 생겼으면 채택 (direct).
         RefItem이 2개 이상(복수 캡션)이면 캡션 패턴에 맞는 텍스트를 모두 병합.
      2. RefItem이 없거나 전부 파편을 가리키면 같은 페이지 bbox 거리로 재탐색 (inferred)
      3. 같은 페이지에서도 못 찾으면 표 바로 앞/뒤 페이지에서 재탐색
         (inferred, cross_page=True)
      4. 전부 실패하면 캡션 없음 (none)
    """
    t_dict = table.model_dump()
    table_page = _get_prov_page(t_dict)
    prov_list = t_dict.get("prov", [])
    table_bbox = prov_list[0].get("bbox", {}) if prov_list else {}
    table_top_y = table_bbox.get("t", 0.0)
    table_bot_y = table_bbox.get("b", 0.0)

    cap_refs = t_dict.get("captions", [])
    multi_caption = len(cap_refs) > 1
    cross_page = False

    # [8/3] 그림 캡션으로 구조적으로 확인된 텍스트 ref 집합 (doc.pictures[*].captions).
    # 1차 방어선 -- 텍스트가 "Fig/Figure/그림"으로 시작하는지 보는 3-13/3-11의
    # 표면 패턴 매칭보다 강한 근거(저자가 실제로 그 텍스트를 그림 캡션으로
    # 지정했다는 구조적 사실)이므로, direct RefItem 검증에도 우선 적용한다.
    picture_caption_refs = _get_picture_caption_refs(doc)

    # captions RefItem이 가리키는 텍스트 전부 검증 후 유효한 것만 병합
    # (복수 캡션 케이스: "Table 1: ..." + "(continued)" 같이 여러 RefItem이
    #  각각 캡션답게 생길 수 있음)
    # [3-13] 여기서는 _looks_like_table_caption(table/표 전용)을 쓴다.
    # RefItem이 그림 캡션을 가리키면 direct로 인정하지 않고 아래 bbox
    # fallback으로 넘긴다 — fallback은 여전히 넓은 패턴을 쓰므로 텍스트
    # 자체를 잃지는 않고, confidence만 "direct"→"inferred"로 정직해진다.
    resolved_texts: list[str] = []
    resolved_refs: list[str] = []
    for ref in cap_refs:
        cref = _cref_of(ref)
        if cref in picture_caption_refs:
            continue  # [8/3] 구조적으로 그림 캡션 확인됨 -> direct 후보에서 제외
        cap_dict = resolve_ref(doc, cref)
        candidate_text = cap_dict.get("text") or None
        if not _looks_like_table_caption(candidate_text):
            continue
        caption_page = _get_prov_page(cap_dict)
        if caption_page != -1 and caption_page != table_page:
            cross_page = True
        resolved_texts.append(candidate_text.strip())
        resolved_refs.append(cref)

    if resolved_texts:
        return CaptionMapping(
            table_index=table_index,
            page_no=table_page,
            caption_text=" ".join(resolved_texts),
            caption_ref="; ".join(resolved_refs),
            confidence="direct",
            multi_caption=multi_caption,
            cross_page=cross_page,
        )

    fallback_text = _find_caption_by_bbox(doc, table_page, table_top_y, table_bot_y, picture_caption_refs)
    if fallback_text:
        return CaptionMapping(
            table_index=table_index,
            page_no=table_page,
            caption_text=fallback_text,
            caption_ref=None,
            confidence="inferred",
            multi_caption=multi_caption,
            cross_page=False,
        )

    adjacent_text, found_adjacent = _find_caption_adjacent_page(
        doc, table_page, table_top_y, table_bot_y, picture_caption_refs
    )
    if found_adjacent:
        return CaptionMapping(
            table_index=table_index,
            page_no=table_page,
            caption_text=adjacent_text,
            caption_ref=None,
            confidence="inferred",
            multi_caption=multi_caption,
            cross_page=True,
        )

    return CaptionMapping(
        table_index=table_index,
        page_no=table_page,
        caption_text=None,
        caption_ref=None,
        confidence="none",
        multi_caption=multi_caption,
        cross_page=cross_page,
    )


def map_all_captions(doc) -> list[CaptionMapping]:
    """문서 내 모든 표에 대해 캡션 매핑 수행."""
    return [map_table_caption(doc, table, i) for i, table in enumerate(doc.tables)]


def validate_mapping(mappings: list[CaptionMapping]) -> dict:
    """
    1:1 매핑 검증.

    반환:
        total_tables, mapped, unmapped, mapping_rate
        multi_caption_tables : 복수 캡션 RefItem을 가진 표 인덱스 목록
        cross_page_tables    : 캡션이 다른 페이지에 있는 표 인덱스 목록
        collisions           : 같은 caption_ref를 참조하는 표가 2개 이상인 경우
                                {caption_ref: [table_index, ...]}
    """
    total = len(mappings)
    mapped = sum(1 for m in mappings if m.caption_text)
    unmapped = total - mapped
    multi = [m.table_index for m in mappings if m.multi_caption]
    cross_page = [m.table_index for m in mappings if m.cross_page]
    by_confidence = {
        "direct": sum(1 for m in mappings if m.confidence == "direct"),
        "inferred": sum(1 for m in mappings if m.confidence == "inferred"),
        "none": sum(1 for m in mappings if m.confidence == "none"),
    }

    ref_owners: dict[str, list[int]] = {}
    for m in mappings:
        if m.caption_ref:
            ref_owners.setdefault(m.caption_ref, []).append(m.table_index)
    collisions = {ref: idxs for ref, idxs in ref_owners.items() if len(idxs) > 1}

    return {
        "total_tables": total,
        "mapped": mapped,
        "unmapped": unmapped,
        "mapping_rate": round(mapped / total, 3) if total else 0.0,
        "by_confidence": by_confidence,
        "multi_caption_tables": multi,
        "cross_page_tables": cross_page,
        "collisions": collisions,
    }
