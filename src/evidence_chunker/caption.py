"""
caption.py

캡션 ↔ 표 매핑 알고리즘 (W2 PoC, 팀원 1 담당).

역할:
    parser.base.TableBlock.captions_refs(정수 인덱스)를 사용해 표와 캡션
    텍스트를 1:1로 연결. 복수 캡션 참조, 캡션 충돌(같은 캡션을 여러 표가
    참조), cross-page 캡션 여부를 감지해 validate_mapping()의 통계로 노출한다.

    실제 PDF(한국어 보고서) 검증 중 발견한 Docling 데이터 품질 이슈 2건에 대한
    bbox 거리 기반 fallback 포함:
      1. captions RefItem이 비어있는데 caption 라벨 텍스트는 실제로 존재하는 경우
      2. captions RefItem은 있는데 캡션이 아닌 파편(단위 표기 등)을 가리키는 경우
         (진짜 캡션이 section_header로 잘못 라벨링된 채 근처에 있는 경우 포함)

    두 경우 모두 caption_confidence="inferred"로 표시 (EvidenceUnit.
    caption_confidence의 direct/inferred/none 표기와 통일).

W3 예외처리 (본 파일에서 처리):
    - 캡션 없는 표: 아래 fallback을 모두 거치고도 못 찾으면 confidence="none"
    - 복수 캡션 병합: captions 참조가 2개 이상이면 캡션 패턴에 맞는 텍스트를
      모두 이어붙여 caption_text로 사용 (multi_caption=True로 표시)
    - 다음/이전 페이지 캡션: 같은 페이지 bbox fallback도 실패하면 표 바로 앞/뒤
      페이지에서 캡션 패턴 텍스트를 재탐색 (_find_caption_adjacent_page,
      cross_page=True로 표시)

    실제 테스트 PDF(영어 논문 2 + 한국어 보고서 + GPT-3) 어디에서도 복수 캡션·
    cross-page 케이스가 실제로 발동한 적은 없었음 (README 참고). 그래도 향후
    입력 문서에서 발생할 수 있는 케이스라 합성 데이터로 검증함
    (tests/test_caption_exceptions.py).

Stage 4 파서 추상화: parser.base.ParsedDoc/TableBlock 기반 — Docling
DoclingDocument를 직접 만지지 않는다. cref 문자열("#/texts/12") 파싱과
self_ref 문자열 대조가 사라지고, 전부 parsed.texts 인덱스 정수 비교가 됐다.
좌표 비교는 TOPLEFT(파서 경계에서 정규화, parser/base.py 참고) 기준 —
BOTTOMLEFT였던 원본과 부등호 방향이 전부 반대다("표 위" = cy가 작을수록
위쪽). bbox flip 자체는 tests/test_parser_docling.py가 별도로 잠갔다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from .parser.base import BBox, ParsedDoc, TableBlock

# 캡션 fallback 탐색 범위 (표 위/아래, PDF 포인트).
# 일반 컨텍스트 단락(300pt)보다 좁게 잡음 — 캡션은 보통 표 바로 옆에 있음.
CAPTION_SEARCH_PT = 200.0

# fallback 후보로 볼 라벨. caption이 section_header로 잘못 라벨링되는
# 실사례(한국어 보고서 "<표 3>...")가 있어 section_header도 포함.
_CANDIDATE_LABEL_NAMES = {"caption", "section_header", "text"}

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
    caption_ref: Optional[str]        # 예: "3" (texts 인덱스. fallback으로 찾은 경우 None)
    confidence: Literal["direct", "inferred", "none"]
    multi_caption: bool = False       # 표 하나가 captions 참조 2개 이상 보유
    cross_page: bool = False          # 캡션이 표와 다른 페이지에 위치


def _center_y(bbox: "BBox") -> float:
    return (bbox.t + bbox.b) / 2.0


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


# [3-13] direct 참조 검증 전용 — table.captions가 그림(Figure) 캡션을
# 가리키는 경우 "direct"로 잘못 신뢰하지 않기 위함. ieee1.pdf 실사례:
# 표의 captions 참조가 "Fig. 3. Framework of ISAC technologies..."를
# 가리키는데도 confidence="direct"로 확정돼버려서, EvidenceUnit 쪽에서
# 이 잘못된 캡션이 EU의 모든 하위 유닛에 접두사로 증폭되는 사고로 이어졌음
# (3-11/3-12 참고). caption_confidence는 "출처의 신뢰도"를 뜻하는데,
# 그림 캡션을 direct로 확정하는 건 그 신뢰도 자체가 거짓이므로 여기서 막는다.
#
# bbox/인접 페이지 fallback(_find_caption_by_bbox, _find_caption_adjacent_page)은
# 기존 _CAPTION_PATTERN(figure/fig/그림 포함)을 그대로 쓴다. 표 캡션이 정말
# 없는 페이지에서는 근처 텍스트라도 "inferred"로 남겨야, EvidenceUnit의
# safe_caption/context_before_with_fallback이 여전히 그 텍스트를 문맥으로
# 활용할 수 있다 — direct 검증만 좁히고 fallback까지 좁히면 회수 가능한
# 정보(ieee5.pdf의 "NWPU-RESISC45" 키워드)까지 원천 봉쇄되는 부작용이 있음.
_TABLE_CAPTION_PATTERN = re.compile(r"(table|표)\s*\.?\s*[A-Z]?\.?\s*\d+", re.IGNORECASE)


def _looks_like_table_caption(text: Optional[str]) -> bool:
    """direct 참조 검증 전용: table/표 패턴만 인정(그림 캡션 오매핑 방지)."""
    if not text or len(text.strip()) < _MIN_CAPTION_LEN:
        return False
    prefix = text.strip()[:CAPTION_PREFIX_CHARS]
    return bool(_TABLE_CAPTION_PATTERN.search(prefix))


def _find_caption_by_bbox(
    parsed: "ParsedDoc", table_page: int, table_top_y: float, table_bot_y: float,
    picture_caption_refs: set[int] | None = None,
) -> Optional[str]:
    """
    captions 참조가 없거나 품질 미달일 때, 같은 페이지에서 bbox 거리 기준으로
    캡션처럼 보이는 텍스트를 찾는 fallback.

    picture_caption_refs가 주어지면, 이미 어느 PictureItem의 캡션으로
    구조적으로 확인된 텍스트(index가 그 집합에 있음)는 후보에서 제외한다.
    """
    candidates: list[tuple[float, str]] = []
    picture_caption_refs = picture_caption_refs or set()

    for item in parsed.texts:
        if item.label.value not in _CANDIDATE_LABEL_NAMES:
            continue
        if item.page_no != table_page:
            continue
        if item.index in picture_caption_refs:
            continue  # 그림 캡션으로 구조적으로 확인됨 -> 표 캡션 후보에서 제외

        text = item.text.strip()
        if not _looks_like_caption(text):
            continue

        cy = _center_y(item.bbox)
        # TOPLEFT: 위쪽일수록 y가 작다.
        if cy < table_top_y:
            dist = table_top_y - cy
        elif cy > table_bot_y:
            dist = cy - table_bot_y
        else:
            dist = 0.0  # 표 bbox 범위 안 (드묾)

        if dist <= CAPTION_SEARCH_PT:
            candidates.append((dist, text))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _find_caption_adjacent_page(
    parsed: "ParsedDoc", table_page: int, table_top_y: float, table_bot_y: float,
    picture_caption_refs: set[int] | None = None,
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
    page_height = parsed.page_sizes.get(table_page, (0.0, 0.0))[1]
    if not page_height:
        return None, False

    # TOPLEFT: 페이지 위쪽 = y가 0에 가까움, 페이지 아래쪽 = y가 page_height에 가까움.
    near_top = table_top_y <= page_height * _ADJACENT_PAGE_EDGE_RATIO
    near_bottom = table_bot_y >= page_height * (1 - _ADJACENT_PAGE_EDGE_RATIO)
    if not near_top and not near_bottom:
        return None, False

    picture_caption_refs = picture_caption_refs or set()
    prev_candidates: list[tuple[float, str]] = []
    next_candidates: list[tuple[float, str]] = []

    for item in parsed.texts:
        if item.label.value not in _CANDIDATE_LABEL_NAMES:
            continue
        if item.index in picture_caption_refs:
            continue  # 그림 캡션으로 구조적으로 확인됨 -> 표 캡션 후보에서 제외
        text = item.text.strip()
        if not _looks_like_caption(text):
            continue
        cy = _center_y(item.bbox)

        if near_top and item.page_no == table_page - 1:
            prev_candidates.append((cy, text))  # 이전 페이지 맨 아래 = TOPLEFT에서 cy 최댓값
        elif near_bottom and item.page_no == table_page + 1:
            next_candidates.append((cy, text))  # 다음 페이지 맨 위 = TOPLEFT에서 cy 최솟값

    if prev_candidates:
        prev_candidates.sort(key=lambda x: -x[0])
        return prev_candidates[0][1], True

    if next_candidates:
        next_candidates.sort(key=lambda x: x[0])
        return next_candidates[0][1], True

    return None, False


# ---------------------------------------------------------------------------
# fallback 4: 표 구조 자체에 병합된 캡션 (W5 EU 완성도 진단, 팀원2)
# ---------------------------------------------------------------------------
#
# 위 세 fallback(direct/bbox/인접 페이지)은 전부 parsed.texts만 순회하는데,
# Docling이 "Table N: ..."을 별도 텍스트 아이템이 아니라 표 자체의 첫 행
# (다수 컬럼을 병합한 th/td 하나)으로 렌더링하는 경우가 있다 — 이러면 세
# fallback 모두 원천적으로 못 본다. 실사례: 47. gao-26-107681-p19(EU 완성도
# 진단, docs/EU완성도_진단.md 참고)의 table.html이
# `<th colspan="2">Table 1: Expert-Identified Risks...</th>`로 시작.
#
# 표본(77개) 검증: 13개(16.9%) 회수, "Total"/"Tables"/"Single-Family..." 같은
# 컬럼그룹·섹션 라벨은 캡션 패턴(table/표 + 숫자)에 안 맞아 정상 제외됨.
#
# [수정] 최초 구현은 발견한 캡션 행을 table.html에서 제거했으나, 리뷰에서
# 치명적 desync를 지적받아 되돌림: TableBlock은 표를 html(문자열, split.py가
# 씀)과 cells(dict 리스트, flatten.py가 씀) 두 개의 독립된 표현으로 병렬
# 보유한다(parser/base.py) — 서로 같은 Docling 표에서 각자 따로 계산된
# 것이라, 한쪽(html)만 고치면 cells는 캡션 행이 row 0에 그대로 남는다.
# 그러면 (a) build_col_header_map/infer_headers_fallback이 그 행을 열
# 헤더로 흡수해 flattened_rows 문장이 전부 캡션 텍스트로 오염되고,
# (b) split.py의 header_row_offset(html 기준)과 row_sentence_map(cells
# 기준)의 행 인덱스가 어긋나 분할 조각에 엉뚱한 행 문장이 붙는다. 이 함수는
# caption_text만 추출하고 table.html/cells는 건드리지 않는다 — 대가는
# EU.text에 캡션이 표 안에도 한 번 더 남는 경미한 토큰 중복뿐, 정확성
# 문제는 없다. html/cells를 동시에 일관되게 벗기려면 파서 경계(둘 다 같은
# 소스에서 만들어지는 parser/docling.py)에서 해야 하며, 여기(caption.py)는
# table_html 소유권이 없어 안전하게 못 한다.

_MERGED_HEADER_PATTERN = re.compile(
    r'<t[hd][^>]*colspan="(\d+)"[^>]*>\s*(.*?)\s*</t[hd]>',
    re.IGNORECASE | re.DOTALL,
)


def _find_caption_in_merged_header(table_html: Optional[str]) -> Optional[str]:
    """
    표 첫 행이 다수 컬럼을 병합한 th/td 하나로만 이루어져 있고, 그 안 텍스트가
    캡션 패턴(_looks_like_table_caption)에 맞으면 채택. table_html/cells는
    건드리지 않는다(위 주석 참고) — 캡션 텍스트만 뽑아서 반환.

    Returns:
        caption_text — 못 찾으면 None
    """
    if not table_html:
        return None
    m = _MERGED_HEADER_PATTERN.search(table_html)
    if not m:
        return None
    cell_text = re.sub(r"<[^>]+>", " ", m.group(2)).strip()
    if not _looks_like_table_caption(cell_text):
        return None
    return cell_text


def map_table_caption(parsed: "ParsedDoc", table: "TableBlock", table_index: int) -> CaptionMapping:
    """
    단일 표에 대한 캡션 매핑 (1:1).

    우선순위:
      1. captions 참조가 가리키는 텍스트가 캡션답게 생겼으면 채택 (direct).
         참조가 2개 이상(복수 캡션)이면 캡션 패턴에 맞는 텍스트를 모두 병합.
      2. 참조가 없거나 전부 파편을 가리키면 같은 페이지 bbox 거리로 재탐색 (inferred)
      3. 같은 페이지에서도 못 찾으면 표 바로 앞/뒤 페이지에서 재탐색
         (inferred, cross_page=True)
      4. 표 구조 자체(table.html)에 캡션이 병합된 행으로 들어있는지 확인 (inferred)
      5. 전부 실패하면 캡션 없음 (none)
    """
    table_page = table.page_no
    table_top_y = table.bbox.t
    table_bot_y = table.bbox.b

    cap_refs = table.caption_refs
    multi_caption = len(cap_refs) > 1
    cross_page = False

    picture_caption_refs = parsed.picture_caption_refs

    # captions 참조가 가리키는 텍스트 전부 검증 후 유효한 것만 병합
    # (복수 캡션 케이스: "Table 1: ..." + "(continued)" 같이 여러 참조가
    #  각각 캡션답게 생길 수 있음)
    # [3-13] 여기서는 _looks_like_table_caption(table/표 전용)을 쓴다.
    # 참조가 그림 캡션을 가리키면 direct로 인정하지 않고 아래 bbox
    # fallback으로 넘긴다 — fallback은 여전히 넓은 패턴을 쓰므로 텍스트
    # 자체를 잃지는 않고, confidence만 "direct"→"inferred"로 정직해진다.
    resolved_texts: list[str] = []
    resolved_refs: list[str] = []
    for idx in cap_refs:
        if idx in picture_caption_refs:
            continue  # 구조적으로 그림 캡션 확인됨 -> direct 후보에서 제외
        candidate_text = parsed.texts[idx].text or None
        if not _looks_like_table_caption(candidate_text):
            continue
        caption_page = parsed.texts[idx].page_no
        if caption_page != -1 and caption_page != table_page:
            cross_page = True
        resolved_texts.append(candidate_text.strip())
        resolved_refs.append(str(idx))

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

    fallback_text = _find_caption_by_bbox(parsed, table_page, table_top_y, table_bot_y, picture_caption_refs)
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
        parsed, table_page, table_top_y, table_bot_y, picture_caption_refs
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

    # fallback 4: 표 구조 자체에 병합된 캡션. table.html/cells는 그대로 둔다
    # (위 주석 참고 — cells까지 같이 벗기지 않으면 desync가 생기므로 추출만).
    merged_caption = _find_caption_in_merged_header(table.html)
    if merged_caption:
        return CaptionMapping(
            table_index=table_index,
            page_no=table_page,
            caption_text=merged_caption,
            caption_ref=None,
            confidence="inferred",
            multi_caption=multi_caption,
            cross_page=cross_page,
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


def map_all_captions(parsed: "ParsedDoc") -> list[CaptionMapping]:
    """문서 내 모든 표에 대해 캡션 매핑 수행."""
    return [map_table_caption(parsed, table, table.index) for table in parsed.tables]


def validate_mapping(mappings: list[CaptionMapping]) -> dict:
    """
    1:1 매핑 검증.

    반환:
        total_tables, mapped, unmapped, mapping_rate
        multi_caption_tables : 복수 캡션 참조를 가진 표 인덱스 목록
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
