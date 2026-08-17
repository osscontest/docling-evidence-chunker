"""
caption.py

캡션 ↔ 표 매핑 알고리즘.

parser.base.TableBlock.caption_refs를 사용해 표와 캡션 텍스트를 1:1로
연결한다. 우선순위(direct → bbox → 인접 페이지 → 병합 헤더 → none)는
map_table_caption() 참고.

parser.base.ParsedDoc/TableBlock 기반 — Docling DoclingDocument를 직접
만지지 않는다. 좌표 비교는 TOPLEFT(parser/base.py 참고) 기준 — "표 위" =
cy가 작을수록 위쪽.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from .parser.base import BBox, ParsedDoc, TableBlock

# 캡션 fallback 탐색 범위(표 위/아래, pt) — 일반 컨텍스트(300pt)보다 좁게,
# 캡션은 보통 표 바로 옆에 있음.
CAPTION_SEARCH_PT = 200.0

# section_header도 후보에 포함 — 캡션이 section_header로 잘못 라벨링되는
# 경우가 있음.
_CANDIDATE_LABEL_NAMES = {"caption", "section_header", "text"}

# "Table 1", "표1", "<표 3>" 등 번호 붙은 캡션 패턴. 부록 번호 체계
# ("Table C.1")도 인식하도록 숫자 앞 문자 하나를 허용한다.
_CAPTION_PATTERN = re.compile(r"(table|figure|fig|표|그림)\s*\.?\s*[A-Z]?\.?\s*\d+", re.IGNORECASE)
_MIN_CAPTION_LEN = 8

# 캡션 패턴은 텍스트 맨 앞 이 글자 수 안에서만 확인한다 — 전체 텍스트를
# 훑으면 본문 중간의 우연한 참조("...as shown in Figure 2.2...")까지
# 캡션으로 오탐한다.
CAPTION_PREFIX_CHARS = 20

# 표가 페이지 위/아래 이 비율 이내에 걸쳐 있을 때만 인접 페이지를 탐색한다
# — 게이팅 없이 인접 페이지를 보면 무관한 옆 페이지 캡션을 잘못 채택할 수 있음.
_ADJACENT_PAGE_EDGE_RATIO = 0.15


@dataclass
class CaptionMapping:
    table_index: int
    page_no: int
    caption_text: Optional[str]
    caption_ref: Optional[str]        # texts 인덱스 문자열. fallback으로 찾은 경우 None
    confidence: Literal["direct", "inferred", "none"]
    multi_caption: bool = False       # 표 하나가 captions 참조 2개 이상 보유
    cross_page: bool = False          # 캡션이 표와 다른 페이지에 위치


def _center_y(bbox: "BBox") -> float:
    """bbox의 수직 중심. TOPLEFT라 값이 작을수록 페이지 위쪽이다."""
    return (bbox.t + bbox.b) / 2.0


def _looks_like_caption(text: Optional[str]) -> bool:
    """번호 붙은 캡션 패턴("Table 1", "<표 3>" 등)인지 확인.

    표/그림 캡션을 모두 인정하는 넓은 패턴 — 그림 캡션까지 후보로 받아
    confidence만 "inferred"로 낮추는 fallback 경로에서 쓴다. 텍스트
    앞부분(CAPTION_PREFIX_CHARS자)만 보는 이유는 그 상수 주석 참고.
    """
    if not text or len(text.strip()) < _MIN_CAPTION_LEN:
        return False
    prefix = text.strip()[:CAPTION_PREFIX_CHARS]
    return bool(_CAPTION_PATTERN.search(prefix))


# direct 참조 검증 전용 — table.captions가 그림(Figure) 캡션을 가리키는
# 경우 "direct"로 잘못 신뢰하지 않기 위해 table/표 패턴만 인정한다
# (그림 캡션을 direct로 확정하면 EvidenceUnit.safe_caption이 걸러내지
# 못하고 EU 전체에 증폭됨). bbox/인접 페이지 fallback은 여전히 넓은
# 패턴(fig/figure/그림 포함)을 써서 confidence만 "inferred"로 낮추고
# 텍스트 자체는 잃지 않는다.
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
            continue

        text = item.text.strip()
        if not _looks_like_caption(text):
            continue

        cy = _center_y(item.bbox)
        if cy < table_top_y:
            dist = table_top_y - cy
        elif cy > table_bot_y:
            dist = cy - table_bot_y
        else:
            dist = 0.0

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

    페이지가 다르면 좌표계(페이지별 y 원점)가 서로 달라 bbox 거리 비교가
    무의미하므로, "이전 페이지에서 가장 아래" / "다음 페이지에서 가장 위"에
    있는 캡션 패턴 텍스트를 채택한다.

    Returns:
        (caption_text, found) — 못 찾거나 게이팅에 걸리면 (None, False)
    """
    page_height = parsed.page_sizes.get(table_page, (0.0, 0.0))[1]
    if not page_height:
        return None, False

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
            continue
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


# fallback 4: Docling이 캡션을 별도 텍스트 아이템이 아니라 표 자체의 첫 행
# (병합된 th/td 하나)으로 렌더링하는 경우가 있다 — 위 세 fallback은
# parsed.texts만 보므로 이 경우를 원천적으로 못 본다.
#
# table.html/cells는 손대지 않는다 — 둘 다 같은 Docling 표에서 독립적으로
# 계산되므로, 한쪽(html)만 캡션 행을 지우면 cells는 그대로 남아 flatten.py/
# split.py의 행 인덱스가 어긋난다. 대가는 EU.text에 캡션이 표 안에도 한 번
# 더 남는 경미한 토큰 중복뿐 — 정확성 문제는 없다.

_MERGED_HEADER_PATTERN = re.compile(
    r'<t[hd][^>]*colspan="(\d+)"[^>]*>\s*(.*?)\s*</t[hd]>',
    re.IGNORECASE | re.DOTALL,
)


def _find_caption_in_merged_header(table_html: Optional[str]) -> Optional[str]:
    """
    표 첫 행이 다수 컬럼을 병합한 th/td 하나로만 이루어져 있고, 그 안 텍스트가
    캡션 패턴(_looks_like_table_caption)에 맞으면 채택.

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
