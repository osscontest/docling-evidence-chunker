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

W3 예정 (여기서는 처리하지 않음):
    - 다음 페이지 캡션 케이스 실제 연결 (지금은 cross_page 플래그로 감지만 함)
    - 복수 캡션 병합 처리
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
# 실제 버그 사례("(단위: 1), %)" 같은 파편)를 걸러내는 핵심 필터.
_CAPTION_PATTERN = re.compile(r"(table|figure|fig|표|그림)\s*\.?\s*\d+", re.IGNORECASE)
_MIN_CAPTION_LEN = 8


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


def _center_y(bbox: dict) -> float:
    return (bbox.get("t", 0.0) + bbox.get("b", 0.0)) / 2.0


def _looks_like_caption(text: Optional[str]) -> bool:
    """
    "Table 1", "<표 3>" 같은 번호 붙은 캡션 패턴인지 확인.
    실사례: RefItem이 "(단위: 1), %)" 같은 각주 파편을 가리키는 버그를 걸러내기 위함.
    """
    if not text or len(text.strip()) < _MIN_CAPTION_LEN:
        return False
    return bool(_CAPTION_PATTERN.search(text))


def _find_caption_by_bbox(
    doc, table_page: int, table_top_y: float, table_bot_y: float
) -> Optional[str]:
    """
    captions RefItem이 없거나 품질 미달일 때, 같은 페이지에서 bbox 거리 기준으로
    캡션처럼 보이는 텍스트를 찾는 fallback.
    """
    candidates: list[tuple[float, str]] = []

    for item in doc.texts:
        d = item.model_dump()
        if d.get("label") not in _CANDIDATE_LABELS:
            continue
        prov_list = d.get("prov", [])
        if not prov_list or prov_list[0].get("page_no", -1) != table_page:
            continue

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


def map_table_caption(doc, table, table_index: int) -> CaptionMapping:
    """
    단일 표에 대한 캡션 매핑 (1:1).

    우선순위:
      1. captions RefItem이 가리키는 텍스트가 캡션답게 생겼으면 그대로 채택 (direct)
      2. RefItem이 없거나 파편을 가리키면 bbox 거리로 캡션 재탐색 (inferred)
      3. 둘 다 실패하면 캡션 없음 (none)
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
    direct_text: Optional[str] = None
    direct_ref: Optional[str] = None

    if cap_refs:
        primary_cref = _cref_of(cap_refs[0])
        cap_dict = resolve_ref(doc, primary_cref)
        candidate_text = cap_dict.get("text") or None
        caption_page = _get_prov_page(cap_dict)
        cross_page = caption_page != -1 and caption_page != table_page

        if _looks_like_caption(candidate_text):
            direct_text, direct_ref = candidate_text, primary_cref

    if direct_text:
        return CaptionMapping(
            table_index=table_index,
            page_no=table_page,
            caption_text=direct_text,
            caption_ref=direct_ref,
            confidence="direct",
            multi_caption=multi_caption,
            cross_page=cross_page,
        )

    fallback_text = _find_caption_by_bbox(doc, table_page, table_top_y, table_bot_y)
    if fallback_text:
        return CaptionMapping(
            table_index=table_index,
            page_no=table_page,
            caption_text=fallback_text,
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
