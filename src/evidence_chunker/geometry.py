"""
geometry.py

bbox 좌표 변환 유틸리티.

unit.py는 EvidenceUnit 데이터클래스 정의만 담당하고, 좌표 변환 로직은
여기로 분리 (export/ 패키지가 LangChain/LlamaIndex 변환 로직을 분리한 것과
동일한 패턴).
"""

from __future__ import annotations


def normalize_bbox(
    bbox: dict,
    page_width: float,
    page_height: float,
) -> tuple[float, float, float, float]:
    """
    Docling PDF 포인트 bbox (BOTTOMLEFT origin) → 0~1 normalized (x1,y1,x2,y2).

    Docling bbox 키: l(left), t(top), r(right), b(bottom), coord_origin=BOTTOMLEFT
    BOTTOMLEFT에서 t > b (top이 y축 더 큼).

    반환값은 BOTTOMLEFT 유지, 0~1로만 스케일 축소:
        x1 = l / width
        y1 = b / height   (시각적 하단, 작은 y)
        x2 = r / width
        y2 = t / height   (시각적 상단, 큰 y)
    """
    l = bbox.get("l", 0.0)
    t = bbox.get("t", 0.0)
    r = bbox.get("r", 0.0)
    b = bbox.get("b", 0.0)
    return (
        l / page_width,
        b / page_height,
        r / page_width,
        t / page_height,
    )
