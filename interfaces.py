"""
interfaces.py

모듈 간 인터페이스 정의.

역할:
    표          → EU로 변환
    텍스트/그림 → Docling HybridChunker

26.06.29 기준
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# LangChain 미설치 환경에서도 import 가능
try:
    from langchain_core.documents import Document as LangChainDocument
except ImportError:
    LangChainDocument = None  # type: ignore


# ---------------------------------------------------------------------------
# EvidenceUnit
# ---------------------------------------------------------------------------

@dataclass
class EvidenceUnit:
    """
    표 + 캡션 + 설명단락을 하나로 묶은 검색 단위.

    EvidenceUnit은 항상 표 EU.

    text property:
        직접 넣는 필드가 아님. 각 필드를 채우면 자동으로 조립.
        임베딩 또는 LangChain에 넘길 시 eu.text 사용.
    """

    # ------------------------------------------------------------------
    # 식별
    # ------------------------------------------------------------------
    eu_id: str          # "eu-p3-0" = 3페이지의 첫 번째 EU
    page_no: int        # 표가 있는 페이지 번호

    # ------------------------------------------------------------------
    # 내용
    # ------------------------------------------------------------------
    caption_text: Optional[str] = None   # 표 제목 (e.g. "Table 3: 지역별 매출")
    table_html: Optional[str] = None     # 표 데이터. Docling table.export_to_html()
    footnote_text: Optional[str] = None  # 표 아래 주석 (e.g. "(단위: 백만 달러)")

    # bbox 거리 + 임베딩 유사도 모두를 통과한 단락만 저장
    # 임계값 미달 단락은 Docling HybridChunker가 처리
    context_before: list[str] = field(default_factory=list)  # 표 위쪽 설명 단락들
    context_after: list[str] = field(default_factory=list)   # 표 아래쪽 설명 단락들

    # ------------------------------------------------------------------
    # 위치
    # ------------------------------------------------------------------
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    # 표 위치 (x1, y1, x2, y2), 0~1 normalized (Docling 기본 좌표계)

    # ------------------------------------------------------------------
    # 분할
    # 512토큰 이하: is_split=False, 나머지 None
    # 512토큰 초과: 행 분할 후 is_split=True, split_index/total_splits 설정
    # ------------------------------------------------------------------
    is_split: bool = False
    split_index: Optional[int] = None   # 0-based (첫 번째 분할 = 0)
    total_splits: Optional[int] = None  # 분할 안 됐으면 None

    # ------------------------------------------------------------------
    # text property (자동 계산되므로 값을 넣지 말 것)
    # ------------------------------------------------------------------
    @property
    def text(self) -> str:
        """
        context_before + caption + table + footnote + context_after 순으로 합침.
        각 필드를 채우면 자동 반영.
        """
        parts = [
            *self.context_before,
            self.caption_text,
            self.table_html,
            self.footnote_text,
            *self.context_after,
        ]
        return "\n".join(p for p in parts if p)