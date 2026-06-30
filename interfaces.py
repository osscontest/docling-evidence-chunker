"""
interfaces.py

모듈 간 인터페이스 정의.

역할:
    표          → EU로 변환
    텍스트/그림 → Docling HybridChunker

26.07.01 기준
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
    section_header: Optional[str] = None # 표가 속한 섹션 제목 (e.g. "2.2. 실험 결과")
    caption_text: Optional[str] = None   # 표 제목 (e.g. "Table 3: 지역별 매출")
    table_html: Optional[str] = None     # 표 데이터. Docling table.export_to_html()
    footnote_text: Optional[str] = None  # 표 아래 주석 (e.g. "(단위: 백만 달러)")

    # bbox 거리 + 임베딩 유사도 모두를 통과한 단락만 저장
    # 임계값 미달 단락은 Docling HybridChunker가 처리
    context_before: list[str] = field(default_factory=list)  # 표 위쪽 설명 단락들
    context_after: list[str] = field(default_factory=list)   # 표 아래쪽 설명 단락들

    # ------------------------------------------------------------------
    # 행 플래트닝 (Row Flattening)
    # 각 데이터 셀을 "행헤더의 열헤더는 값이다." 형태 자연어 문장으로 변환.
    # 임베딩 모델이 표 구조보다 자연어에 더 잘 반응 → Recall@1 향상 핵심.
    # _table_utils.flatten_to_sentences()로 생성.
    # ------------------------------------------------------------------
    flattened_rows: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # 표 단위 요약 (Table Abstract)
    # "이 표가 뭘 담고 있는가"에 답하는 한 줄 요약.
    # 광범위 질의("지역별 매출 표가 어디 있어?")에는 abstract가 더 강하게 반응.
    # LLM 없이 caption + 헤더 조합으로 자동 생성.
    # ------------------------------------------------------------------
    table_abstract: Optional[str] = None

    # ------------------------------------------------------------------
    # 위치
    # ------------------------------------------------------------------
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    # 표 위치 (x1, y1, x2, y2), 0~1 normalized
    # Docling 원본은 PDF 포인트(BOTTOMLEFT) → normalize_bbox()로 변환해서 저장

    # ------------------------------------------------------------------
    # 분할
    # 512토큰 이하: is_split=False, 나머지 None
    # 512토큰 초과: 행 분할 후 is_split=True, split_index/total_splits 설정
    # ------------------------------------------------------------------
    is_split: bool = False
    split_index: Optional[int] = None   # 0-based (첫 번째 분할 = 0)
    total_splits: Optional[int] = None  # 분할 안 됐으면 None

    # ------------------------------------------------------------------
    # 신뢰도 메타데이터
    # "high": captions RefItem으로 직접 연결된 캡션
    # "low" : bbox 거리로 추정했거나 캡션 없음
    # 평가 시 low EU에서 오답 집중 여부 분석에 활용
    # ------------------------------------------------------------------
    caption_confidence: str = "high"

    # ------------------------------------------------------------------
    # text property (자동 계산되므로 값을 넣지 말 것)
    # ------------------------------------------------------------------
    @property
    def text(self) -> str:
        """
        조립 순서:
            section_header
            context_before
            caption_text
            table_abstract    ← 표 요약 (광범위 질의용)
            table_html        ← 표 구조
            flattened_rows    ← 셀별 자연어 문장 (구체 질의용)
            footnote_text
            context_after
        """
        parts = [
            self.section_header,
            *self.context_before,
            self.caption_text,
            self.table_abstract,
            self.table_html,
            *self.flattened_rows,
            self.footnote_text,
            *self.context_after,
        ]
        return "\n".join(p for p in parts if p)

    # ------------------------------------------------------------------
    # LangChain 변환 (선택)
    # ------------------------------------------------------------------
    def to_langchain_document(self) -> "LangChainDocument":
        if LangChainDocument is None:
            raise ImportError("langchain_core is not installed. pip install langchain-core")
        return LangChainDocument(
            page_content=self.text,
            metadata={
                "eu_id": self.eu_id,
                "page_no": self.page_no,
                "section_header": self.section_header,
                "caption_text": self.caption_text,
                "bbox": list(self.bbox),
                "is_split": self.is_split,
                "split_index": self.split_index,
                "total_splits": self.total_splits,
                "caption_confidence": self.caption_confidence,
            },
        )


# ---------------------------------------------------------------------------
# bbox 변환 유틸
# ---------------------------------------------------------------------------

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
