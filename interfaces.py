"""
interfaces.py

모듈 간 인터페이스 정의.

역할:
    표          → EU로 변환
    텍스트/그림 → Docling HybridChunker

26.07.06 기준
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

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
    # 행별 자연어 문장 맵 (내부 전용 — table_splitter가 표를 행 단위로
    # 분할할 때 각 조각에 해당 행의 flattened_rows만 함께 실어 보내기 위한
    # 빌더 내부 데이터. {원본 표의 row_offset_idx: [문장, ...]}
    # _table_utils.group_sentences_by_row()로 생성. text/retrieval_*
    # property에는 관여하지 않음 — flattened_rows만 참조한다.
    # ------------------------------------------------------------------
    row_sentence_map: dict[int, list[str]] = field(default_factory=dict, repr=False)

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
    # Docling 원본은 PDF 포인트(BOTTOMLEFT) → bbox_utils.normalize_bbox()로 변환해서 저장

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
    # "direct"  : captions RefItem으로 직접 연결된 캡션
    # "inferred": bbox 거리로 추정한 캡션
    # "none"    : 캡션 없음
    # 평가 시 confidence별 오답 분포 분석에 활용
    # ------------------------------------------------------------------
    caption_confidence: Literal["direct", "inferred", "none"] = "none"

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
    # retrieval_text property (임베딩 검색 전용, 자동 계산)
    # ------------------------------------------------------------------
    @property
    def retrieval_text(self) -> str:
        """
        임베딩 기반 검색(코사인 유사도)에 넣을 축약 텍스트.

        all-MiniLM-L6-v2 같은 임베딩 모델은 입력을 256 토큰에서 잘라버린다.
        text property는 raw table_html(태그로 뒤덮인 마크업, 토큰 낭비가 큼)이
        flattened_rows(자연어 문장, 실제 검색 신호)보다 앞에 오기 때문에,
        표가 조금만 커도 truncate 시 flattened_rows 전체가 잘려나가
        정작 검색에 필요한 내용이 임베딩에 반영되지 않는 문제가 있었다
        (W4 Recall@1 회귀 0.60 -> 0.40의 핵심 원인).

        table_html은 제외하고, 신호 밀도가 높은 순서로 배치:
            caption_text -> table_abstract -> section_header
            -> flattened_rows -> footnote_text -> context_before/after
        LLM에 넘길 전체 맥락(table_html 포함)이 필요하면 text를 사용할 것.
        """
        parts = [
            self.caption_text,
            self.table_abstract,
            self.section_header,
            *self.flattened_rows,
            self.footnote_text,
            *self.context_before,
            *self.context_after,
        ]
        return "\n".join(p for p in parts if p)

    # ------------------------------------------------------------------
    # retrieval_units property (행 단위 다중 벡터 검색용, 자동 계산)
    # ------------------------------------------------------------------
    @property
    def retrieval_units(self) -> list[str]:
        """
        문장 단위 다중 벡터(multi-vector) 검색용 세부 텍스트 목록.

        retrieval_text 하나로 표 전체(모든 행)를 한 벡터에 뭉치면, "표 안의
        특정 셀 값"을 묻는 질의(예: 13행짜리 표에서 YOLOv5 행 하나)가 나머지
        행 데이터에 묻혀 코사인 유사도에서 밀리는 문제가 있다. 대신
        flattened_rows 문장 하나하나를 별도 벡터로 인덱싱하고, 어느 문장이
        매칭되든 이 EU 전체(caption_text/table_html 포함, eu.text)를
        반환하는 "small-to-big" 패턴에 쓰기 위한 단위 목록.

        요약 정보(캡션+표 요약+섹션헤더)는 "이 표가 뭘 담고 있는가" 같은
        광범위 질의용으로 별도 유닛 1개로 묶는다.

        벡터스토어 구성 시: 각 유닛을 임베딩하고 검색 결과의 unit이 어느
        EU(eu_id)에 속하는지만 기록해뒀다가, 매칭되면 해당 EU의 eu.text를
        반환하면 된다 (langchain_wrapper.eu_to_langchain_units 참고).
        """
        units: list[str] = []

        summary = "\n".join(
            p for p in (self.caption_text, self.table_abstract, self.section_header) if p
        )
        if summary:
            units.append(summary)

        units.extend(self.flattened_rows)

        if self.footnote_text:
            units.append(self.footnote_text)

        # context_before/after도 문단 단위로 각각 별도 유닛화. "이 표 위/아래에
        # 뭐라고 써있는지"를 묻는 질의는 표 데이터가 아니라 이 문단 자체가
        # 정답이므로, flattened_rows와 뭉쳐서 하나로 임베딩하면 신호가 흐려진다.
        units.extend(self.context_before)
        units.extend(self.context_after)

        if not units:
            fallback = self.retrieval_text
            if fallback:
                units.append(fallback)

        return units