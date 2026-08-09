"""
interfaces.py

모듈 간 인터페이스 정의.

역할:
    표          → EU로 변환
    텍스트/그림 → Docling HybridChunker

26.08.01 기준
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
    eu_id: str          # "{doc_id}-p3-0" = doc_id 문서의 3페이지 첫 번째 EU
    page_no: int        # 표가 있는 페이지 번호
    doc_id: str = ""    # 문서 식별자. 기본은 파일명 stem(build_evidence_units 참고).
                         # PDF 여러 개를 한 인덱스에 넣을 때 eu_id 충돌(예: 서로 다른
                         # 문서의 "eu-p3-0")을 막기 위해 eu_id 접두사로 들어간다.

    # [fix] page_no 는 표 자신의 페이지 하나뿐이라, context_before/after 가
    # 인접 페이지(table_page-1/+1)에서 끌어온 경우 그 사실이 어디에도 안 남았다
    # — 평가 스크립트가 "정답 페이지를 포함하는가"를 판정할 때 이 EU가 실제로
    # 걸친 모든 페이지를 몰라서 인접 페이지 오차를 전부 오답 처리하던 문제의
    # 원인. attach_context_paragraphs()가 채운다(기본값은 {page_no}).
    page_span: set[int] = field(default_factory=set)

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
    # Docling 원본은 PDF 포인트(BOTTOMLEFT) → geometry.normalize_bbox()로 변환해서 저장
    #
    # 주의: 패키지 내부(parser.base.BBox, caption.py/context.py가 쓰는 좌표)는
    # Stage 4부터 TOPLEFT로 정규화돼 있다 — 이 필드(EvidenceUnit.bbox)만
    # 예외적으로 BOTTOMLEFT를 유지한다(공개 API 필드라 기존 소비자의 계약을
    # 바꾸지 않기 위한 의도적 결정). 패키지 안에 좌표계가 두 개 공존하니,
    # parser.base.BBox 값을 그대로 이 필드에 넣지 말 것 — 반드시
    # geometry.normalize_bbox()를 거칠 것.

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
    # [3-11] 그림(Fig) 캡션 오탐 방어용 헬퍼 프로퍼티
    # ------------------------------------------------------------------
    # caption_confidence(direct/inferred)로는 못 걸러낸다 — ieee1.pdf 실측에서
    # Docling이 "direct"로도 그림 캡션을 표에 구조적으로 잘못 연결하는
    # 케이스가 확인됨 (eu-p21-0-s1). confidence(출처)가 아니라 캡션
    # 텍스트 자체의 패턴(의미)으로 걸러야 안전하다.
    #
    # [8/3] 1차 방어선은 scripts/_caption_mapper.py의 _get_picture_caption_refs()로
    # 옮겼다 — doc.pictures[*].captions와 구조적으로 대조해서, 실제로 어느
    # PictureItem의 캡션으로 지정된 텍스트인지를 텍스트 내용과 무관하게(언어
    # 상관없이) 판별한다. 여기 있는 fig/figure/그림 텍스트 패턴 매칭은 그
    # 구조적 체크가 놓칠 수 있는 경우(예: doc.pictures가 비어있는 등) 대비한
    # 2차 안전망으로 남겨둠 — 비용이 문자열 비교 한 줄이라 지우는 것보다
    # 남겨두는 쪽의 손해가 없음. -> 하지만 추후 논의 필요
    @property
    def safe_caption(self) -> str | None:
        """캡션이 Fig/Figure/그림으로 시작하면 표 정체성으로 신뢰하지 않고 None.

        [8/3] 1차 방어선 아님 -- _caption_mapper.py의 구조적 체크(그림 아이템의
        captions RefItem과 직접 대조)가 1차. 이건 그게 놓칠 경우를 위한 2차 백업.
        """
        if not self.caption_text:
            return None
        lower_cap = self.caption_text.strip().lower()
        if lower_cap.startswith(("fig", "figure", "그림")):
            return None
        return self.caption_text

    @property
    def safe_abstract(self) -> str | None:
        """table_abstract는 caption_text로 조립되므로, 캡션이 오염됐다면 같이 제거."""
        if not self.table_abstract:
            return None
        if self.caption_text and not self.safe_caption:
            return self.table_abstract.replace(self.caption_text, "").strip()
        return self.table_abstract

    # ------------------------------------------------------------------
    # [3-12] 강등된(Downgrade) 그림 캡션을 문맥으로 복구하는 헬퍼
    # ------------------------------------------------------------------
    # safe_caption이 캡션 자격을 박탈(None)하면 표 정체성 오염(3-11에서 막은
    # prefix 증폭)은 재발하지 않지만, 캡션 안에 들어있던 진짜 유효한 키워드
    # (예: "NWPU-RESISC45")까지 통째로 사라지는 부작용이 ieee5.pdf 실측에서
    # 확인됨. 대신 이 텍스트를 "표의 정체성"이 아니라 "표 주변의 평범한
    # 문맥 단락 1개"로 강등시켜 context_before에 편입한다 — prefix로 모든
    # 하위 유닛에 반복 주입되지 않고, 독립된 유닛 1개로만 존재하므로 ieee1
    # 같은 무관 캡션이 다시 표 전체를 오염시킬 위험은 없다.
    @property
    def context_before_with_fallback(self) -> list[str]:
        """context_before + safe_caption에서 탈락한 그림 캡션(문맥으로 강등, 맨 앞 삽입)."""
        fallback = list(self.context_before)
        if self.caption_text and not self.safe_caption:
            # 캡션 자격은 잃었지만 주변 텍스트로서의 가치는 있으므로 문맥으로 강등.
            # 3-2의 positional bias 대응 설계(캡션/context_before를 맨 앞에 배치)를
            # 따라 리스트 맨 앞에 삽입한다.
            fallback.insert(0, self.caption_text)
        return fallback

    # ------------------------------------------------------------------
    # text property (자동 계산되므로 값을 넣지 말 것)
    # ------------------------------------------------------------------
    @property
    def text(self) -> str:
        """
        [3-2] 조립 순서 (positional bias 대응):
            caption_text      ← 표의 정체성 (Table 3: 매출...)
            context_before    ← 표를 설명하는 핵심 문맥
            table_abstract    ← 표 요약
            section_header
            table_html        ← 표 구조 원본 (LLM 컨텍스트용)
            flattened_rows    ← 셀별 자연어 문장
            footnote_text
            context_after

        임베딩 모델(all-MiniLM-L6-v2 등)이 앞부분 토큰에 더 크게 반응하는
        경향(positional bias)을 감안해, 표의 정체성(caption)과 이를
        설명하는 핵심 문맥(context_before)을 맨 앞에 배치한다.
        """
        parts = [
            self.safe_caption,
            *self.context_before_with_fallback,
            self.safe_abstract,
            self.section_header,
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
        (W4 Recall@1 회귀 0.60 -> 0.40의 핵심 원인이었음.)

        table_html은 제외하고, [3-2] 신호 밀도 + positional bias를 함께
        고려한 순서로 배치:
            caption_text -> context_before -> table_abstract
            -> section_header -> flattened_rows -> footnote_text
            -> context_after
        LLM에 넘길 전체 맥락(table_html 포함)이 필요하면 text를 사용할 것.
        """
        parts = [
            self.safe_caption,
            *self.context_before_with_fallback,
            self.safe_abstract,
            self.section_header,
            *self.flattened_rows,
            self.footnote_text,
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

        [3-2] Symmetric Contextual Chunking:
        context_dependent 유형에서 EU가 hybrid 청크에 지는 현상(3-4에서
        확인, 3-1 dedup 적용 후에도 갭 -9.9pp 잔존)의 원인으로 지목된
        부분. flattened_rows/footnote_text/context_before/context_after가
        전부 캡션 없이 순수 텍스트로만 떠 있어서, 벡터 공간에서 "이게 어느
        표 얘기인지" 정체성을 잃는 문제가 있었다.

        해결: 각 하위 유닛의 독립성(문단/문장 단위로 별도 벡터화)은 그대로
        유지하면서, 모든 하위 유닛 앞에 캡션을 접두사로 주입해 표 정체성을
        앵커링한다(Anthropic Contextual Retrieval과 동일 패턴).
        context_before/context_after 모두 대칭적으로 동일 처리한다 — 한쪽만
        summary에 흡수시키는 안은 그 문단 자체가 정답인 질문에서 손해를 볼
        수 있어 리뷰 과정에서 폐기하고, 독립 유닛 + prefix 방식으로 확정.

        요약 정보(캡션+표 요약+섹션헤더)는 "이 표가 뭘 담고 있는가" 같은
        광범위 질의용으로 별도 유닛 1개로 묶는다 (prefix 없음 — 캡션이
        이미 본문에 포함돼 있음).

        벡터스토어 구성 시: 각 유닛을 임베딩하고 검색 결과의 unit이 어느
        EU(eu_id)에 속하는지만 기록해뒀다가, 매칭되면 해당 EU의 eu.text를
        반환하면 된다 (langchain_wrapper.eu_to_langchain_units 참고).
        """
        units: list[str] = []

        # 1. 요약 유닛: "이 표가 뭘 담고 있는가" 같은 광범위 질의용
        summary = "\n".join(
            p for p in (self.safe_caption, self.safe_abstract, self.section_header) if p
        )
        if summary:
            units.append(summary)

        # 2. 하위 유닛 전부에 캡션 접두사 주입 (표 정체성 앵커링).
        # safe_caption이 없으면(캡션 없음, 또는 Fig/Figure/그림으로 시작하는
        # 오탐 캡션이라 걸러짐) prefix는 빈 문자열 — "[None] "이나 잘못된
        # 그림 캡션이 모든 하위 유닛에 증폭되는 것(3-11) 둘 다 방지.
        prefix = f"[{self.safe_caption}] " if self.safe_caption else ""

        # context_before: 독립 유닛 유지 + 대칭적 앵커링
        # [3-12] safe_caption에서 탈락한 그림 캡션이 있으면 여기 맨 앞에
        # 문맥 유닛 1개로 포함됨(prefix 없이) — 표 전체 오염 없이 키워드만 복구.
        units.extend(f"{prefix}{para}" for para in self.context_before_with_fallback)

        # 개별 행(Row): 어느 표의 데이터인지 캡션으로 명시
        units.extend(f"{prefix}{row}" for row in self.flattened_rows)

        if self.footnote_text:
            units.append(f"{prefix}{self.footnote_text}")

        # context_after: context_before와 대칭적으로 동일 처리
        units.extend(f"{prefix}{para}" for para in self.context_after)

        if not units:
            fallback = self.retrieval_text
            if fallback:
                units.append(fallback)

        return units

    # ------------------------------------------------------------------
    # RetrievalChunk 프로토콜 (export.RetrievalChunk 참고)
    #
    # build_corpus()가 EU와 HybridChunker 유래 일반 청크(export.TextChunk)를
    # 같은 리스트에 섞어 반환하므로, 두 타입이 같은 속성 집합(chunk_id/text/
    # retrieval_text/retrieval_units/metadata)을 노출해야 to_langchain() 등
    # export 함수가 isinstance 분기 없이 균일하게 동작한다.
    # ------------------------------------------------------------------
    @property
    def chunk_id(self) -> str:
        return self.eu_id

    @property
    def is_atomic(self) -> bool:
        """export.to_*_units()가 이 chunk를 retrieval_units 단위로 쪼갤지 판단하는 신호.

        EvidenceUnit은 항상 쪼갤 수 있는 대상이라 False — retrieval_units가
        우연히 1개뿐이어도(작은 표 등) 예전부터 "쪼개는 취급"이었으므로
        그 동작을 그대로 유지한다. export.TextChunk(표와 무관한 일반 청크,
        그 자체로 최소 단위)만 True.
        """
        return False

    @property
    def metadata(self) -> dict:
        """LangChain/LlamaIndex 문서 메타데이터 (retrieval_text 포함, whole-chunk 모드용).

        export.to_langchain_units() 등 exploded 모드는 이 dict에서
        retrieval_text를 빼고 parent_text로 대체해서 쓴다.
        """
        return {
            # [fix] max-pool 재랭킹(export.langchain/llamaindex.dedupe_by_chunk_id)이
            # 같은 부모 청크에서 나온 검색 결과를 그룹핑할 키. eu_id와 값은
            # 같지만(self.chunk_id == self.eu_id) export.TextChunk 쪽과 동일한
            # 필드명으로 통일해야 두 타입을 isinstance 분기 없이 균일하게
            # dedupe할 수 있다 — eu_id만 쓰면 TextChunk.metadata["eu_id"]가
            # 항상 None이라 서로 다른 TextChunk들이 전부 한 그룹으로 잘못
            # 묶인다(실측: Benchmark.md §06).
            "chunk_id": self.chunk_id,
            "eu_id": self.eu_id,
            "doc_id": self.doc_id,
            "page_no": self.page_no,
            # page_span 이 비어있으면(attach_context_paragraphs 안 거친 경로 등)
            # page_no 하나짜리로 안전하게 fallback — 호출자가 매번 None 체크 안 해도 됨.
            "page_span": sorted(self.page_span) if self.page_span else [self.page_no],
            "section_header": self.section_header,
            "caption_text": self.caption_text,
            "bbox": list(self.bbox),
            "is_split": self.is_split,
            "split_index": self.split_index,
            "total_splits": self.total_splits,
            "caption_confidence": self.caption_confidence,
            "retrieval_text": self.retrieval_text,
        }