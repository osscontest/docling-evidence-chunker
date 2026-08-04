"""
parser/base.py

Docling과 무관한 내부 문서 모델 + 파서 Protocol.

flatten.py/caption.py/context.py/filters.py가 Docling의 DoclingDocument를
직접 만지지 않고 이 모델(ParsedDoc/TextBlock/TableBlock)만 알게 하려는 목적.
tests/test_caption_exceptions.py의 FakeItem이 이미 최소 프로토콜(label,
text, page_no, bbox 네 개)을 실증하고 있었음 — TextBlock이 그 형태 그대로다.

핵심 3가지:
  - 좌표계를 파서 경계에서 TOPLEFT로 정규화. Docling 원본은 BOTTOMLEFT
    (t > b, 시각적 위쪽 = 큰 y)인데, context.py/caption.py가 각자 "BOTTOMLEFT
    좌표계: 시각적으로 표 위 = cy > table_top_y" 같은 주석으로 이 가정을
    반복 재서술하고 있었다. 여기서 한 번만 뒤집어서 이후로는 "시각적으로
    위쪽 = 작은 y"라는, 화면 좌표계에 익숙한 사람 누구나 맞는 직관을 쓴다.
    (알고리즘 쪽 부등호 반전은 별도 커밋에서, bbox flip 테스트로 먼저 잠근
    다음에 진행 — 여기서는 파서 구현만 한다.)
  - cref 문자열("#/texts/3")을 정수 인덱스로. caption.py의 resolve_ref()
    같은 파싱 코드가 사라지고, self_ref 문자열 대조(그림 캡션 구조적 판별)도
    정수 비교가 된다.
  - 라벨을 자체 Enum으로 축소. 실제로 쓰는 5종(text/list_item/paragraph/
    section_header/caption)만 남긴다 — 이 5종 외에는 알고리즘 어디서도
    라벨 문자열을 비교하지 않음을 grep으로 확인함.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple, Optional, Protocol


class BlockLabel(str, Enum):
    TEXT = "text"
    LIST_ITEM = "list_item"
    PARAGRAPH = "paragraph"
    SECTION_HEADER = "section_header"
    CAPTION = "caption"
    OTHER = "other"  # 위 5종 외 Docling 라벨(picture, table 등)은 전부 여기로


class BBox(NamedTuple):
    """항상 TOPLEFT 정규화: 시각적으로 위쪽일수록 t(top)가 작다."""
    l: float
    t: float
    r: float
    b: float


@dataclass
class TextBlock:
    index: int          # ParsedDoc.texts 안에서의 위치. cref/self_ref를 대체하는 정수 식별자.
    text: str
    label: BlockLabel
    page_no: int
    bbox: BBox


@dataclass
class TableBlock:
    index: int                          # ParsedDoc.tables 안에서의 위치.
    page_no: int
    bbox: BBox
    cells: list[dict]                   # Docling table_cells 형태 그대로 유지
                                         # (column_header/row_header/start_row_offset_idx/
                                         # end_row_offset_idx/start_col_offset_idx/
                                         # end_col_offset_idx/text). flatten.py가 이미
                                         # 이 dict 형태로만 동작하므로 재설계 범위 밖.
    num_rows: int
    num_cols: int
    html: Optional[str]                 # export_to_html() 결과, 파싱 시점에 미리 계산
    caption_refs: list[int] = field(default_factory=list)   # texts 인덱스
    footnote_refs: list[int] = field(default_factory=list)  # texts 인덱스

    def num_data_rows(self) -> int:
        """헤더 행을 제외한 데이터 행 개수. export_to_dataframe().shape[0]과
        같은 기준 — last_column_values()는 빈 텍스트 행을 걸러내서 개수가
        줄어들 수 있으므로(마지막 열이 빈 행이 있는 경우), min_rows류
        게이트에는 이쪽을 쓸 것.

        행 단위로 헤더를 판정한다(그 행에 column_header 셀이 하나라도
        있으면 헤더 행) — 셀 단위로 판정하면 다단 헤더 표의 좌상단 모서리
        셀(예: "CPU")이 column_header=False로 찍힌 경우를 데이터 행으로
        잘못 셈(실측: docling_technical_report.pdf table[0]에서 헤더 2행 중
        1행이 이렇게 새서 3으로 잘못 나왔고, pandas export_to_dataframe()는
        2였음 — 셀 단위 판정 대신 행 단위로 바꿔서 일치시킴).
        """
        all_rows: dict[int, bool] = {}
        for c in self.cells:
            row = c.get("start_row_offset_idx")
            if row is None:
                continue
            all_rows[row] = all_rows.get(row, False) or bool(c.get("column_header"))
        return sum(1 for is_header in all_rows.values() if not is_header)

    def last_column_values(self) -> list[str]:
        """마지막 열의 셀 값(행 순서). filters.is_toc_or_lof_decoy 전용.

        Docling의 export_to_dataframe()(pandas 의존) 대신 cells에서 직접
        뽑는다 — 이 메서드가 파서 추상화의 유일한 소비처라, pandas가 알고리즘
        레이어에 새는 걸 막는다.

        헤더 셀(column_header=True)은 제외 — export_to_dataframe()가 첫 행을
        컬럼명으로 쓰고 데이터에서 빼는 것과 동일한 기준.
        """
        last_col = self.num_cols - 1
        by_row: dict[int, str] = {}
        for cell in self.cells:
            if cell.get("column_header"):
                continue
            if cell.get("start_col_offset_idx", -1) != last_col:
                continue
            row = cell.get("start_row_offset_idx", -1)
            text = (cell.get("text") or "").strip()
            if text:
                by_row[row] = text
        return [by_row[r] for r in sorted(by_row)]


@dataclass
class ParsedDoc:
    texts: list[TextBlock]
    tables: list[TableBlock]
    picture_caption_refs: set[int]              # doc.pictures[*].captions -> texts 인덱스 집합
    page_sizes: dict[int, tuple[float, float]]  # page_no -> (width, height)


class PdfParser(Protocol):
    def parse(self, path: str) -> ParsedDoc: ...
