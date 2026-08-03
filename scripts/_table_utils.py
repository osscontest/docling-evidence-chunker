"""
_table_utils.py

표 처리 유틸리티 모음.

기능:
  - build_col_header_map  : 다단/병합 헤더 열 매핑 (col_span 처리)
  - build_row_header_map  : 행헤더 행 매핑 (row_span 처리)
  - infer_headers_fallback: Docling이 헤더 태깅 못 했을 때 휴리스틱 추론
  - flatten_to_sentences  : 셀 → "행헤더 | 열헤더: 값" 구조화 문자열 (임베딩용)
                            [8/3] 한글 문법 문장에서 언어 무관 key:value로 변경
  - detect_cell_marker    : 셀 내 각주 마커(*†‡) 감지
  - build_table_abstract  : 표 전체 요약 문자열 (multi-granularity 검색용)
  - find_duplicate_tables : Docling이 표 1개를 TableItem 2개로 중복 감지하는
                            케이스 탐지 (W4 Recall@1 회귀 원인 중 하나)
  - is_toc_or_lof_decoy   : 목차(ToC)/그림·표 목록(LoF)이 표로 오인식된 경우 감지
                            (v03 p3 필터. 실측 근거: context_dependent_maxpooling_실험.md 11절)
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional

# 각주 마커 패턴 (유니코드 포함)
_MARKER_RE = re.compile(r'[*†‡※#]')


# ---------------------------------------------------------------------------
# 1. 열헤더 맵 구성 (다단/병합 처리)
# ---------------------------------------------------------------------------

def build_col_header_map(cells: list[dict], num_cols: int) -> dict[int, str]:
    """
    column_header=True 셀을 수집해 col_index → 헤더 텍스트 매핑 반환.
    col_span > 1 이면 해당 범위 전체에 텍스트 기록.
    다단 헤더(여러 행에 걸친 헤더)는 ' / '로 이어붙임.

    예:
        상위 헤더 "분기" (col_span=2) + 하위 헤더 "Q1", "Q2"
        → col_map = {0: "분기 / Q1", 1: "분기 / Q2"}
    """
    col_map: dict[int, str] = {}

    for cell in cells:
        if not cell.get("column_header"):
            continue
        text = cell.get("text", "").strip()
        if not text:
            continue
        c_start = cell.get("start_col_offset_idx", 0)
        c_end   = cell.get("end_col_offset_idx", c_start + 1)

        for col in range(c_start, min(c_end, num_cols)):
            if col in col_map:
                col_map[col] = col_map[col] + " / " + text
            else:
                col_map[col] = text

    return col_map


# ---------------------------------------------------------------------------
# 2. 행헤더 맵 구성 (row_span 처리)
# ---------------------------------------------------------------------------

def build_row_header_map(cells: list[dict], num_rows: int) -> dict[int, str]:
    """
    row_header=True 셀을 수집해 row_index → 헤더 텍스트 매핑 반환.
    row_span > 1 이면 해당 범위 전체 행에 동일 텍스트 기록.

    예:
        "APAC" (row_span=3) → rows 2, 3, 4 모두 row_map[r] = "APAC"
    """
    row_map: dict[int, str] = {}

    for cell in cells:
        if not cell.get("row_header"):
            continue
        text = cell.get("text", "").strip()
        if not text:
            continue
        r_start = cell.get("start_row_offset_idx", 0)
        r_end   = cell.get("end_row_offset_idx", r_start + 1)

        for row in range(r_start, min(r_end, num_rows)):
            if row not in row_map:
                row_map[row] = text

    return row_map


# ---------------------------------------------------------------------------
# 3. 헤더 추론 폴백
# ---------------------------------------------------------------------------

def infer_headers_fallback(cells: list[dict], num_cols: int) -> dict[int, str]:
    """
    column_header 태깅이 전혀 없을 때 첫 번째 행을 헤더로 추정.
    첫 행이 전부 숫자면 Col0, Col1, ... 로 대체.
    """
    first_row = sorted(
        [c for c in cells if c.get("start_row_offset_idx", -1) == 0],
        key=lambda x: x.get("start_col_offset_idx", 0),
    )

    # 첫 행이 전부 숫자패턴이면 헤더로 쓰기 부적합
    def _is_numeric(text: str) -> bool:
        return bool(re.match(r'^[\d\s.,+\-±%()]+$', text.strip()))

    col_map: dict[int, str] = {}
    for cell in first_row:
        col = cell.get("start_col_offset_idx", 0)
        text = cell.get("text", "").strip()
        col_map[col] = text if (text and not _is_numeric(text)) else f"Col{col}"

    return col_map


# ---------------------------------------------------------------------------
# 4. 각주 마커 감지
# ---------------------------------------------------------------------------

def detect_cell_marker(text: str) -> tuple[str, Optional[str]]:
    """
    셀 값에서 각주 마커(*†‡ 등)를 분리.
    반환: (cleaned_text, marker_or_None)

    예: "100*" → ("100", "*")
        "O(n²)" → ("O(n²)", None)
    """
    m = _MARKER_RE.search(text)
    if m:
        cleaned = _MARKER_RE.sub("", text).strip()
        return cleaned, m.group()
    return text.strip(), None


# ---------------------------------------------------------------------------
# 5. 행 플래트닝 (핵심)
# ---------------------------------------------------------------------------

def group_sentences_by_row(
    cells: list[dict],
    num_rows: int,
    num_cols: int,
    footnote_text: Optional[str] = None,
) -> dict[int, list[str]]:
    """
    표 셀 데이터를 자연어 문장으로 변환하되, 원본 표의 행 인덱스
    (start_row_offset_idx)별로 묶어서 반환.

    table_splitter가 표를 행 단위로 분할할 때, 분할된 조각에 해당 행의
    flattened_rows 문장만 함께 실어 보내기 위해 사용한다. 그렇지 않으면
    분할된 EU는 자연어 문장 없이 raw table_html만 남아 임베딩 검색 신호를
    잃는다 (retrieval_units가 표 요약 1개 유닛으로 쪼그라듦).

    처리 순서 및 문장 포맷은 flatten_to_sentences() 참고.

    Returns:
        {row_offset_idx: [문장, ...]}  — 헤더 행은 포함되지 않음
    """
    col_map = build_col_header_map(cells, num_cols)
    row_map = build_row_header_map(cells, num_rows)

    # 열헤더 없으면 폴백
    if not col_map:
        col_map = infer_headers_fallback(cells, num_cols)

    grouped: dict[int, list[str]] = defaultdict(list)

    for cell in cells:
        # 헤더 셀은 문장화 대상 아님
        if cell.get("column_header") or cell.get("row_header"):
            continue

        raw_value = cell.get("text", "").strip()
        if not raw_value:
            continue

        row = cell.get("start_row_offset_idx", 0)
        col = cell.get("start_col_offset_idx", 0)

        value, marker = detect_cell_marker(raw_value)
        if not value:
            continue

        col_header = col_map.get(col, f"Col{col}")
        row_header = row_map.get(row)

        # [8/3 재정리] 기존엔 "{row}의 {col}는 {value}이다." 한글 문법 문장을
        # if/else로 케이스(row_header 있음/없음, marker+footnote/marker만)
        # 별로 다른 템플릿 문자열을 골라 조립했음. "문장을 만들 필요가
        # 있는가"라는 지적을 받고 재검토 -- 애초에 문장(어느 언어든 문법)이
        # 필요한 게 아니라 각 셀을 검색 가능한 짧은 단위로 쪼개는 게 목적
        # (W4 회귀 노트: 256토큰 잘림 문제 해결용). 그러면 "케이스별로 다른
        # 템플릿을 고르는" 대신, build_table_abstract()가 이미 쓰던 "있는
        # 값만 모아서 이어붙이기" 패턴으로 통일하는 게 맞다 -- 분기가 아니라
        # 데이터가 있고 없고의 문제이므로 리스트 필터링으로 표현.
        label_parts = [p for p in (row_header, col_header) if p]
        sentence = f"{' | '.join(label_parts)}: {value}"
        if marker:
            note = footnote_text.strip() if footnote_text else marker
            sentence += f" [{note}]"

        grouped[row].append(sentence)

    return dict(grouped)


def flatten_to_sentences(
    cells: list[dict],
    num_rows: int,
    num_cols: int,
    footnote_text: Optional[str] = None,
) -> list[str]:
    """
    표 셀 데이터를 자연어 문장 목록으로 변환 (행 순서대로 이어붙임).
    행별로 묶인 결과가 필요하면 group_sentences_by_row() 사용.
    """
    grouped = group_sentences_by_row(cells, num_rows, num_cols, footnote_text)
    return [sentence for row in sorted(grouped) for sentence in grouped[row]]


# ---------------------------------------------------------------------------
# 6. 표 단위 요약 (Table Abstract)
# ---------------------------------------------------------------------------

def build_table_abstract(
    caption_text: Optional[str],
    col_map: dict[int, str],
    num_rows: int,
    section_header: Optional[str] = None,
) -> str:
    """
    LLM 없이 규칙 기반으로 표 요약 생성.
    광범위 질의("지역별 매출 표가 어디 있어?")에 대한 검색용.

    포맷 (8/3 개정, 한글 라벨 -> 언어 무관 구조 라벨):
        "[섹션헤더] > [캡션]. Columns: [열헤더 목록]. [N] data row(s)."
    """
    parts: list[str] = []

    if section_header:
        parts.append(f"[{section_header}]")

    if caption_text:
        parts.append(caption_text)

    # [8/3] "열: "/"총 N행 데이터." 한글 라벨 제거 -- group_sentences_by_row()와
    # 동일한 이유(영어 PDF + 영어 중심 임베딩 모델). table_about 유형(열 이름/
    # 행 개수를 묻는 질문)이 이 table_abstract에서 답을 찾는 경우가 많아서
    # flattened_rows 쪽만 고치면 이 유형은 여전히 안 고쳐진 상태로 남았을 것.
    cols = [v for v in col_map.values() if v]
    if cols:
        parts.append("Columns: " + ", ".join(cols[:6]) + ("..." if len(cols) > 6 else ""))

    data_rows = max(0, num_rows - 1)  # 헤더 행 제외
    if data_rows > 0:
        parts.append(f"{data_rows} data row(s).")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# 7. 중복 표 감지 (W4 Recall@1 회귀 원인 #2)
# ---------------------------------------------------------------------------

_DUP_TOKEN_RE = re.compile(r"[.,()%]+$")


def _cell_text_signature(cells: list[dict]) -> set[str]:
    """셀 텍스트를 정규화된 토큰 집합으로 변환 (중복 표 비교용)."""
    tokens: set[str] = set()
    for cell in cells:
        text = (cell.get("text") or "").strip()
        if not text:
            continue
        for line in re.split(r"[\n,]+", text):
            for tok in line.split():
                tok = _DUP_TOKEN_RE.sub("", tok).lower()
                if len(tok) >= 2:
                    tokens.add(tok)
    return tokens


def find_duplicate_tables(doc, overlap_ratio: float = 0.6) -> dict[int, int]:
    """
    같은 페이지 내 표 쌍의 셀 텍스트 중복도로 중복 TableItem 감지.

    실사례(docling 기술보고서 8페이지): 같은 물리적 표를 Docling이
    TableItem 2개로 나눠 인식함 — 하나는 행이 뭉개진 채(1~2행) 캡션
    RefItem과 연결되어 있고, 다른 하나는 행이 올바르게 분리(예: 13행)됐지만
    캡션이 없음. 두 표 모두 EU로 만들면 코퍼스가 중복 표 조각으로
    오염되고, 정작 제대로 구조화된 표는 캡션을 잃는다.

    셀 텍스트 토큰 집합의 포함비율(containment = |교집합| / min(|A|,|B|))이
    overlap_ratio 이상이면 같은 표로 간주하고, 행 수가 더 많은(더 세밀하게
    구조화된) 쪽만 남긴다.

    Returns:
        {제거할 표의 doc.tables 인덱스: 남길 표의 doc.tables 인덱스}
    """
    pages: dict[int, list[int]] = defaultdict(list)
    for i, table in enumerate(doc.tables):
        d = table.model_dump()
        prov = d.get("prov", [])
        if not prov:
            continue
        pages[prov[0].get("page_no", -1)].append(i)

    signatures: dict[int, set[str]] = {}
    row_counts: dict[int, int] = {}
    for i, table in enumerate(doc.tables):
        d = table.model_dump()
        data = d.get("data", {})
        signatures[i] = _cell_text_signature(data.get("table_cells", []))
        row_counts[i] = data.get("num_rows", 0)

    drop_map: dict[int, int] = {}
    for idxs in pages.values():
        if len(idxs) < 2:
            continue
        for a_pos in range(len(idxs)):
            for b_pos in range(a_pos + 1, len(idxs)):
                a, b = idxs[a_pos], idxs[b_pos]
                if a in drop_map or b in drop_map:
                    continue
                sig_a, sig_b = signatures[a], signatures[b]
                if not sig_a or not sig_b:
                    continue
                containment = len(sig_a & sig_b) / min(len(sig_a), len(sig_b))
                if containment >= overlap_ratio:
                    loser, winner = (
                        (a, b) if row_counts[a] < row_counts[b] else (b, a)
                    )
                    drop_map[loser] = winner

    return drop_map


# ---------------------------------------------------------------------------
# 8. p3 필터 — ToC/그림·표 목록 오탐 감지
# ---------------------------------------------------------------------------

P3_MAX_PAGE = 5
P3_NUMERIC_RATIO_THRESHOLD = 0.7
P3_MIN_ROWS = 3
P3_HEADER_KEYWORDS = ("content", "list of table", "list of figure")


def _has_toc_like_header(doc, page_no: int) -> bool:
    """표 바로 앞(같은 페이지 또는 이전 페이지)에 ToC/LoF 계열 헤더가 있는지."""
    for item in doc.texts:
        d = item.model_dump()
        if d.get("label") != "section_header":
            continue
        prov = d.get("prov", [])
        if not prov:
            continue
        pg = prov[0].get("page_no")
        if pg not in (page_no, page_no - 1):
            continue
        text = d.get("text", "").strip().lower()
        if any(kw in text for kw in P3_HEADER_KEYWORDS):
            return True
    return False


def is_toc_or_lof_decoy(
    doc,
    table,
    max_page: int = P3_MAX_PAGE,
    numeric_ratio_threshold: float = P3_NUMERIC_RATIO_THRESHOLD,
    min_rows: int = P3_MIN_ROWS,
) -> bool:
    """목차(ToC)나 그림/표 목록(LoF)이 표로 오인식된 경우 True.

    이 표는 build_evidence_units()의 EU 생성 대상에서 제외해야 한다
    (baseline=HybridChunker 청킹에는 영향 주지 않음 — 호출자가
    doc.tables를 필터링 전후로 원복해서 사용할 것).
    """
    prov = table.model_dump().get("prov", [])
    if not prov:
        return False
    page_no = prov[0].get("page_no")
    if page_no is None or page_no > max_page:
        return False

    try:
        df = table.export_to_dataframe(doc)
    except TypeError:
        df = table.export_to_dataframe()
    except Exception:
        return False

    if df.shape[0] < min_rows or df.shape[1] == 0:
        return False

    last_col_values = [str(v).strip() for v in df.iloc[:, -1] if str(v).strip()]
    if not last_col_values:
        return False

    numeric_ratio = sum(1 for v in last_col_values if v.isdigit()) / len(last_col_values)
    if numeric_ratio < numeric_ratio_threshold:
        return False

    return _has_toc_like_header(doc, page_no)
