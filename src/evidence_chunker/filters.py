"""
filters.py

EU 생성 대상에서 표를 제외해야 하는 케이스 감지.

기능:
  - find_duplicate_tables : Docling이 표 1개를 TableItem 2개로 중복 감지하는
                            케이스 탐지 (W4 Recall@1 회귀 원인 중 하나)
  - is_toc_or_lof_decoy   : 목차(ToC)/그림·표 목록(LoF)이 표로 오인식된 경우 감지
                            (v03 p3 필터. 실측 근거: context_dependent_maxpooling_실험.md 11절)
"""
from __future__ import annotations

import re
from collections import defaultdict

# ---------------------------------------------------------------------------
# 1. 중복 표 감지 (W4 Recall@1 회귀 원인 #2)
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
# 2. p3 필터 — ToC/그림·표 목록 오탐 감지
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
