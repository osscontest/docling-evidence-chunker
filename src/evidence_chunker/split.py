from __future__ import annotations

from dataclasses import dataclass
from bs4 import BeautifulSoup

from .unit import EvidenceUnit
from .tokens import (
    count_tokens,
    exceeds_token_limit,
    DEFAULT_TOKEN_LIMIT,
)

SPLIT_LIMIT = DEFAULT_TOKEN_LIMIT
LLM_SUMMARY_LIMIT = DEFAULT_TOKEN_LIMIT * 6


@dataclass
class SplitResult:
    strategy: str
    chunks: list[EvidenceUnit]
    original_eu_id: str
    total_tokens: int


# ----------------------------
# HTML Parsing
# ----------------------------

def parse_table(table_html: str):
    """
    Returns
    -------
    header_html : str
    body_rows : list[str]
    """

    soup = BeautifulSoup(table_html, "html.parser")

    table = soup.find("table")
    if table is None:
        return "", []

    # -------- header --------

    thead = table.find("thead")

    if thead:
        header_html = str(thead)
    else:
        rows = table.find_all("tr", recursive=True)

        if not rows:
            return "", []

        header_html = str(rows[0])

    # -------- body rows --------

    tbody = table.find("tbody")

    if tbody:
        body_rows = [str(r) for r in tbody.find_all("tr", recursive=False)]
    else:
        rows = table.find_all("tr")

        if len(rows) <= 1:
            body_rows = []
        else:
            body_rows = [str(r) for r in rows[1:]]

    return header_html, body_rows


def build_table(header_html, rows):

    if header_html.startswith("<thead"):
        return (
            "<table>"
            + header_html
            + "<tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )

    return (
        "<table>"
        + header_html
        + "".join(rows)
        + "</table>"
    )


def estimate_chunk_tokens(caption, html, footnote=""):

    text = "\n".join(
        x for x in [
            caption,
            html,
            footnote,
        ] if x
    )

    return count_tokens(text)


# ----------------------------
# Main
# ----------------------------

def split_eu(
    eu: EvidenceUnit,
    limit=SPLIT_LIMIT,
):

    total = count_tokens(eu.text)

    # -----------------
    # Stage 1
    # -----------------

    if total <= limit:

        return SplitResult(
            strategy="single",
            chunks=[eu],
            original_eu_id=eu.eu_id,
            total_tokens=total,
        )

    # -----------------
    # Stage 3
    # -----------------

    if total > LLM_SUMMARY_LIMIT:

        return SplitResult(
            strategy="llm_summary",
            chunks=[eu],
            original_eu_id=eu.eu_id,
            total_tokens=total,
        )

    # -----------------
    # Stage 2
    # -----------------

    header_html, rows = parse_table(eu.table_html or "")

    if len(rows) == 0:

        return SplitResult(
            strategy="single",
            chunks=[eu],
            original_eu_id=eu.eu_id,
            total_tokens=total,
        )

    # 헤더가 차지하는 원본 행 개수. row_sentence_map의 키(start_row_offset_idx)는
    # 헤더 행까지 포함해서 매겨지므로, body_rows의 0-based 위치를 원본 행
    # 인덱스로 되돌리려면 이 오프셋을 더해야 한다 (단일 헤더 행 가정 —
    # parse_table()도 동일하게 가정함).
    header_row_offset = max(1, header_html.count("<tr"))

    def _flattened_for(positions: list[int]) -> list[str]:
        """분할 조각에 포함된 원본 행들의 flattened_rows 문장만 추출."""
        return [
            sentence
            for pos in positions
            for sentence in eu.row_sentence_map.get(pos + header_row_offset, [])
        ]

    chunks = []

    current_rows = []
    current_positions: list[int] = []

    split_idx = 1

    for pos, row in enumerate(rows):

        # 행 하나만으로도 limit 초과하면 요약 권고

        single_html = build_table(header_html, [row])

        if estimate_chunk_tokens(
            eu.caption_text,
            single_html,
            eu.footnote_text,
        ) > limit:

            return SplitResult(
                strategy="llm_summary",
                chunks=[eu],
                original_eu_id=eu.eu_id,
                total_tokens=total,
            )

        candidate_rows = current_rows + [row]

        candidate_html = build_table(
            header_html,
            candidate_rows,
        )

        token = estimate_chunk_tokens(
            eu.caption_text,
            candidate_html,
        )

        if token <= limit:

            current_rows.append(row)
            current_positions.append(pos)

        else:

            chunk_html = build_table(
                header_html,
                current_rows,
            )

            chunks.append(

                EvidenceUnit(
                    eu_id=f"{eu.eu_id}-s{split_idx}",
                    page_no=eu.page_no,
                    doc_id=eu.doc_id,
                    section_header=eu.section_header,
                    caption_text=eu.caption_text,
                    table_html=chunk_html,
                    footnote_text="",
                    # [fix] 결정4-A(문맥 단락은 모든 분할 조각에 전부 반복)대로 수정.
                    # 기존엔 context_before가 s1에만, context_after는 아예 어느 비-마지막
                    # 조각에도 안 들어갔음 — context_dependent 유형에서 recall은 맞는데
                    # (정답 표는 찾음) EM만 실패하는 현상의 주 원인으로 실측 확인됨.
                    # dev20 A/B 검증 완료(260808): 악화 0건 / 개선 5건, cell_value 손해 없음
                    # (예전에 "cell_value -2.5pp 손해"로 폐기됐다는 기록이 있었으나, 지금
                    # 코드베이스로 재검증한 결과 재현 안 됨 — 그 기록은 stale로 판단).
                    context_before=list(eu.context_before),
                    context_after=list(eu.context_after),
                    page_span=set(eu.page_span),  # [fix] 부모 EU의 page_span 그대로 전파
                    flattened_rows=_flattened_for(current_positions),
                    table_abstract=eu.table_abstract,
                    bbox=eu.bbox,
                    is_split=True,
                    split_index=split_idx,
                    total_splits=None,
                    caption_confidence=eu.caption_confidence,
                )

            )

            split_idx += 1

            current_rows = [row]
            current_positions = [pos]

    # 마지막 chunk

    if current_rows:

        chunk_html = build_table(
            header_html,
            current_rows,
        )

        chunks.append(

            EvidenceUnit(
                eu_id=f"{eu.eu_id}-s{split_idx}",
                page_no=eu.page_no,
                doc_id=eu.doc_id,
                section_header=eu.section_header,
                caption_text=eu.caption_text,
                table_html=chunk_html,
                footnote_text=eu.footnote_text,
                # [fix] 위와 동일 — 마지막 조각도 split_idx==1(=단일 조각)일 때만
                # context_before를 받던 걸 항상 받도록 수정. dev20 A/B 검증 완료(260808).
                context_before=list(eu.context_before),
                context_after=list(eu.context_after),
                flattened_rows=_flattened_for(current_positions),
                table_abstract=eu.table_abstract,
                bbox=eu.bbox,
                is_split=True,
                split_index=split_idx,
                total_splits=None,
                caption_confidence=eu.caption_confidence,
            )

        )

    total_chunks = len(chunks)

    for c in chunks:
        c.total_splits = total_chunks

    return SplitResult(
        strategy="row_split",
        chunks=chunks,
        original_eu_id=eu.eu_id,
        total_tokens=total,
    )


def split_oversized_units(
    eu_list: list[EvidenceUnit],
    stats: dict | None = None,
) -> list[EvidenceUnit]:
    """
    512토큰(DEFAULT_TOKEN_LIMIT) 초과 EU를 split_eu()로 행 단위 분할.
    한도 이내 EU는 그대로 통과.

    stats를 넘기면 전략별 건수와 분할 규모(split_eus / split_chunks)를
    채워준다. 문맥 배분을 바꾸면 조각 수가 크게 흔들리므로(위 주석 참고)
    그 폭을 관측하기 위한 계측. 동작에는 영향 없다.
    """
    result = []
    for eu in eu_list:
        r = split_eu(eu)
        if stats is not None:
            stats[f"strategy:{r.strategy}"] = stats.get(f"strategy:{r.strategy}", 0) + 1
            if r.strategy == "row_split":
                stats["split_eus"] = stats.get("split_eus", 0) + 1
                stats["split_chunks"] = stats.get("split_chunks", 0) + len(r.chunks)
        result.extend(r.chunks)
    return result


if __name__ == "__main__":
    # 단계 1: 작은 EU (단일 반환)
    eu_small = EvidenceUnit(
        eu_id="eu-p1-0",
        page_no=1,
        caption_text="Table 1: Small table",
        table_html="<table><tr><th>CPU</th><th>TTS</th></tr><tr><td>Xeon</td><td>375s</td></tr></table>",
    )

    # 단계 2: 행 분할 필요
    rows = "".join(
        f"<tr><td>Model {i}</td><td>{i*10} tokens</td><td>{'text ' * 20}</td></tr>"
        for i in range(30)
    )
    eu_large = EvidenceUnit(
        eu_id="eu-p2-0",
        page_no=2,
        caption_text="Table 2: Large benchmark results",
        table_html=f"<table><thead><tr><th>Model</th><th>Tokens</th><th>Description</th></tr></thead><tbody>{rows}</tbody></table>",
        context_before=["This table shows benchmark results."],
        context_after=["Results indicate significant improvement."],
    )

    # 단계 3: LLM 요약 필요
    rows_huge = "".join(
        f"<tr><td>Model {i}</td><td>{'very long description ' * 15}</td></tr>"
        for i in range(100)
    )
    eu_huge = EvidenceUnit(
        eu_id="eu-p3-0",
        page_no=3,
        caption_text="Table 3: Huge table",
        table_html=f"<table><thead><tr><th>Model</th><th>Description</th></tr></thead><tbody>{rows_huge}</tbody></table>",
    )

    for eu in [eu_small, eu_large, eu_huge]:
        result = split_eu(eu)
        print(f"\n{'='*50}")
        print(f"EU: {eu.eu_id} | 전체 토큰: {result.total_tokens}")
        print(f"전략: {result.strategy} | 분할 수: {len(result.chunks)}")
        for c in result.chunks:
            print(f"  └ {c.eu_id} | split_index: {c.split_index}/{c.total_splits} | 토큰: {count_tokens(c.text)}")