from __future__ import annotations

from dataclasses import dataclass
from bs4 import BeautifulSoup

from interfaces import EvidenceUnit
from token_counter import (
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

    chunks = []

    current_rows = []

    split_idx = 1

    for row in rows:

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

        else:

            chunk_html = build_table(
                header_html,
                current_rows,
            )

            chunks.append(

                EvidenceUnit(
                    eu_id=f"{eu.eu_id}-s{split_idx}",
                    page_no=eu.page_no,
                    caption_text=eu.caption_text,
                    table_html=chunk_html,
                    footnote_text="",
                    context_before=(
                        eu.context_before
                        if split_idx == 1
                        else []
                    ),
                    context_after=[],
                    bbox=eu.bbox,
                    is_split=True,
                    split_index=split_idx,
                    total_splits=None,
                )

            )

            split_idx += 1

            current_rows = [row]

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
                caption_text=eu.caption_text,
                table_html=chunk_html,
                footnote_text=eu.footnote_text,
                context_before=(
                    eu.context_before
                    if split_idx == 1
                    else []
                ),
                context_after=eu.context_after,
                bbox=eu.bbox,
                is_split=True,
                split_index=split_idx,
                total_splits=None,
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