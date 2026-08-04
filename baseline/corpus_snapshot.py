"""
baseline/corpus_snapshot.py

Stage 3 (3) 리팩터링(chunk()/build_corpus() 분리) 전 코퍼스 출력 스냅샷.

EU 스냅샷(baseline/snapshot.py)은 build_evidence_units() 결과만 덮는다.
HybridChunker 청킹과 filter_consumed_paragraphs()는 어떤 기준선도 커버하지
않는데, 이번 리팩터링이 정확히 그 경로를 건드리므로 먼저 캡처해둔다.

output="langchain"(EU 전체 1개=문서 1개)과 "langchain_units"(EU를
retrieval_units 단위로 쪼갠 것) 둘 다 캡처 — 메타데이터 딕셔너리 구조가
서로 다르므로(하나는 retrieval_text, 하나는 parent_text) 둘 다 diff
기준으로 남겨야 리팩터링 후 정확히 같은지 확인할 수 있다.
"""
import os
import json

from evidence_chunker import EvidenceChunker

PDF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "pdfs", "docling_technical_report.pdf"
)


def dump(output_mode: str, out_name: str) -> None:
    chunker = EvidenceChunker()
    docs = chunker.chunk(PDF_PATH, output=output_mode, include_text=True)
    data = [{"content": d.page_content, "metadata": d.metadata} for d in docs]
    out_path = os.path.join(os.path.dirname(__file__), out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"{output_mode}: {len(data)}개 문서 -> {out_path}")


if __name__ == "__main__":
    dump("langchain", "corpus_snapshot_langchain.json")
    dump("langchain_units", "corpus_snapshot_langchain_units.json")
