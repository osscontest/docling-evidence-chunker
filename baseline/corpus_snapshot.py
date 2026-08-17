"""
baseline/corpus_snapshot.py

코퍼스(build_corpus() + to_langchain()/to_langchain_units()) 출력 스냅샷.

EU 스냅샷(baseline/snapshot.py)은 build_evidence_units() 결과만 덮는다.
HybridChunker 청킹과 filter_consumed_paragraphs()는 어떤 기준선도 커버하지
않으므로 별도로 캡처해둔다.

to_langchain()(EU/일반 청크 전체 1개=문서 1개)과 to_langchain_units()
(EU를 retrieval_units 단위로 쪼갠 것) 둘 다 캡처 — 메타데이터 딕셔너리
구조가 서로 다르므로(하나는 retrieval_text, 하나는 parent_text) 둘 다
diff 기준으로 남겨야 리팩터링 후 정확히 같은지 확인할 수 있다.

커밋된 baseline/corpus_snapshot_langchain*.json이 비교 기준이며, 이
스크립트로 다시 생성해 diff를 뜨는 방식으로 쓴다.
"""
import os
import json

from evidence_chunker import EvidenceChunker
from evidence_chunker.export.langchain import to_langchain, to_langchain_units

PDF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "pdfs", "docling_technical_report.pdf"
)


def dump(chunks, to_fn, out_name: str) -> None:
    docs = to_fn(chunks)
    data = [{"content": d.page_content, "metadata": d.metadata} for d in docs]
    out_path = os.path.join(os.path.dirname(__file__), out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"{to_fn.__name__}: {len(data)}개 문서 -> {out_path}")


if __name__ == "__main__":
    # build_corpus()는 한 번만 — PDF 파싱을 두 번 하면 같은 프로세스 안에서
    # 메모리 누적으로 std::bad_alloc이 날 수 있음(README 참고).
    chunks = EvidenceChunker().build_corpus(PDF_PATH)
    dump(chunks, to_langchain, "corpus_snapshot_langchain.json")
    dump(chunks, to_langchain_units, "corpus_snapshot_langchain_units.json")
