"""
SmartChunker.chunk()가 evidence_chunker.chunker.build_evidence_units()와
정확히 같은 EU를 낸다는 것을 증명한다 — 두 경로가 다시 갈라지는 걸 막는
회귀 테스트 (Stage 2 "빌더 통합"의 실행 가능한 증거).

baseline/snapshot.py는 scripts/06_build_eu.py를 직접 import해서 돌기
때문에, chunker.py 안에 build_evidence_units()를 옮겨 붙여도 그 스크립트는
계속 통과한다 — SmartChunker가 실제로 같은 코드를 타는지는 별도로
검증해야 한다.

주의: split_oversized_units가 아직 chunk()에 연결되기 전이므로 지금은
완전히 같아야 한다. 분할을 연결하는 커밋에서 chunk() 쪽 결과가 더 잘게
쪼개지도록 이 테스트도 함께 갱신할 것.
"""
import os

from docling.document_converter import DocumentConverter

from evidence_chunker import SmartChunker
from evidence_chunker.chunker import build_evidence_units

SAMPLE_PDF = os.path.join(
    os.path.dirname(__file__), "..", "data", "pdfs", "docling_technical_report.pdf"
)


def test_chunk_matches_reference_builder():
    doc = DocumentConverter().convert(SAMPLE_PDF).document

    via_chunker = SmartChunker().chunk(SAMPLE_PDF)
    via_direct = build_evidence_units(doc)

    assert [e.text for e in via_chunker] == [e.text for e in via_direct]
