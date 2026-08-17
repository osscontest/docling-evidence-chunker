"""
baseline/snapshot.py

회귀 기준선: EU 산출물 자체를 JSON 스냅샷으로 저장해두고 리팩터링 전후로
diff를 뜬다. recall 수치는 임계값 근처에서 흔들리므로, 산출물 비교가 회귀를
더 정확히 잡는다.
"""
import os
import json

from docling.document_converter import DocumentConverter
from evidence_chunker.chunker import build_evidence_units
from evidence_chunker.parser.docling import DoclingParser
from evidence_chunker.split import split_oversized_units

PDF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "pdfs", "docling_technical_report.pdf"
)

doc = DocumentConverter().convert(PDF_PATH).document
parsed = DoclingParser().from_doc(doc)
eus = split_oversized_units(build_evidence_units(parsed))
json.dump(
    [{"eu_id": e.eu_id, "is_split": e.is_split, "text": e.text, "units": e.retrieval_units} for e in eus],
    open(os.path.join(os.path.dirname(__file__), "eu_snapshot.json"), "w", encoding="utf-8"),
    ensure_ascii=False, indent=2,
)
