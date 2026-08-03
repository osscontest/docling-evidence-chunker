"""
baseline/snapshot.py

Stage 0 회귀 기준선: EU 산출물 자체를 스냅샷으로 저장.
recall 수치보다 이쪽이 회귀를 더 정확히 잡는다.
"""
import os
import json
import importlib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from docling.document_converter import DocumentConverter

PDF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "pdfs", "docling_technical_report.pdf"
)

doc = DocumentConverter().convert(PDF_PATH).document
m = importlib.import_module("06_build_eu")
eus = m.split_oversized_units(m.build_evidence_units(doc))
json.dump(
    [{"eu_id": e.eu_id, "is_split": e.is_split, "text": e.text, "units": e.retrieval_units} for e in eus],
    open(os.path.join(os.path.dirname(__file__), "eu_snapshot.json"), "w", encoding="utf-8"),
    ensure_ascii=False, indent=2,
)
