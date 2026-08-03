"""
세션 전체에서 테스트 PDF 파싱을 정확히 한 번만 수행.

같은 프로세스 안에서 Docling 파싱을 반복하면 메모리가 누적돼
std::bad_alloc이 나는 문제가 있음 (README "큰 PDF 처리 시 메모리 부족"
절 참고). 이 픽스처가 없으면 각 테스트가 개별적으로 파싱하면서 같은
문제를 반복 재현하게 된다.
"""
import os

import pytest

SAMPLE_PDF = os.path.join(
    os.path.dirname(__file__), "..", "data", "pdfs", "docling_technical_report.pdf"
)


@pytest.fixture(scope="session")
def sample_doc():
    if not os.path.exists(SAMPLE_PDF):
        pytest.skip("테스트 PDF 없음 (benchmarks/tools/download_test_pdfs.py로 받을 것)")

    from docling.document_converter import DocumentConverter

    return DocumentConverter().convert(SAMPLE_PDF).document
