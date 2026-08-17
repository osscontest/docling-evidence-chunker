"""
baseline.py

비교 기준선: Docling HybridChunker 단독으로 청킹했을 때의 Recall@1.

baseline_url.py와 동일한 로직이며, arXiv에서 내려받는 대신 로컬 PDF
(data/pdfs/docling_technical_report.pdf)를 쓰고 Windows 콘솔 UTF-8 출력만
추가했다. EU 파이프라인 쪽 수치는 measure_recall_single.py에서 측정한다.
"""
import sys
import os

if hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from sentence_transformers import SentenceTransformer
import numpy as np

PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pdfs", "docling_technical_report.pdf")

# 1. 파싱 & 청킹
converter = DocumentConverter()
result = converter.convert(PDF_PATH)
doc = result.document

chunker = HybridChunker()
chunks = list(chunker.chunk(doc))
chunk_texts = [c.text for c in chunks]

# 2. 임베딩
model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = model.encode(chunk_texts, normalize_embeddings=True)

# 3. Q&A 셋 (질문, 정답이 있어야 할 청크 인덱스)
qa_set = [
    {
        "question": "What is the TTS of Intel Xeon with 4 threads using native backend?",
        "answer_chunk_idx": 18,  # 실제 숫자가 있는 청크
    },
    {
        "question": "What is the runtime characteristics table about?",
        "answer_chunk_idx": 17,  # 캡션 청크
    },
    {
        "question": "What is the mAP of YOLOv5 on DocLayNet?",
        "answer_chunk_idx": 33,  # Table 2 데이터
    },
    {
        "question": "Which models were used for baseline experiments on DocLayNet?",
        "answer_chunk_idx": 32,  # Table 2 캡션
    },
    {
        "question": "What are the pages per second for pypdfium backend with 16 threads?",
        "answer_chunk_idx": 18,  # 실제 숫자
    },
]

# 4. Recall@1 측정
hits = 0
for qa in qa_set:
    q_emb = model.encode(qa["question"], normalize_embeddings=True)
    scores = np.dot(chunk_embeddings, q_emb)
    top1_idx = int(np.argmax(scores))

    correct = top1_idx == qa["answer_chunk_idx"]
    hits += int(correct)

    print(f"Q: {qa['question']}")
    print(f"   정답 청크: {qa['answer_chunk_idx']} | 검색된 청크: {top1_idx} | {'OK' if correct else 'MISS'}")
    print()

recall_at_1 = hits / len(qa_set)
print(f"=== Docling HybridChunker 단독 Recall@1: {recall_at_1:.2f} ({hits}/{len(qa_set)}) ===")
