"""
baseline_url.py

비교 기준선(Docling HybridChunker 단독)을 arXiv URL에서 PDF를 바로 받아 측정.
로컬 PDF로 같은 측정을 하려면 baseline.py를 쓸 것.
"""
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from sentence_transformers import SentenceTransformer
import numpy as np

# 1. 파싱 & 청킹
converter = DocumentConverter()
result = converter.convert("https://arxiv.org/pdf/2408.09869")
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
    print(f"   정답 청크: {qa['answer_chunk_idx']} | 검색된 청크: {top1_idx} | {'✅' if correct else '❌'}")
    print()

recall_at_1 = hits / len(qa_set)
print(f"=== Docling HybridChunker 단독 Recall@1: {recall_at_1:.2f} ({hits}/{len(qa_set)}) ===")