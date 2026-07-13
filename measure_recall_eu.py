"""
measure_recall_eu.py

W4: 3모듈 통합 파이프라인(caption_mapper + context_attacher + table_splitter)의
Recall@1을 baseline_recall_local.py(Docling HybridChunker 단독, 0.60)와 비교 측정.

파이프라인:
    PDF --DocumentConverter()--> doc (baseline과 동일 컨버터/옵션)
      +-- doc.tables --> build_evidence_units() --> [table_splitter.split_eu()] --> EU 청크
      +-- HybridChunker(doc) --> 청크 중 표 관련(doc_item.label == "table") 청크는 EU로 대체되므로 제외
    두 그룹을 합쳐 최종 검색 코퍼스 구성 후 baseline과 동일 질문으로 Recall@1 측정.

표 청크 판별 기준(= baseline과 동일 기준으로 고정):
    chunk.meta.doc_items 중 하나라도 label == DocItemLabel.TABLE 이면 "표 청크"로 간주하고 제외.
    (Docling은 표 캡션도 표 데이터와 함께 label=table로 직렬화하므로 캡션 중복도 함께 제거됨)

Usage:
    python measure_recall_eu.py             # split 적용 (기본 파이프라인)
    python measure_recall_eu.py --no-split   # table_splitter 끄고 측정 (ablation)
"""
import sys
import os
import importlib

if hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
sys.path.insert(0, os.path.dirname(__file__))

PDF_PATH = os.path.join(os.path.dirname(__file__), "data", "pdfs", "docling_technical_report.pdf")

APPLY_SPLIT = "--no-split" not in sys.argv


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def main():
    from docling.document_converter import DocumentConverter
    from docling.chunking import HybridChunker
    from docling_core.types.doc import DocItemLabel
    from sentence_transformers import SentenceTransformer
    import numpy as np

    build_eu_mod = importlib.import_module("06_build_eu")

    # ------------------------------------------------------------------
    # 1. 파싱 (baseline과 동일한 기본 DocumentConverter — 로컬 모델 경로를 쓰는
    #    make_converter()는 이 PDF의 7페이지에서 std::bad_alloc으로 Table 2를
    #    통째로 날려버려서 사용하지 않음. README "큰 PDF 메모리 이슈" 참고)
    # ------------------------------------------------------------------
    section("PDF 파싱")
    converter = DocumentConverter()
    result = converter.convert(PDF_PATH)
    doc = result.document
    print(f"  파싱 완료. 표 {len(doc.tables)}개 감지")

    # ------------------------------------------------------------------
    # 2. EvidenceUnit 빌드 (+ 옵션에 따라 table_splitter 적용)
    # ------------------------------------------------------------------
    section("EvidenceUnit 빌드")
    eu_list = build_eu_mod.build_evidence_units(doc)
    print(f"  원본 표 EU: {len(eu_list)}개")

    if APPLY_SPLIT:
        eu_list = build_eu_mod.split_oversized_units(eu_list)
        n_split = sum(1 for eu in eu_list if eu.is_split)
        print(f"  분할(table_splitter) 적용 후: {len(eu_list)}개 ({n_split}개는 분할 조각)")
    else:
        print(f"  분할(table_splitter) 미적용 (--no-split)")

    for eu in eu_list:
        print(f"    [{eu.eu_id}] p{eu.page_no} caption={ (eu.caption_text or '')[:50]!r}")

    # ------------------------------------------------------------------
    # 3. HybridChunker 청크에서 표 관련 청크 제외 (EU로 대체되는 부분)
    # ------------------------------------------------------------------
    section("HybridChunker 청크 (표 청크 제외)")
    chunker = HybridChunker()
    all_chunks = list(chunker.chunk(doc))

    def is_table_chunk(chunk) -> bool:
        return any(di.label == DocItemLabel.TABLE for di in chunk.meta.doc_items)

    table_chunk_idxs = [i for i, c in enumerate(all_chunks) if is_table_chunk(c)]
    non_table_chunks = [c for c in all_chunks if not is_table_chunk(c)]
    print(f"  전체 HybridChunker 청크: {len(all_chunks)}개")
    print(f"  표 관련(제외): {len(table_chunk_idxs)}개 -> {table_chunk_idxs}")
    print(f"  유지(비-표): {len(non_table_chunks)}개")

    # ------------------------------------------------------------------
    # 4. 최종 코퍼스 구성: EU 청크 + 비-표 HybridChunker 청크
    # ------------------------------------------------------------------
    section("최종 코퍼스")
    corpus_texts: list[str] = []
    corpus_sources: list[str] = []

    for eu in eu_list:
        corpus_texts.append(eu.text)
        corpus_sources.append(eu.eu_id)

    for c in non_table_chunks:
        corpus_texts.append(c.text)
        corpus_sources.append("hybrid")

    print(f"  총 {len(corpus_texts)}개 청크 (EU {len(eu_list)}개 + 비-표 HybridChunker {len(non_table_chunks)}개)")

    # ------------------------------------------------------------------
    # 5. 임베딩 + Recall@1 (baseline_recall_local.py와 동일 질문/모델)
    # ------------------------------------------------------------------
    section("Recall@1 측정")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus_embeddings = model.encode(corpus_texts, normalize_embeddings=True)

    # baseline qa_set과 동일 질문. 정답은 청크 인덱스가 아니라
    # "정답 EU가 속한 표(eu_id 접두사)"로 정의 — EU 파이프라인에서는
    # 캡션+데이터가 한 청크로 합쳐지므로 baseline처럼 인덱스로 고정할 수 없음.
    #
    # 주의: Docling이 8페이지에서 Table 2를 TableItem 2개로 중복 감지함
    # (eu-p8-0: 캡션만 있고 실제 데이터 없는 표, eu-p8-1: mAP 숫자가 실제로 들어있는 표).
    # 정답 판정은 실제 데이터가 있는 eu-p8-1 기준으로 함.
    qa_set = [
        {
            "question": "What is the TTS of Intel Xeon with 4 threads using native backend?",
            "answer_eu_prefix": "eu-p5-0",  # Table 1
        },
        {
            "question": "What is the runtime characteristics table about?",
            "answer_eu_prefix": "eu-p5-0",  # Table 1
        },
        {
            "question": "What is the mAP of YOLOv5 on DocLayNet?",
            "answer_eu_prefix": "eu-p8-1",  # Table 2 (실제 데이터가 있는 TableItem)
        },
        {
            "question": "Which models were used for baseline experiments on DocLayNet?",
            "answer_eu_prefix": "eu-p8-1",  # Table 2 (실제 데이터가 있는 TableItem)
        },
        {
            "question": "What are the pages per second for pypdfium backend with 16 threads?",
            "answer_eu_prefix": "eu-p5-0",  # Table 1
        },
    ]

    hits = 0
    for qa in qa_set:
        q_emb = model.encode(qa["question"], normalize_embeddings=True)
        scores = np.dot(corpus_embeddings, q_emb)
        top1_idx = int(np.argmax(scores))
        top1_source = corpus_sources[top1_idx]

        correct = top1_source.startswith(qa["answer_eu_prefix"])
        hits += int(correct)

        print(f"Q: {qa['question']}")
        print(f"   정답 표: {qa['answer_eu_prefix']} | 검색된 청크 출처: {top1_source} | {'OK' if correct else 'MISS'}")
        print(f"   검색된 청크 미리보기: {corpus_texts[top1_idx][:100]!r}")
        print()

    recall_at_1 = hits / len(qa_set)
    mode = "split ON" if APPLY_SPLIT else "split OFF"
    print(f"=== EU 통합 파이프라인 ({mode}) Recall@1: {recall_at_1:.2f} ({hits}/{len(qa_set)}) ===")


if __name__ == "__main__":
    main()
