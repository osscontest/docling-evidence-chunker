"""
measure_recall_multi.py

여러 PDF x 여러 유형의 질문으로 Recall@1을 측정하는 확장판.

measure_recall_eu.py(문서 1개, 질문 5개)는 표본이 너무 작아서 질문 하나
뒤집힐 때마다 20%p씩 흔들리는 문제가 있다. 문서 3개(영어 논문 2 + 기술
보고서 1) x 질문 24개로 확장해서 좀 더 안정적인 수치를 본다.

모든 질문의 정답은 실제로 해당 PDF를 파싱해서 확인한 표 내용(캡션/셀 값/
컨텍스트 문단)을 기준으로 만들었다 — 지어낸 질문 아님.

질문 유형(type) 태그별로 파이프라인의 어느 기능을 검증하는지:
    cell_value    : 표 안 특정 셀 값을 묻는 질문 -> flattened_rows / 행 단위 다중 벡터
    table_about   : "이 표가 뭐에 대한 표냐"는 광범위 질문 -> caption / table_abstract
    context_only  : 표 위아래 설명 문단에만 답이 있는 질문 -> context_attacher

Usage:
    python benchmarks/measure_recall_multi.py
    python benchmarks/measure_recall_multi.py --no-split
"""
import sys
import os

if hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PDF_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "pdfs")
APPLY_SPLIT = "--no-split" not in sys.argv

# ---------------------------------------------------------------------------
# 문서별 질문 세트
# ---------------------------------------------------------------------------

QA_SETS = {
    "docling_technical_report.pdf": [
        {"question": "What is the TTS of Intel Xeon with 4 threads using native backend?",
         "answer_eu_prefix": "docling_technical_report-p5-0", "type": "cell_value"},
        {"question": "What is the runtime characteristics table about?",
         "answer_eu_prefix": "docling_technical_report-p5-0", "type": "table_about"},
        {"question": "What is the mAP of YOLOv5 on DocLayNet?",
         "answer_eu_prefix": "docling_technical_report-p8-0", "type": "cell_value"},
        {"question": "Which models were used for baseline experiments on DocLayNet?",
         "answer_eu_prefix": "docling_technical_report-p8-0", "type": "context_only"},
        {"question": "What are the pages per second for pypdfium backend with 16 threads?",
         "answer_eu_prefix": "docling_technical_report-p5-0", "type": "cell_value"},
        {"question": "What is Table 2 in the docling technical report about?",
         "answer_eu_prefix": "docling_technical_report-p8-0", "type": "table_about"},
    ],
    "attention_is_all_you_need.pdf": [
        {"question": "What is the maximum path length for Self-Attention layers?",
         "answer_eu_prefix": "attention_is_all_you_need-p6-0", "type": "cell_value"},
        {"question": "What is the sequential operations complexity of Recurrent layers?",
         "answer_eu_prefix": "attention_is_all_you_need-p6-0", "type": "cell_value"},
        {"question": "What is Table 1 in this paper about?",
         "answer_eu_prefix": "attention_is_all_you_need-p6-0", "type": "table_about"},
        {"question": "What is the BLEU score of the Transformer (big) model on English-to-German?",
         "answer_eu_prefix": "attention_is_all_you_need-p8-0", "type": "cell_value"},
        {"question": "What is the Transformer base model's BLEU score on English-to-French?",
         "answer_eu_prefix": "attention_is_all_you_need-p8-0", "type": "cell_value"},
        {"question": "What does Table 2 compare between different translation models?",
         "answer_eu_prefix": "attention_is_all_you_need-p8-0", "type": "table_about"},
        {"question": "What beam size was used for the WSJ constituency parsing experiments?",
         "answer_eu_prefix": "attention_is_all_you_need-p10-0", "type": "context_only"},
        {"question": "What is the training cost in FLOPs of the ConvS2S model on English-to-French?",
         "answer_eu_prefix": "attention_is_all_you_need-p8-0", "type": "cell_value"},
    ],
    "bert_paper.pdf": [
        {"question": "What is BERT_BASE's MNLI-m accuracy on the Dev Set in the ablation study?",
         "answer_eu_prefix": "bert_paper-p8-0", "type": "cell_value"},
        {"question": "What is the SQuAD F1 score of the 'No NSP' model on the Dev Set?",
         "answer_eu_prefix": "bert_paper-p8-0", "type": "cell_value"},
        {"question": "What is the MRPC accuracy of the LTR & No NSP model?",
         "answer_eu_prefix": "bert_paper-p8-0", "type": "cell_value"},
        {"question": "What ablation is performed in Table 5 of the BERT paper?",
         "answer_eu_prefix": "bert_paper-p8-0", "type": "table_about"},
        {"question": "Where can additional ablation studies be found according to the paper?",
         "answer_eu_prefix": "bert_paper-p8-0", "type": "context_only"},
        {"question": "What is the SST-2 accuracy of the '+BiLSTM' configuration?",
         "answer_eu_prefix": "bert_paper-p8-0", "type": "cell_value"},
        {"question": "What is ELMo's Test F1 score in the CoNLL-2003 NER results?",
         "answer_eu_prefix": "bert_paper-p9-1", "type": "cell_value"},
        {"question": "What is BERT_BASE's QNLI accuracy on the Dev Set in the ablation table?",
         "answer_eu_prefix": "bert_paper-p8-0", "type": "cell_value"},
        {"question": "What does Table 1 (GLUE results) evaluate?",
         "answer_eu_prefix": "bert_paper-p6-0", "type": "table_about"},
        {"question": "How many systems does the BERT ensemble use for SQuAD 1.1?",
         "answer_eu_prefix": "bert_paper-p7-0", "type": "table_about"},
    ],
}


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def build_corpus(doc, eu_list):
    """measure_recall_eu.py와 동일한 행 단위 다중 벡터 코퍼스 구성."""
    from docling.chunking import HybridChunker
    from docling_core.types.doc import DocItemLabel

    chunker = HybridChunker()
    all_chunks = list(chunker.chunk(doc))

    def is_table_chunk(chunk) -> bool:
        return any(di.label == DocItemLabel.TABLE for di in chunk.meta.doc_items)

    non_table_chunks = [c for c in all_chunks if not is_table_chunk(c)]

    corpus_embed_texts, corpus_display, corpus_sources = [], [], []
    for eu in eu_list:
        for unit_text in eu.retrieval_units:
            corpus_embed_texts.append(unit_text)
            corpus_display.append(eu.text)
            corpus_sources.append(eu.eu_id)
    for i, c in enumerate(non_table_chunks):
        corpus_embed_texts.append(c.text)
        corpus_display.append(c.text)
        corpus_sources.append(f"hybrid-{i}")

    return corpus_embed_texts, corpus_display, corpus_sources


def evaluate_pdf(pdf_name, qa_list, model):
    from docling.document_converter import DocumentConverter
    from evidence_chunker.chunker import build_evidence_units
    from evidence_chunker.split import split_oversized_units
    import numpy as np

    pdf_path = os.path.join(PDF_DIR, pdf_name)
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document

    eu_list = build_evidence_units(doc)
    if APPLY_SPLIT:
        eu_list = split_oversized_units(eu_list)

    corpus_embed_texts, corpus_display, corpus_sources = build_corpus(doc, eu_list)
    corpus_embeddings = model.encode(corpus_embed_texts, normalize_embeddings=True)

    results = []
    for qa in qa_list:
        q_emb = model.encode(qa["question"], normalize_embeddings=True)
        scores = np.dot(corpus_embeddings, q_emb)
        top1_idx = int(np.argmax(scores))
        top1_source = corpus_sources[top1_idx]
        correct = top1_source.startswith(qa["answer_eu_prefix"])
        results.append({**qa, "pdf": pdf_name, "correct": correct, "top1_source": top1_source})
    return results


def main():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    all_results = []
    for pdf_name, qa_list in QA_SETS.items():
        section(f"평가: {pdf_name} ({len(qa_list)}문제)")
        results = evaluate_pdf(pdf_name, qa_list, model)
        all_results.extend(results)
        for r in results:
            tag = "OK  " if r["correct"] else "MISS"
            print(f"  [{tag}] ({r['type']:12s}) {r['question'][:70]}")
            if not r["correct"]:
                print(f"         정답: {r['answer_eu_prefix']} | 검색됨: {r['top1_source']}")

    section("결과 요약")
    total = len(all_results)
    hits = sum(1 for r in all_results if r["correct"])
    print(f"  전체 Recall@1: {hits/total:.2f} ({hits}/{total})\n")

    print("  문서별:")
    for pdf_name in QA_SETS:
        subset = [r for r in all_results if r["pdf"] == pdf_name]
        h = sum(1 for r in subset if r["correct"])
        print(f"    {pdf_name}: {h/len(subset):.2f} ({h}/{len(subset)})")

    print("\n  유형별:")
    types = sorted(set(r["type"] for r in all_results))
    for t in types:
        subset = [r for r in all_results if r["type"] == t]
        h = sum(1 for r in subset if r["correct"])
        print(f"    {t}: {h/len(subset):.2f} ({h}/{len(subset)})")


if __name__ == "__main__":
    main()
