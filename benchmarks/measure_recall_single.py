"""
measure_recall_single.py

W4: 3모듈 통합 파이프라인(caption_mapper + context_attacher + table_splitter)의
Recall@1을 baseline.py(Docling HybridChunker 단독, 0.60)와 비교 측정.

파이프라인:
    PDF --DocumentConverter()--> doc (baseline과 동일 컨버터/옵션)
      +-- doc.tables --> build_evidence_units() --> [table_splitter.split_eu()] --> EU 청크
      +-- HybridChunker(doc) --> 청크 중 표 관련(doc_item.label == "table") 청크는 EU로 대체되므로 제외
    두 그룹을 합쳐 최종 검색 코퍼스 구성 후 baseline과 동일 질문으로 Recall@1 측정.

표 청크 판별 기준(= baseline과 동일 기준으로 고정):
    chunk.meta.doc_items 중 하나라도 label == DocItemLabel.TABLE 이면 "표 청크"로 간주하고 제외.
    (Docling은 표 캡션도 표 데이터와 함께 label=table로 직렬화하므로 캡션 중복도 함께 제거됨)

W4 중간측정 회귀(0.60 -> 0.40) 원인 3건 및 조치:
    1. context_attacher가 같은 페이지 내에서만 컨텍스트를 탐색 -> 인접 페이지 경계
       탐색 추가로 해결 (context_attacher._collect_by_bbox의 ADJACENT_PAGE_EDGE_RATIO).
    2. Docling이 표 1개를 TableItem 2개로 중복 감지 -> _table_utils.find_duplicate_tables()
       로 셀 텍스트 중복도를 비교해 더 세밀하게 구조화된 쪽만 남기고 캡션을 물려줌.
    3. 캡션+표+컨텍스트를 합친 긴 EU 청크가 짧은 서술형 문단 청크보다 임베딩 유사도에서
       밀림 -> 원인은 all-MiniLM-L6-v2가 입력을 256 토큰에서 truncate하는데, eu.text는
       raw table_html(토큰 낭비가 큰 마크업)이 flattened_rows(자연어 문장, 실제 검색
       신호)보다 앞에 있어 truncate 시 검색에 필요한 내용이 통째로 잘려나갔기 때문.
       EvidenceUnit.retrieval_text(table_html 제외, 신호 밀도 높은 순 배치)로 임베딩하고
       eu.text(전체 내용)는 표시/생성용으로만 사용하도록 분리해 해결.

0.60 -> 0.80 회귀 후속 조치:
    - context_attacher._same_column의 20pt 허용오차가 표-문단 간 x축 간격이
      그보다 살짝 큰(실사례 28pt) 경우를 걸러내던 문제 -> 같은 컬럼에서 못 찾으면
      컬럼 제한 없이 재탐색하는 폴백 추가.
    - attach_context_paragraphs가 eu_id로 표를 재추정하던 방식이 find_duplicate_tables
      dedup 이후 doc.tables 스캔 순서와 어긋나 엉뚱한(제거된 중복) 표 기준으로 컨텍스트를
      찾던 버그 -> 호출자가 이미 알고 있는 table_bbox를 직접 넘기도록 수정.

남은 miss 대응 시도 — 2단계 재순위화(bi-encoder 후보 -> cross-encoder 재채점):
    bi-encoder(all-MiniLM-L6-v2) top-K 후보를 cross-encoder(ms-marco-MiniLM-L-6-v2)로
    재채점해봤으나, 이 5문제 벤치마크에서는 오히려 Recall@1이 0.80 -> 0.60으로 하락함
    (실측 확인, 되돌리지 않고 옵션으로만 남겨둠). 원인: ms-marco-MiniLM은 MS MARCO
    (서술형 문장 지문 위주) 데이터로 학습된 모델이라, 표에서 뽑은 짧고 구조화된
    retrieval_text보다 "질문에 자연스럽게 답하는 것처럼 읽히는" 서술형 문단을 구조적으로
    선호하는 경향이 있음 — 표 안의 정확한 수치보다, 모델명만 언급하고 실제 값은 없는
    문단을 더 관련 있다고 채점한 사례 확인. 기본값은 OFF, --rerank로 켜서 비교 가능.

남은 miss 대응 시도 — 검색 특화(비대칭 query/passage) 임베딩 모델 교체:
    all-MiniLM-L6-v2 대신 bge-small-en-v1.5 / e5-small-v2로 교체해봤으나 둘 다
    이 벤치마크에서는 더 나빴음 (실측: bge 0.40, e5 0.60, all-MiniLM 0.80).
    두 모델 모두 max_seq_length가 512라 truncation 문제는 없었는데도 밀렸다는 점에서,
    단순 truncation 회피보다 이 문서(영어 논문, 표 기반 QA)에 대한 모델별 궁합 차이가
    더 크게 작용한 것으로 보임. 기본값은 all-MiniLM 유지, --embed-model=bge|e5로 비교 가능.

Usage:
    python benchmarks/measure_recall_single.py                      # split 적용, rerank 미적용, all-MiniLM (기본)
    python benchmarks/measure_recall_single.py --no-split            # table_splitter 끄고 측정 (ablation)
    python benchmarks/measure_recall_single.py --rerank              # cross-encoder 재순위화 켜고 측정 (이 벤치마크에선 손해로 측정됨)
    python benchmarks/measure_recall_single.py --embed-model=bge     # BAAI/bge-small-en-v1.5로 측정
    python benchmarks/measure_recall_single.py --embed-model=e5      # intfloat/e5-small-v2로 측정
"""
import sys
import os

if hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pdfs", "docling_technical_report.pdf")

APPLY_SPLIT = "--no-split" not in sys.argv
APPLY_RERANK = "--rerank" in sys.argv  # 기본 OFF — 이 벤치마크에서 Recall@1 0.80->0.60 손해로 측정됨
RERANK_TOP_K = 8

# 검색 특화(비대칭 query/passage) 임베딩 모델 후보.
# all-MiniLM-L6-v2는 범용 문장 유사도용이라 질문<->문서 비대칭 매칭에 약함 —
# bge/e5는 query/passage를 다르게 인코딩하도록 프리픽스로 학습돼 있어 근소한
# 점수 차이로 밀리던 케이스를 뒤집을 가능성이 있음.
EMBED_MODELS = {
    "minilm": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",
        "passage_prefix": "",
    },
    "bge": {
        "name": "BAAI/bge-small-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
    },
    "e5": {
        "name": "intfloat/e5-small-v2",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
}
EMBED_MODEL_KEY = "minilm"
for _arg in sys.argv:
    if _arg.startswith("--embed-model="):
        EMBED_MODEL_KEY = _arg.split("=", 1)[1]


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

    from evidence_chunker.chunker import build_evidence_units
    from evidence_chunker.parser.docling import DoclingParser
    from evidence_chunker.split import split_oversized_units

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
    parsed = DoclingParser().from_doc(doc)
    eu_list = build_evidence_units(parsed)
    print(f"  원본 표 EU: {len(eu_list)}개")

    if APPLY_SPLIT:
        eu_list = split_oversized_units(eu_list)
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
    # 4. 최종 코퍼스 구성: EU를 행 단위(retrieval_units)로 쪼개서 다중 벡터로
    #    인덱싱 + 비-표 HybridChunker 청크
    #
    #    EU 하나를 통짜로 한 벡터에 임베딩하면(retrieval_text) "표 안의 특정
    #    셀 값"을 묻는 질의가 같은 표의 나머지 행 데이터에 묻혀 유사도에서
    #    밀리는 문제가 있었다 (원인 #3 잔여 이슈). flattened_rows 문장을
    #    각각 별도 벡터로 인덱싱하고, 어느 문장이 매칭되든 그 문장이 속한
    #    EU 전체(eu.text)를 반환하는 small-to-big 패턴으로 바꾼다.
    # ------------------------------------------------------------------
    section("최종 코퍼스 (행 단위 다중 벡터)")
    corpus_embed_texts: list[str] = []  # 임베딩 대상 — EU 1개당 여러 unit(행)
    corpus_display: list[str] = []      # 표시/생성용 — unit이 속한 부모 청크 전체
    corpus_sources: list[str] = []      # unit이 속한 eu_id / "hybrid-{i}" (재순위화 후보 dedup용 고유 키)
    corpus_rerank_texts: list[str] = []  # cross-encoder 재채점용 — source당 동일한 대표 텍스트

    for eu in eu_list:
        for unit_text in eu.retrieval_units:
            corpus_embed_texts.append(unit_text)
            corpus_display.append(eu.text)
            corpus_sources.append(eu.eu_id)
            corpus_rerank_texts.append(eu.retrieval_text)

    for i, c in enumerate(non_table_chunks):
        corpus_embed_texts.append(c.text)
        corpus_display.append(c.text)
        corpus_sources.append(f"hybrid-{i}")  # source별 고유 키 — dedup 시 서로 다른 hybrid 청크가 뭉치지 않게
        corpus_rerank_texts.append(c.text)

    n_eu_units = sum(len(eu.retrieval_units) for eu in eu_list)
    print(f"  총 {len(corpus_embed_texts)}개 검색 유닛 "
          f"(EU {len(eu_list)}개 -> 행 단위 {n_eu_units}개 + 비-표 HybridChunker {len(non_table_chunks)}개)")

    # ------------------------------------------------------------------
    # 5. 임베딩 + Recall@1 (baseline_recall_local.py와 동일 질문/모델)
    #    검색은 행 단위 유닛으로, 검색된 청크 표시/실사용 콘텐츠는
    #    corpus_display(부모 EU의 eu.text, 전체) 사용.
    # ------------------------------------------------------------------
    section("Recall@1 측정")
    embed_cfg = EMBED_MODELS[EMBED_MODEL_KEY]
    print(f"  임베딩 모델: {embed_cfg['name']}")
    model = SentenceTransformer(embed_cfg["name"])
    passage_texts = [embed_cfg["passage_prefix"] + t for t in corpus_embed_texts]
    corpus_embeddings = model.encode(passage_texts, normalize_embeddings=True)

    reranker = None
    if APPLY_RERANK:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # baseline qa_set과 동일 질문. 정답은 청크 인덱스가 아니라
    # "정답 EU가 속한 표(eu_id 접두사)"로 정의 — EU 파이프라인에서는
    # 캡션+데이터가 한 청크로 합쳐지므로 baseline처럼 인덱스로 고정할 수 없음.
    #
    # 주의: Docling이 8페이지에서 Table 2를 TableItem 2개로 중복 감지했었음
    # (행이 뭉개진 채 캡션만 연결된 표 + 행이 올바르게 분리됐지만 캡션이 없는 표).
    # build_evidence_units()의 find_duplicate_tables() dedup으로 더 세밀하게
    # 구조화된(행이 분리된) 쪽만 남기고 캡션을 물려주므로, 8페이지 표는 이제
    # docling_technical_report-p8-0 하나로 합쳐짐.
    qa_set = [
        {
            "question": "What is the TTS of Intel Xeon with 4 threads using native backend?",
            "answer_eu_prefix": "docling_technical_report-p5-0",  # Table 1
        },
        {
            "question": "What is the runtime characteristics table about?",
            "answer_eu_prefix": "docling_technical_report-p5-0",  # Table 1
        },
        {
            "question": "What is the mAP of YOLOv5 on DocLayNet?",
            "answer_eu_prefix": "docling_technical_report-p8-0",  # Table 2 (dedup 후 유일한 8페이지 표)
        },
        {
            "question": "Which models were used for baseline experiments on DocLayNet?",
            "answer_eu_prefix": "docling_technical_report-p8-0",  # Table 2 (dedup 후 유일한 8페이지 표)
        },
        {
            "question": "What are the pages per second for pypdfium backend with 16 threads?",
            "answer_eu_prefix": "docling_technical_report-p5-0",  # Table 1
        },
    ]

    hits = 0
    for qa in qa_set:
        q_emb = model.encode(embed_cfg["query_prefix"] + qa["question"], normalize_embeddings=True)
        scores = np.dot(corpus_embeddings, q_emb)

        # bi-encoder 1차 순위
        order = np.argsort(-scores)
        bi_top1_idx = int(order[0])

        top1_idx = bi_top1_idx
        rerank_note = ""

        if reranker is not None:
            # top-K 후보를 source(eu_id / hybrid-i)별로 dedup — 같은 표의 여러
            # 행 유닛이 top-K를 독점하는 것을 방지하고, source당 대표 후보 하나만 재채점.
            candidates: dict[str, int] = {}
            for idx in order:
                idx = int(idx)
                src = corpus_sources[idx]
                if src not in candidates:
                    candidates[src] = idx
                if len(candidates) >= RERANK_TOP_K:
                    break

            cand_idxs = list(candidates.values())
            pairs = [(qa["question"], corpus_rerank_texts[idx]) for idx in cand_idxs]
            rerank_scores = reranker.predict(pairs)
            best_pos = int(np.argmax(rerank_scores))
            reranked_idx = cand_idxs[best_pos]

            if reranked_idx != bi_top1_idx:
                rerank_note = f" (rerank 전: {corpus_sources[bi_top1_idx]})"
            top1_idx = reranked_idx

        top1_source = corpus_sources[top1_idx]

        correct = top1_source.startswith(qa["answer_eu_prefix"])
        hits += int(correct)

        print(f"Q: {qa['question']}")
        print(f"   정답 표: {qa['answer_eu_prefix']} | 검색된 청크 출처: {top1_source}{rerank_note} | {'OK' if correct else 'MISS'}")
        print(f"   매칭된 유닛: {corpus_embed_texts[top1_idx][:80]!r}")
        print(f"   반환될 부모 청크 미리보기: {corpus_display[top1_idx][:100]!r}")
        print()

    recall_at_1 = hits / len(qa_set)
    mode = "split ON" if APPLY_SPLIT else "split OFF"
    rerank_mode = "rerank ON" if APPLY_RERANK else "rerank OFF"
    print(f"=== EU 통합 파이프라인 ({mode}, {rerank_mode}, embed={EMBED_MODEL_KEY}) "
          f"Recall@1: {recall_at_1:.2f} ({hits}/{len(qa_set)}) ===")


if __name__ == "__main__":
    main()
