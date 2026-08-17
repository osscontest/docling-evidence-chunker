# evidence-chunker

> PDF의 표를 캡션·인접 설명 단락·각주까지 하나의 검색 단위로 묶어주는 RAG 청킹 라이브러리.

![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)

---

## 목차

- [문제 정의](#문제-정의)
- [핵심 아이디어](#핵심-아이디어)
- [주요 기능](#주요-기능)
- [설치](#설치)
- [빠른 시작](#빠른-시작)
- [동작 방식](#동작-방식)
- [API 레퍼런스](#api-레퍼런스)
- [설정 옵션](#설정-옵션)
- [예제](#예제)
- [성능/평가](#성능평가)
- [한계 및 로드맵](#한계-및-로드맵)
- [라이선스](#라이선스)

---

## 문제 정의

일반 텍스트 청커(예: Docling `HybridChunker`)는 표를 문서의 다른 부분과 똑같이 취급한다. 그 결과 캡션과 표 데이터가 서로 다른 청크로 쪼개지고, 긴 표는 청크 경계에서 잘려 뒤쪽 조각이 헤더 없이 숫자만 남으며, 표 전체가 벡터 하나로 뭉쳐서 "표 안 특정 셀 값이 얼마냐" 같은 정밀 질의가 나머지 행 데이터에 묻혀 코사인 유사도에서 밀린다. PDF 90종·2725문항 기준 실측: `HybridChunker` 단독 EM 0.314 → 이 라이브러리 적용 후 0.569(+25.5pp, 95% CI 반폭 ±1.88pp — 아래 "성능/평가" 참고).

## 핵심 아이디어

표 하나 = 캡션 + 표 데이터(HTML + 행 단위 자연어 문장) + 앞뒤 설명 문단 + 섹션 헤더 + 각주를 하나로 묶은 검색 단위 **Evidence Unit(EU)**.

```
PDF
 └─ 표 (TableItem)
     ├─ 캡션          "Table 1: Runtime characteristics..."   (4단계 fallback 매핑)
     ├─ 섹션 헤더      "4.2 Benchmarking"                      (표 바로 위 최근접 헤더)
     ├─ 앞 문단        표 위 300pt 이내 + 임베딩 유사도 통과분
     ├─ 표 데이터      table_html (LLM 컨텍스트용) + flattened_rows (행별 자연어 문장, 검색용)
     ├─ 각주          "OCR is disabled."
     └─ 뒤 문단        표 아래 300pt 이내 + 임베딩 유사도 통과분
              │
              ▼
      EvidenceUnit (eu.text / eu.retrieval_text / eu.retrieval_units)
              │
      512토큰 초과 시 행 단위로 분할 ─┐
              │                     │
              ▼                     ▼
      단일 벡터 (whole-doc)   행 단위 다중 벡터 (small-to-big)
      "표가 뭐에 관한 표냐"    "특정 셀 값이 얼마냐"
      같은 광범위 질의에 강함   같은 정밀 질의에 강함
```

`eu_id`는 `"{doc_id}-p{page}-{idx}"` 형식이라 PDF 여러 개를 한 인덱스에 합쳐도 충돌하지 않는다.

## 주요 기능

- Docling 기반 PDF 파싱 → 표 자동 감지 + 중복 TableItem/목차·그림·표 목록 오탐 필터링
- 캡션 4단계 fallback 매핑(RefItem 직접 연결 → bbox 거리 → 인접 페이지 → 표에 병합된 헤더 행)
- bbox 거리 + 임베딩 코사인 유사도로 표 위아래 설명 단락 자동 흡수(대칭 컨텍스트 앵커링)
- 표 셀 → "행헤더 | 열헤더: 값" 자연어 문장 변환(다단/병합 헤더, 각주 마커 처리)
- 512토큰 초과 표는 헤더/캡션을 유지한 채 행 단위로 자동 분할
- LangChain / LlamaIndex export, 행 단위 다중 벡터(small-to-big) + max-pool 재랭킹
- `PdfParser` 프로토콜로 파서 교체 가능 — `chunk()`만 해당(`build_corpus()`는 Docling `HybridChunker`가 고정 결합돼 있어 미지원)

## 설치

```bash
pip install evidence-chunker[langchain]   # 또는 [llamaindex], 둘 다 필요하면 evidence-chunker[langchain,llamaindex]
```

Python 3.10 이상 필요. GPU 사용 권장 — CPU 환경에서는 큰 PDF 처리 중 메모리 부족이 발생할 수 있다("한계 및 로드맵" 참고).

## 빠른 시작

```python
from evidence_chunker import EvidenceChunker
from evidence_chunker.export.langchain import to_langchain

chunker = EvidenceChunker()
docs = to_langchain(chunker.build_corpus("paper.pdf"))   # 표(EU) + 일반 본문, 벡터스토어에 바로 삽입 가능
```

## 동작 방식

1. **파싱**: Docling(`DocumentConverter`)으로 PDF를 파싱하고, `DoclingParser`가 그 결과를 Docling과 무관한 내부 모델(`ParsedDoc`)로 변환(BOTTOMLEFT → TOPLEFT 좌표 정규화 포함).
2. **필터링**: 같은 물리적 표가 TableItem 2개로 중복 감지된 경우 정리하고, 목차/그림·표 목록이 표로 오인식된 경우를 제외.
3. **캡션/문맥 매핑**: 표마다 캡션을 4단계 fallback으로 찾고, bbox 거리 + 임베딩 유사도로 인접 설명 단락과 섹션 헤더를 붙임.
4. **행 플래트닝**: 표 셀을 자연어 문장으로 변환하고, 캡션+헤더 조합으로 표 요약(`table_abstract`)을 생성.
5. **분할**: 512토큰을 넘는 EU는 헤더·캡션·문맥을 유지한 채 행 단위로 쪼갬.
6. **(선택) 일반 본문 병합**: `build_corpus()`는 Docling `HybridChunker`로 표 이외 본문 청크도 만들고, EU가 이미 흡수한 문단과 겹치는 청크는 제거해 합침.

## API 레퍼런스

### `EvidenceChunker`

| 메서드 | 설명 |
|---|---|
| `chunk(pdf_path, doc_id=None)` | 표(Evidence Unit)만 추출. `List[EvidenceUnit]` 반환. `parser` 주입 시 완전히 교체 가능. |
| `build_corpus(pdf_path, doc_id=None)` | 표 + 일반 본문을 합친 최종 검색 코퍼스. `List[RetrievalChunk]` 반환. `parser` 주입 미지원(`NotImplementedError`). |

### `evidence_chunker.export`

| 함수/클래스 | 설명 |
|---|---|
| `to_langchain(chunks)` / `to_llamaindex(chunks)` | 1 chunk = 1 Document/TextNode |
| `to_langchain_units(chunks)` / `to_llamaindex_units(chunks)` | 행 단위 다중 벡터(small-to-big) — `metadata["parent_text"]`에 EU 전체 맥락 |
| `EvidenceRetriever(vectorstore_or_retriever, k=5)` | 같은 EU에서 나온 검색 결과를 `chunk_id` 기준으로 max-pool 재랭킹 |
| `dedupe_by_chunk_id(results, k=None)` | `EvidenceRetriever` 내부에서 쓰는 재랭킹 함수 직접 호출용 |

상세 동작/파라미터는 각 함수의 docstring 참고.

## 설정 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `parser` | `None` | `PdfParser` 프로토콜 구현체. `None`이면 기본 `DoclingParser` 사용. `chunk()`만 교체 가능. |
| `artifacts_path` | `None` | Docling 로컬 모델 경로. `None`이면 HuggingFace Hub에서 자동 다운로드. `parser` 직접 넘기면 무시됨. |
| `bbox_threshold` | `300.0` | 표 위아래 설명 단락 수집 범위(PDF 포인트). |
| `sim_threshold` | `0.0` | 인접 단락 채택 임베딩 코사인 유사도 임계값. 기본값(0.0)에서는 이 필터가 사실상 비활성화되어(bbox 거리만으로 채택) `sentence-transformers` 없이도 동작한다. |

## 예제

**벡터스토어에 바로 넣기** (표+본문 통짜 1 chunk = 1 Document):

```python
from evidence_chunker import EvidenceChunker
from evidence_chunker.export.langchain import to_langchain

chunks = EvidenceChunker().build_corpus("paper.pdf")
docs = to_langchain(chunks)
```

**행 단위 다중 벡터 + max-pool 재랭킹** (표 안 특정 셀 값을 묻는 질의에 강함):

```python
from evidence_chunker.export.langchain import to_langchain_units, EvidenceRetriever

docs = to_langchain_units(chunks)          # 검색은 이 작은 조각들로
vectorstore = ...                          # docs로 구성한 LangChain VectorStore
retriever = EvidenceRetriever(vectorstore, k=5)
results = retriever.get_relevant_documents(query)
# 매칭된 조각의 metadata["parent_text"](표 전체 맥락)를 LLM에 넘길 것
```

LlamaIndex는 `evidence_chunker.export.llamaindex`의 동일한 이름의 함수/클래스로 쓴다.

**파서 교체**:

```python
from evidence_chunker import EvidenceChunker
from evidence_chunker.parser.base import PdfParser, ParsedDoc

class MyParser:  # PdfParser 프로토콜: parse(path) -> ParsedDoc
    def parse(self, path: str) -> ParsedDoc:
        ...

eus = EvidenceChunker(parser=MyParser()).chunk("paper.pdf")   # Docling을 아예 거치지 않음
```

## 성능/평가

PDF 90종·2725문항(자동 생성 QA) 기준 실측. baseline = Docling `HybridChunker` 그대로, EU = 이 라이브러리(`build_corpus()`).

| 지표 | baseline | EU | 갭 |
|---|---|---|---|
| Recall@1 | 0.574 | 0.636 | +6.2pp |
| EM | 0.314 | 0.569 | +25.5pp |

95% CI 반폭 ±1.88pp — EM 갭은 CI의 13배 이상이다. Recall 갭은 작은데 EM 갭이 큰 이유: baseline도 정답 표를 찾기는 하지만, 그 청크 안에 정답이 실제로 담겨 있지 않은 경우가 많다.

지표 정의, 유형별/청크 크기별 대조군, max-pool 재랭킹 트레이드오프, 문항 단위 승패 분해 등 전체 결과는 **[docs/BENCHMARK.md](docs/BENCHMARK.md)** 참고.

## 한계 및 로드맵

- **max-pool 재랭킹 R@10 트레이드오프**: 행 분할 조각이 서로 다른 `chunk_id`를 받아 재랭킹 시 하나로 안 묶인다. → 분할 조각에 원본 표 식별자(`original_eu_id`)를 추가해 dedup 키를 확장할 예정.
- **표 간 혼동(cross-table confusion)**: 한 문서 안에 구조적으로 비슷한 표가 다수 반복되면(모델별 벤치마크 결과표처럼 컬럼 구조만 같은 표가 10~40개씩 있는 경우) 캡션·section_header가 아니라 표 내용 자체가 임베딩 공간에서 잘 구분되지 않아 엉뚱한 표를 고르는 경우가 남아있다 — 표 2개 이상 문서의 EM 개선폭이 단일 표 문서의 절반 이하. → 표 캡션에 고유 식별자(섹션 경로·표 번호)를 더 강하게 prefix로 반영하거나 문서 내 중복 임계값 조정을 검토할 예정.
- **스캔본(OCR 필요) PDF**: `do_ocr` 하드코딩(`False`)을 제거하고 Docling 기본값을 따르도록 바꿨지만, 아직 재검증 전. 한국어 인식 품질(RapidOCR)도 별도 확인 필요. → OCR 재현성을 확보한 뒤 스캔본 PDF 지원을 재개할 예정.
- **큰 PDF 처리 시 메모리 부족(CPU)**: Docling 변환 중 누적 메모리가 일정 수준(실측 약 2.3GB)을 넘으면 `std::bad_alloc` 발생. GPU 사용을 권장하는 이유 중 하나("설치" 참고). 같은 프로세스에서 같은 PDF를 반복 파싱하면 재현되므로, 테스트/반복 파싱 시에는 세션 스코프 픽스처로 1회만 수행할 것(`tests/conftest.py` 참고).
- **`bbox_threshold`(300pt) 값의 대표성**: 소규모 dev 문서셋으로 정해진 뒤 고정된 상수라 전체 90종에 최적인 값인지는 검증되지 않았다 — 이 경계 밖에 있는 127문항이 여전히 안 풀리는 것도 이와 무관하지 않다. → 300pt의 근거를 재검토해 재설정할 예정.
- **평가셋의 편향 가능성**: 90종 PDF가 모두 영어 텍스트 기반 디지털 문서(논문/정부보고서/학술지)라, 스캔본·비영어·표 형식이 크게 다른 문서(양식, 인보이스 등)에 대한 일반화는 검증되지 않았다.

## 라이선스

[Apache 2.0](LICENSE)
