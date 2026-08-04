# evidence-chunker

표 하나를 캡션·주변 문맥·각주까지 통째로 붙잡아 검색 가능한 단위로 만드는 RAG 청킹 라이브러리. Docling으로 PDF를 파싱한 뒤 이 표-중심 단위(Evidence Unit)를 만든다.

## 왜 필요한가

일반 텍스트 청커(예: Docling `HybridChunker`)는 표를 문서의 다른 부분과 똑같이 취급한다. 그 결과:

- **캡션과 표가 분리된다.** "Table 2: Prediction performance..."라는 캡션과 그 아래 13행짜리 표 데이터가 서로 다른 청크로 쪼개지거나, 캡션이 아예 없는 청크로 표만 남는다.
- **토큰 한도에서 표가 뭉텅 잘린다.** 긴 표가 청크 경계에서 잘리면 뒤쪽 조각은 헤더 정보 없이 숫자만 남는다.
- **표 전체가 벡터 하나로 뭉친다.** "YOLOv5의 mAP가 얼마냐" 같이 표 안 특정 셀 값을 묻는 질의가, 같은 표의 나머지 12개 행 데이터에 묻혀 코사인 유사도에서 밀린다.

`docling_technical_report.pdf`(9p) 기준 실측: `HybridChunker` 단독 Recall@1 0.60 → Evidence Unit 파이프라인 적용 후 0.80 (아래 "Ablation" 절 참고).

## Evidence Unit

표 하나 = 캡션 + 표 데이터(HTML + 행 단위 자연어 문장) + 앞뒤 설명 문단 + 섹션 헤더 + 각주를 하나로 묶은 검색 단위.

```
PDF
 └─ 표 (TableItem)
     ├─ 캡션          "Table 1: Runtime characteristics..."   (3단계 매핑, 아래 참고)
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
              │                    │
              ▼                    ▼
      단일 벡터 (whole-doc)   행 단위 다중 벡터 (small-to-big)
      "표가 뭐에 관한 표냐"    "특정 셀 값이 얼마냐"
      같은 광범위 질의에 강함   같은 정밀 질의에 강함
```

## 설치

```bash
pip install evidence-chunker[langchain]   # 또는 [llamaindex], 둘 다 필요하면 evidence-chunker[langchain,llamaindex]
```

## 퀵스타트

```python
from evidence_chunker import EvidenceChunker
from evidence_chunker.export.langchain import to_langchain

chunker = EvidenceChunker()
docs = to_langchain(chunker.build_corpus("paper.pdf"))   # 표(EU) + 일반 본문, 벡터스토어에 바로 삽입 가능
```

표만 필요하면:

```python
eus = chunker.chunk("paper.pdf")   # List[EvidenceUnit]
```

표 안 특정 셀 값을 묻는 질의가 많다면 행 단위 다중 벡터(small-to-big)를 쓸 것 — 검색은 작은 조각으로, 매칭되면 `metadata["parent_text"]`(표 전체 맥락)를 LLM에 넘긴다:

```python
from evidence_chunker.export.langchain import to_langchain_units
docs = to_langchain_units(chunker.build_corpus("paper.pdf"))
```

LlamaIndex는 `evidence_chunker.export.llamaindex`의 `to_llamaindex` / `to_llamaindex_units`로 동일하게 쓴다.

## Ablation

`docling_technical_report.pdf`(9p, 표 2개) 기준, 질문 5개(`benchmarks/measure_recall_single.py`) Recall@1:

| 설정 | Recall@1 |
|---|---|
| `HybridChunker` 단독 (Evidence Unit 없음) | 0.60 (3/5) |
| **Evidence Unit + 512토큰 분할 (기본값)** | **0.80 (4/5)** |
| Evidence Unit, 분할 없음 (`--no-split`) | 0.80 (4/5) |
| Evidence Unit + cross-encoder 재순위화 (`--rerank`) | 0.60 (3/5) |
| Evidence Unit, 임베딩 모델 = bge-small-en-v1.5 (`--embed-model=bge`) | 0.60 (3/5) |
| Evidence Unit, 임베딩 모델 = e5-small-v2 (`--embed-model=e5`) | 0.60 (3/5) |

재현: `python benchmarks/measure_recall_single.py [--no-split|--rerank|--embed-model=bge|e5]`, `python benchmarks/baseline.py`(HybridChunker 단독).

**한계**: 문서 1종·질문 5개짜리 표본이라 각 행이 통계적으로 유의하다고 주장하기 어렵다. 질문 하나가 뒤집히면 20%p씩 흔들린다 — 문서 3종·질문 24개로 확장한 `benchmarks/measure_recall_multi.py`가 있지만, 필요한 PDF 중 2종(16p, 75p)이 아래 "알려진 이슈"의 메모리 문제로 파싱이 안 돼 아직 전체 실행 결과가 없다. rerank/bge/e5가 전부 minilm보다 나쁘게 나온 것도 "이 조합에서는" 그렇다는 것이지 일반적 우위 주장이 아니다.

## 알고리즘

### 캡션↔표 매핑 (`evidence_chunker.caption`)

`table.captions` RefItem을 파싱해 표-캡션을 1:1로 연결. 우선순위 3단계:

| confidence | 의미 |
|---|---|
| `direct` | `captions` RefItem이 가리키는 텍스트가 캡션 패턴(`Table N`, `표 N` 등)에 맞아 그대로 채택 |
| `inferred` | RefItem이 없거나 엉뚱한 파편을 가리켜서, 표 위/아래 bbox 거리(200pt 이내)에서 캡션처럼 생긴 텍스트를 재탐색해 찾음 |
| `none` | 둘 다 실패, 캡션 없음 |

```python
from evidence_chunker.caption import map_table_caption, map_all_captions, validate_mapping

mapping = map_table_caption(doc, table, table_index)
# CaptionMapping(caption_text=..., confidence="direct"|"inferred"|"none", multi_caption=..., cross_page=...)
```

검증 리포트: `python benchmarks/report_caption_mapping.py --all`

**한국어 PDF 테스트로 발견한 버그 2건.** 영어 논문 2종만으로는 100% 매핑돼서 예외 케이스가 0건 — 알고리즘이 좋아서가 아니라 테스트셋이 너무 쉬워서였을 가능성이 있었음. 한국은행 잠재성장률 보고서(한국어)와 GPT-3 논문을 테스트셋에 추가(`benchmarks/tools/download_test_pdfs.py`)해서 검증한 결과, 실제 버그 2건을 발견하고 수정함:

- RefItem이 캡션이 아닌 파편("(단위: 1), %)")을 가리키는 경우 → 캡션 패턴 검증 후 `inferred`로 재탐색해 정정
- RefItem 자체가 비어있는 경우 → bbox fallback으로 `<표 N>` 형식의 실제 캡션을 찾아냄

전체 4개 PDF(영어 2 + 한국어 1 + GPT-3) 기준 표 23개 중 20개 매핑 성공 (87%).

**캡션 예외처리.** 캡션 없는 표(fallback 전부 실패 시 `confidence="none"`, 표는 유지), 복수 캡션(RefItem 2개 이상이면 캡션 패턴에 맞는 것만 이어붙임), 다음/이전 페이지 캡션(`cross_page=True`, 페이지가 다르면 좌표계가 달라 이전 페이지 맨 아래/다음 페이지 맨 위 텍스트를 채택) 세 가지를 처리. 실제 테스트 PDF 4종(표 23개) 어디서도 `multi_caption`/`cross_page`가 실제로 발동한 적은 없어서 합성(mock) 문서로 직접 검증: `pytest tests/test_caption_exceptions.py`.

### Row Flattening (`evidence_chunker.flatten`)

표 셀을 "행헤더 | 열헤더: 값" 형태 자연어 문장으로 변환 — 임베딩 모델이 원본 표 구조(HTML)보다 자연어 문장에 훨씬 잘 반응한다. 다단/병합 헤더(`col_span`/`row_span`), 각주 마커(`*†‡`) 분리, 헤더 태깅이 없을 때의 휴리스틱 폴백을 처리.

### 대칭 컨텍스트 앵커링 (`evidence_chunker.context`)

표 위아래 bbox 거리(기본 300pt) + 임베딩 코사인 유사도를 통과한 단락만 `context_before`/`context_after`로 편입. 인접 페이지 경계(표가 페이지 상하단 15% 이내에 있을 때), 멀티컬럼 레이아웃(같은 컬럼에서 못 찾으면 컬럼 제한 없이 재탐색), 섹션 헤딩 경계(표와 단락 사이에 새 섹션이 시작되면 차단)를 함께 처리.

### small-to-big (`EvidenceUnit.retrieval_units`)

행 단위로 잘게 쪼갠 벡터(검색 정밀도) + 매칭되면 EU 전체(`parent_text`, LLM 컨텍스트)를 반환하는 2단 구조. 모든 하위 유닛(행/문맥/각주) 앞에 캡션을 접두사로 주입해 표 정체성을 앵커링(Anthropic Contextual Retrieval과 동일 패턴) — 캡션 없이 순수 텍스트로만 떠 있으면 벡터 공간에서 "이게 어느 표 얘기인지" 정체성을 잃는 문제(context_only 유형에서 확인)를 막는다.

## 알려진 이슈 / 진행 중인 문제

### 스캔본(OCR 필요) PDF — 현재 설정으로는 처리 불가

`parser/docling.py`는 `do_ocr=False`로 하드코딩되어 있음. 텍스트 레이어 없는 스캔본 PDF로 테스트한 결과:
- `do_ocr=False` (현재): 표는 감지되나 셀·텍스트 전부 0개 (빈 껍데기)
- `do_ocr=True`로 켜면 구조는 살아나지만, 기본 OCR 엔진(RapidOCR)의 한국어 인식 품질이 실사용 불가 수준으로 나쁨

한국어 전용 OCR 설정이 필요해 보이나 별도 작업으로 분리 필요. **미해결.**

### 큰 PDF 처리 시 메모리 부족 — 페이지 수가 아니라 누적 메모리가 원인

문서 종류·길이와 무관하게 Docling 변환 중 **누적 메모리가 일정 수준을 넘으면 `std::bad_alloc`** 발생. Docling 파이프라인이 페이지 처리마다 메모리를 누적하고 해제하지 않는 문제(메모리 누수)로 추정됨.

- TableFormer를 FAST 모드로 바꿔도 동일 → 표 인식 모델 문제 아님
- `page_range`로 앞부분만 잘라도 같은 프로세스 안에서는 여전히 발생 → 프로세스 단위로 나눠 처리해야 회피 가능할 것으로 추정
- 재측정 결과, "15페이지부터"라는 이전 서술은 부정확했음. 16페이지 문서(`bert_paper.pdf`)가 **12페이지 지점**에서 실패했고, 그 직전 프로세스 메모리는 **2.3GB**까지 증가한 상태였음(9페이지 문서는 정상, 15페이지 문서는 정상, 16페이지 문서는 실패 — 페이지 수 자체가 아니라 그 문서를 파싱하는 동안 누적된 메모리 총량이 임계임을 시사).
- **현재 검증 범위: 15페이지 이하 텍스트 레이어 PDF.** 위 ablation 표도 9페이지 문서 기준. 15페이지 문서(`attention_is_all_you_need.pdf`)는 파싱은 되지만 자동 벤치마크에는 아직 포함 안 됨.
- 원인 진단만 해두고 **아직 해결 안 됨**. PDF 50종 이상 규모 벤치마크(대부분 15페이지 이상 예상)에 영향을 줄 수 있어 공유해둠. 캡션 매핑 알고리즘과는 무관한 별개의 인프라/리소스 문제.

### 인터페이스 설계 미결정 사항

`flattened_rows`, `table_abstract`를 EvidenceUnit 인터페이스에 계속 포함할지, 별도 모듈로 뺄지 아직 논의 중 (현재는 포함된 상태 유지).

### `table_abstract`의 유사도-레퍼런스 폴백이 아직 도달하지 않음

`context.py`의 `attach_context_paragraphs`는 캡션이 없는 표에서 `table_abstract`를 유사도 필터의 레퍼런스 텍스트로 쓰도록 설계돼 있으나, EU 빌더의 현재 호출 순서(`attach_context_paragraphs` → `table_abstract` 계산)상 이 경로에 실제로 도달하지 않음 — 이 순서는 벤치마크 수치(Recall@1 0.80)를 만든 구현을 그대로 채택한 결과. 순서를 바꾸면 이 폴백은 살아나지만 대신 `table_abstract`에 섹션 헤더가 포함되지 않게 되는 트레이드오프가 있고(순환 의존), 검증하려면 캡션 없는 표가 있는 PDF가 필요함. 현재 파싱 가능한 두 PDF(9p, 15p) 모두 캡션 없는 표가 0개라 검증 수단이 없어 보류 — 검증 데이터 확보 후 처리 예정.

### `to_llamaindex`/`to_llamaindex_units` 출력 diff 미검증

`langchain`/`langchain_units` 경로는 리팩터링 전후 코퍼스 스냅샷으로 완전 일치를 확인했으나, `llama-index-core`가 개발 환경에 설치돼 있지 않아 `to_llamaindex`/`to_llamaindex_units`는 import 구조와 `ImportError` 경로만 확인했고 실제 출력 검증은 못 함.
