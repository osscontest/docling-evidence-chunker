# docling-evidence-chunker

Docling 기반 Evidence Unit 청킹 → RAG 성능 향상

---

## 프로젝트 구조

```
docling-evidence-chunker/
├── interfaces.py              # EvidenceUnit 데이터클래스 (팀 공유 인터페이스)
├── download_test_pdfs.py      # 테스트 PDF 다운로드
├── download_models.py         # Docling 모델 로컬 다운로드 (Windows 심볼릭링크 우회)
├── run_all.py                 # 전체 탐색 파이프라인 실행기
├── scripts/
│   ├── _converter.py          # DocumentConverter 팩토리 (로컬 모델 경로)
│   ├── _table_utils.py        # 표 처리 유틸리티 (Row Flattening 등)
│   ├── 01_basic_parse.py      # Docling 전체 구조 덤프
│   ├── 02_explore_tables.py   # 표 필드 탐색 (bbox, captions, footnotes)
│   ├── 03_explore_texts.py    # 텍스트 요소 탐색 (label 종류, 거리 계산)
│   ├── 04_summary_report.py   # W1 탐색 보고서 생성
│   ├── 05_table_cells.py      # 표 셀 구조 + export_to_html 확인
│   └── 06_build_eu.py         # EvidenceUnit 실제 구성 (메인 파이프라인)
├── data/
│   ├── pdfs/                  # 테스트 PDF (Transformer, BERT 논문)
│   └── outputs/               # 탐색 결과 JSON
├── models/
│   └── combined/              # 로컬 모델 (layout + table 병합)
└── reports/                   # W1 탐색 보고서
```


---

## 원본 인터페이스에서 변경된 점

원본: 팀 공유 `interface-main` (26.06.29 초안)

### 필드 추가

| 필드 | 추가 이유 |
|------|-----------|
| `flattened_rows: list[str]` | 임베딩 모델이 표 구조보다 자연어 문장에 더 잘 반응 → Recall@1 향상 핵심 |
| `table_abstract: Optional[str]` | 광범위 질의("이 표가 뭘 담고 있어?")에 대한 검색 커버리지 확보 |
| `caption_confidence: str` | 캡션 연결 방식(직접/추정)을 기록해 오답 분석에 활용 |

### 메서드 추가

| 메서드 | 내용 |
|--------|------|
| `to_langchain_document()` | LangChain `Document` 객체로 직접 변환. `metadata`에 eu_id, bbox, confidence 포함 |

### `text` property 조립 순서 변경

원본:
```
section_header → context_before → caption_text → table_html → footnote_text → context_after
```

변경 후:
```
section_header → context_before → caption_text → table_abstract → table_html → flattened_rows → footnote_text → context_after
```

### 필드 설명 보강

| 필드 | 원본 | 변경 |
|------|------|------|
| `table_html` | `table.export_to_html()` | `table.export_to_html(doc)` — **`doc` 인자 필수** (없으면 빈 문자열) |
| `bbox` | "0~1 normalized (Docling 기본 좌표계)" | Docling 원본은 PDF 포인트(BOTTOMLEFT). `normalize_bbox()`로 변환 필요 |
| `context_before/after` | "bbox + 임베딩 통과" | 현재 bbox 300pt 이내 단락 수집. 임베딩 필터는 추후 추가 예정 |

### 유틸 함수 추가 (`interfaces.py`)

```python
normalize_bbox(bbox, page_width, page_height) -> tuple
# Docling PDF 포인트(BOTTOMLEFT) → 0~1 normalized 변환
```

---

## EvidenceUnit 인터페이스

`interfaces.py` 참고. 표 하나당 EU 하나 생성.

```python
from interfaces import EvidenceUnit

eu = EvidenceUnit(
    eu_id="eu-p6-0",          # "eu-p{페이지}-{인덱스}"
    page_no=6,
    section_header="4 Experiments",
    caption_text="Table 1: ...",
    table_html="<table>...</table>",
    flattened_rows=["BERT LARGE의 MNLI는 86.7이다.", ...],
    table_abstract="[4 Experiments] Table 1: ... 열: MNLI, QQP ... 총 4행.",
    footnote_text=None,
    context_before=[],
    context_after=["The results show that..."],
    bbox=(0.12, 0.82, 0.88, 0.93),   # 0~1 normalized, BOTTOMLEFT
    caption_confidence="high",        # "high" | "low"
)

# 임베딩/LangChain에 넘길 텍스트
print(eu.text)

# LangChain Document로 변환
doc = eu.to_langchain_document()
```

### EvidenceUnit 필드 요약

| 필드 | 타입 | 설명 |
|------|------|------|
| `eu_id` | `str` | 고유 ID (`"eu-p3-0"` = 3페이지 첫 번째 EU) |
| `page_no` | `int` | 페이지 번호 |
| `section_header` | `Optional[str]` | 표가 속한 섹션 제목 |
| `caption_text` | `Optional[str]` | 표 제목 |
| `table_html` | `Optional[str]` | 표 HTML (`table.export_to_html(doc)`) |
| `footnote_text` | `Optional[str]` | 표 아래 주석 |
| `context_before` | `list[str]` | 표 위쪽 설명 단락 (bbox 300pt 이내) |
| `context_after` | `list[str]` | 표 아래쪽 설명 단락 (bbox 300pt 이내) |
| `flattened_rows` | `list[str]` | 셀별 자연어 문장 (임베딩 Recall@1 향상용) |
| `table_abstract` | `Optional[str]` | 표 전체 요약 (광범위 질의용) |
| `bbox` | `tuple[float,float,float,float]` | 표 위치 (x1,y1,x2,y2), 0~1 normalized |
| `caption_confidence` | `str` | `"high"` / `"low"` (캡션 연결 신뢰도) |
| `is_split` | `bool` | 행 분할 여부 (팀원 3 작업과 연동) |
| `split_index` | `Optional[int]` | 분할 순서 (0-based) |
| `total_splits` | `Optional[int]` | 총 분할 수 |
| `text` *(property)* | `str` | 임베딩용 최종 텍스트 (자동 조립, 직접 넣지 말 것) |

---

## 구현된 핵심 기능 (_table_utils.py)

### 1. Row Flattening (셀 → 자연어 문장)

임베딩 모델은 표 구조보다 자연어에 더 잘 반응 → Recall@1 향상 핵심 항목.

```python
# 입력: Docling table_cells
# 출력: 자연어 문장 목록
flatten_to_sentences(cells, num_rows, num_cols, footnote_text)

# 예시 출력
"BERT LARGE의 MNLI-(m/mm) 392k는 86.7/85.9이다."
"Self-Attention의 Complexity per Layer는 O(n²·d)이다."
```

### 2. 다단/병합 헤더 처리

`col_span` / `row_span` 필드를 읽어 헤더 범위를 열 전체에 전파.
다단 헤더는 `"분기 / Q1"` 형태로 이어붙임.

```python
build_col_header_map(cells, num_cols)  # col_span 처리
build_row_header_map(cells, num_rows)  # row_span 처리
```

### 3. 각주 마커 셀 단위 전파

`*†‡` 마커가 있는 셀에만 해당 각주 텍스트를 붙임.
행 분할 시 맥락 손실 방지.

```python
detect_cell_marker("100*")  # → ("100", "*")
# 문장: "APAC의 Q2는 100이다. (주: 잠정치)"
```

### 4. 헤더 추론 폴백

Docling이 헤더 태깅을 못 했을 때 첫 행 패턴으로 자동 추론.
전부 숫자면 `Col0/Col1/...` 로 대체.

```python
infer_headers_fallback(cells, num_cols)
```

### 5. Table Abstract (멀티그래뉼래리티 검색)

표 전체 요약을 별도 필드로 생성. LLM 없이 규칙 기반.
- 광범위 질의 ("지역별 매출 표가 어디 있어?") → abstract가 강하게 반응
- 구체적 질의 ("Q2 APAC 매출이 얼마야?") → flattened_rows가 강하게 반응

```python
build_table_abstract(caption_text, col_map, num_rows, section_header)
# "[4 Experiments] Table 1: GLUE Test results... 열: MNLI, QQP... 총 4행 데이터."
```

### 6. 캡션 신뢰도 메타데이터

```python
caption_confidence = "high"  # captions RefItem으로 직접 연결
caption_confidence = "low"   # 캡션 없거나 bbox 거리로 추정
```

RAG 정답률 분석 시 low EU에서 오답이 집중되는지 추적 가능.

---

## W1 탐색 주요 발견 사항

### captions 필드가 항상 채워져 있는가?

테스트 PDF 2종(Transformer, BERT 논문) 기준 **모든 표에 captions 있음**.
단, 논문·공시 자료 등 실제 PDF에서는 없는 케이스가 존재할 수 있어 방어 코드 필수.

- 캡션 있을 때: `table.captions` RefItem의 `cref` 역참조로 텍스트 추출 → `caption_confidence = "high"`
- 캡션 없을 때: `doc.texts` 내 `label=caption` 원소를 bbox 거리로 매칭하는 폴백 필요 → `caption_confidence = "low"`

### bbox 좌표계

**PDF 포인트 (pt) 단위, 원점은 페이지 왼쪽 하단 (BOTTOMLEFT).**

- 페이지 크기: Letter 기준 612 × 792 pt (72 dpi)
- `t` (top) > `b` (bottom) — y축이 아래에서 위로 증가하므로 시각적 상단이 더 큰 값
- 0~1 정규화가 필요하면 `normalize_bbox(bbox, page_width, page_height)` 사용

### 텍스트 요소 label 종류

Transformer 논문 기준 8종 확인:

| label | 설명 | EU 활용 |
|-------|------|---------|
| `text` | 일반 문단 | context_before / after |
| `section_header` | 섹션 제목 (`level` 필드 포함) | section_header 필드 |
| `caption` | 표·그림 캡션 | caption_text 폴백 탐색 |
| `list_item` | 리스트 항목 | context_before / after |
| `footnote` | 각주 | footnote_text |
| `formula` | 수식 | 미사용 (아래 예외 참고) |
| `page_header` | 페이지 상단 | 제외 |
| `page_footer` | 페이지 하단 | 제외 |

### 표와 캡션이 다른 페이지에 있는 케이스가 있는가?

테스트 PDF 2종 기준 **없음 — 모두 같은 페이지**.
단, 실제 PDF에서 발생 가능. bbox 거리 계산보다 **`captions` RefItem 역참조 방식이 크로스 페이지에서도 안전**하게 동작함.

### 탐색 중 발견한 예외 케이스

| 예외 | 내용 | 대응 |
|------|------|------|
| `formula` label `text` 빈 문자열 | 수식이 plain text로 변환 안 됨 | EU 구성 시 formula 원소 제외 |
| `export_to_html()` 빈 문자열 반환 | `doc` 인자 없으면 항상 빈 값 | `table.export_to_html(doc)` 필수 |
| `coord_origin=BOTTOMLEFT` y축 반전 | t > b이므로 "위"가 더 큰 값 | 거리 계산 시 방향 주의, `normalize_bbox()` 사용 |
| Windows 심볼릭링크 권한 오류 | HuggingFace Hub 모델 다운로드 실패 | `models/combined/` 로컬 경로로 우회 |
| `column_header` 태깅 누락 | Docling이 헤더를 못 잡는 케이스 | `infer_headers_fallback()` 폴백으로 첫 행 추정 |

