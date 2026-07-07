# docling-evidence-chunker

Docling 기반 Evidence Unit 청킹 → RAG 성능 향상

---

## 프로젝트 구조

```
docling-evidence-chunker/
├── interfaces.py              # EvidenceUnit 데이터클래스 (팀 공유 인터페이스, 데이터 구조만)
├── bbox_utils.py               # bbox 좌표 변환 유틸 (normalize_bbox, 팀원1)
├── context_attacher.py         # 인접 단락·섹션헤더 탐지 (팀원2, bbox 거리 기반 PoC)
├── langchain_wrapper.py       # LangChain / LlamaIndex 변환 래퍼 (팀원2)
├── download_test_pdfs.py      # 테스트 PDF 다운로드
├── download_models.py         # Docling 모델 로컬 다운로드 (Windows 심볼릭링크 우회)
├── run_all.py                 # 전체 탐색 파이프라인 실행기
├── scripts/
│   ├── _converter.py          # DocumentConverter 팩토리 (로컬 모델 경로)
│   ├── _table_utils.py        # 표 처리 유틸리티 (Row Flattening 등)
│   ├── _caption_mapper.py      # 캡션↔표 매핑 알고리즘 (팀원1, W2 PoC)
│   ├── 01_basic_parse.py      # Docling 전체 구조 덤프
│   ├── 02_explore_tables.py   # 표 필드 탐색 (bbox, captions, footnotes)
│   ├── 03_explore_texts.py    # 텍스트 요소 탐색 (label 종류, 거리 계산)
│   ├── 04_summary_report.py   # W1 탐색 보고서 생성
│   ├── 05_table_cells.py      # 표 셀 구조 + export_to_html 확인
│   ├── 06_build_eu.py         # EvidenceUnit 실제 구성 (메인 파이프라인)
│   └── 07_caption_table_mapping_poc.py  # 캡션↔표 매핑 검증 리포트 (W2 PoC)
├── data/
│   ├── pdfs/                  # 테스트 PDF (영어 논문 2종 + 한국어 보고서 + GPT-3 등)
│   └── outputs/               # 탐색 결과 JSON
├── models/
│   └── combined/              # 로컬 모델 (layout + table 병합)
└── reports/                   # W1 탐색 보고서
```

---

## 캡션↔표 매핑 PoC (`scripts/_caption_mapper.py`, W2, 팀원1)

`table.captions` RefItem을 파싱해 표-캡션을 1:1로 연결하는 알고리즘. 우선순위 3단계로 동작:

| confidence | 의미 |
|---|---|
| `direct` | `captions` RefItem이 가리키는 텍스트가 캡션 패턴(`Table N`, `표 N` 등)에 맞아 그대로 채택 |
| `inferred` | RefItem이 없거나 엉뚱한 파편을 가리켜서, 표 위/아래 bbox 거리(200pt 이내)에서 캡션처럼 생긴 텍스트를 재탐색해 찾음 |
| `none` | 둘 다 실패, 캡션 없음 |

```python
from _caption_mapper import map_table_caption, map_all_captions, validate_mapping

mapping = map_table_caption(doc, table, table_index)
# CaptionMapping(caption_text=..., confidence="direct"|"inferred"|"none", multi_caption=..., cross_page=...)
```

`06_build_eu.py`가 이 모듈을 직접 호출해 EU의 `caption_text` / `caption_confidence`를 채움.
검증 리포트: `python scripts/07_caption_table_mapping_poc.py --all`

### 한국어 PDF 테스트로 발견한 버그 2건

영어 논문 2종만으로는 100% 매핑돼서 예외 케이스가 0건 — 알고리즘이 좋아서가 아니라 테스트셋이 너무 쉬워서였을 가능성이 있었음. 한국은행 잠재성장률 보고서(한국어)와 GPT-3 논문을 테스트셋에 추가(`download_test_pdfs.py`)해서 검증한 결과, 실제 버그 2건을 발견하고 수정함:

- RefItem이 캡션이 아닌 파편("(단위: 1), %)")을 가리키는 경우 → 캡션 패턴 검증 후 `inferred`로 재탐색해 정정
- RefItem 자체가 비어있는 경우 → bbox fallback으로 `<표 N>` 형식의 실제 캡션을 찾아냄

전체 4개 PDF(영어 2 + 한국어 1 + GPT-3) 기준 표 23개 중 20개 매핑 성공 (87%), confidence 분포(`direct`/`inferred`/`none`)까지 리포트에 포함.

**아직 미검증**: `multi_caption`(캡션 2개 이상), `cross_page`(캡션이 다른 페이지) 감지 로직은 구현돼 있으나, 지금까지 테스트한 PDF 어디서도 실제로 발동한 적이 없어 검증되지 않은 상태.

---

## 알려진 이슈 / 진행 중인 문제

### 스캔본(OCR 필요) PDF — 현재 설정으로는 처리 불가

`_converter.py`는 `do_ocr=False`로 하드코딩되어 있음. 텍스트 레이어 없는 스캔본 PDF로 테스트한 결과:
- `do_ocr=False` (현재): 표는 감지되나 셀·텍스트 전부 0개 (빈 껍데기)
- `do_ocr=True`로 켜면 구조는 살아나지만, 기본 OCR 엔진(RapidOCR)의 한국어 인식 품질이 실사용 불가 수준으로 나쁨

한국어 전용 OCR 설정이 필요해 보이나 별도 작업으로 분리 필요. **미해결.**

### 큰 PDF(15페이지 이상) 처리 시 메모리 부족

15페이지가 넘는 PDF(어텐션 논문 자체, GPT-3 75p, OECD 통계 부록 69p 등)를 변환하면 문서 종류·길이와 무관하게 **항상 15페이지 근처부터 `std::bad_alloc`** 발생. Docling 파이프라인이 페이지 처리마다 메모리를 누적하고 해제하지 않는 문제(메모리 누수)로 추정됨.

- TableFormer를 FAST 모드로 바꿔도 동일 → 표 인식 모델 문제 아님
- `page_range`로 앞부분만 잘라도 같은 프로세스 안에서는 여전히 발생 → 프로세스 단위로 나눠 처리해야 회피 가능할 것으로 추정
- 원인 진단만 해두고 **아직 해결 안 됨**. 팀원3의 W7 벤치마크(PDF 50종, 대부분 15페이지 이상 예상)에 영향을 줄 수 있어 공유해둠. 캡션 매핑 알고리즘과는 무관한 별개의 인프라/리소스 문제.

### 인터페이스 설계 미결정 사항

`flattened_rows`, `table_abstract`를 EvidenceUnit 인터페이스에 계속 포함할지, 별도 모듈로 뺄지 아직 팀 논의 중 (현재는 포함된 상태 유지).
