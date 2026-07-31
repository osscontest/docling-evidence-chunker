"""
STEP 8 (W3): 캡션 예외처리 검증 — 캡션 없는 표 / 복수 캡션 / 다음(이전) 페이지 캡션.

담당: 팀원 1 (캡션↔표 연결, EU 핵심 anchor 로직)

배경:
    07_caption_table_mapping_poc.py로 검증한 실제 테스트 PDF(영어 논문 2 +
    한국어 보고서 + GPT-3, 표 23개) 어디에서도 multi_caption / cross_page 케이스가
    실제로 발동한 적이 없었음 (README "아직 미검증" 항목). 실 데이터로 재현이 안 되므로
    합성(mock) Docling 문서를 만들어 _caption_mapper.map_table_caption()의 세 가지
    예외 처리 분기를 직접 검증한다.

    Docling 실제 객체 대신, resolve_ref()가 기대하는 최소 인터페이스
    (.model_dump() 반환, doc.texts / doc.tables 리스트)만 흉내낸 Fake 객체 사용.

Usage:
    python scripts/08_caption_exception_tests.py
"""
import sys
import os

if hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(__file__))

from _caption_mapper import map_table_caption


# ---------------------------------------------------------------------------
# Fake Docling 객체 (model_dump() 인터페이스만 흉내)
# ---------------------------------------------------------------------------

class FakeItem:
    def __init__(self, label, text, page_no, t, b, l=0.0, r=100.0):
        self._d = {
            "label": label,
            "text": text,
            "prov": [{"page_no": page_no, "bbox": {"l": l, "t": t, "r": r, "b": b}}],
        }

    def model_dump(self):
        return self._d


class FakeTable:
    def __init__(self, page_no, t, b, captions=None):
        self._d = {
            "prov": [{"page_no": page_no, "bbox": {"l": 0.0, "t": t, "r": 100.0, "b": b}}],
            "captions": captions or [],
        }

    def model_dump(self):
        return self._d


class FakePage:
    def __init__(self, height, width=612.0):
        self._d = {"size": {"width": width, "height": height}}

    def model_dump(self):
        return self._d


class FakeDoc:
    def __init__(self, texts, tables=None, page_height=None):
        self.texts = texts
        self.tables = tables or []
        # page_height가 주어지면 1~5페이지 모두 같은 크기라고 가정 (테스트 단순화용)
        self.pages = (
            {pg: FakePage(page_height) for pg in range(1, 6)} if page_height else {}
        )

    # resolve_ref()가 getattr(doc, "texts")[idx] 형태로 접근
    def __getitem__(self, idx):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 검증 헬퍼
# ---------------------------------------------------------------------------

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def section(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ---------------------------------------------------------------------------
# 케이스 1: 캡션 없는 표
# ---------------------------------------------------------------------------

def test_no_caption():
    section("케이스 1: 캡션 없는 표")

    texts = [
        FakeItem("text", "이 표는 실험 결과를 나타내지 않는다.", page_no=1, t=500, b=480),
    ]
    doc = FakeDoc(texts=texts)
    table = FakeTable(page_no=1, t=400, b=200, captions=[])

    m = map_table_caption(doc, table, table_index=0)
    check("confidence == none", m.confidence == "none", m.confidence)
    check("caption_text is None", m.caption_text is None, str(m.caption_text))
    check("multi_caption False", m.multi_caption is False)


# ---------------------------------------------------------------------------
# 케이스 2: 복수 캡션 병합
# ---------------------------------------------------------------------------

def test_multi_caption():
    section("케이스 2: 복수 캡션 병합")

    texts = [
        FakeItem("caption", "Table 1: 지역별 매출 비교", page_no=1, t=420, b=410),
        FakeItem("caption", "표 1. (전년 대비 증감률 포함)", page_no=1, t=410, b=400),
    ]
    doc = FakeDoc(texts=texts)
    captions = [{"cref": "#/texts/0"}, {"cref": "#/texts/1"}]
    table = FakeTable(page_no=1, t=400, b=200, captions=captions)

    m = map_table_caption(doc, table, table_index=0)
    check("confidence == direct", m.confidence == "direct", m.confidence)
    check("multi_caption == True", m.multi_caption is True)
    check(
        "두 캡션 텍스트 모두 병합됨",
        "지역별 매출 비교" in m.caption_text and "전년 대비 증감률" in m.caption_text,
        m.caption_text,
    )
    check(
        "caption_ref에 두 ref 모두 포함",
        "#/texts/0" in m.caption_ref and "#/texts/1" in m.caption_ref,
        m.caption_ref,
    )


# ---------------------------------------------------------------------------
# 케이스 3a: 캡션이 이전 페이지 맨 아래에 있는 경우
# ---------------------------------------------------------------------------

def test_caption_on_previous_page():
    section("케이스 3a: 캡션이 이전 페이지에 있는 경우 (표가 페이지 최상단에서 시작)")

    texts = [
        # 이전 페이지(1페이지) 맨 아래 두 줄 중 캡션 패턴에 맞는 것
        FakeItem("text", "본문 마지막 문단입니다.", page_no=1, t=100, b=90),
        FakeItem("caption", "Table 2: 국가별 GDP 성장률 추이", page_no=1, t=80, b=70),
        # 같은 페이지(2페이지)엔 캡션 패턴 텍스트 없음
        FakeItem("text", "표 아래 설명 텍스트", page_no=2, t=390, b=380),
    ]
    # 페이지 높이 800 기준, 표가 상위 15%(>=680) 안쪽에서 시작 -> 인접 페이지 탐색 게이트 통과
    doc = FakeDoc(texts=texts, page_height=800.0)
    # 같은 페이지 bbox fallback(200pt 이내)이 실패하도록 표 아래 텍스트와 충분히 멀리 둠
    table = FakeTable(page_no=2, t=750, b=400, captions=[])

    m = map_table_caption(doc, table, table_index=1)
    check("confidence == inferred", m.confidence == "inferred", m.confidence)
    check("cross_page == True", m.cross_page is True)
    check(
        "이전 페이지 캡션 텍스트 채택",
        m.caption_text == "Table 2: 국가별 GDP 성장률 추이",
        str(m.caption_text),
    )


# ---------------------------------------------------------------------------
# 케이스 3b: 캡션이 다음 페이지 맨 위에 있는 경우
# ---------------------------------------------------------------------------

def test_caption_on_next_page():
    section("케이스 3b: 캡션이 다음 페이지에 있는 경우 (표가 페이지 최하단에서 끝남)")

    texts = [
        # 3페이지 맨 위에 캡션
        FakeItem("caption", "Figure 3: 손실 함수 수렴 곡선", page_no=3, t=750, b=740),
        FakeItem("text", "본문 이어지는 내용", page_no=3, t=700, b=690),
    ]
    # 페이지 높이 800 기준, 표가 하위 15%(<=120) 안쪽에서 끝남 -> 게이트 통과
    doc = FakeDoc(texts=texts, page_height=800.0)
    # 표가 2페이지 맨 아래(b=50)에서 끝남 -> 같은 페이지 bbox fallback 실패해야 함
    table = FakeTable(page_no=2, t=200, b=50, captions=[])

    m = map_table_caption(doc, table, table_index=2)
    check("confidence == inferred", m.confidence == "inferred", m.confidence)
    check("cross_page == True", m.cross_page is True)
    check(
        "다음 페이지 캡션 텍스트 채택",
        m.caption_text == "Figure 3: 손실 함수 수렴 곡선",
        str(m.caption_text),
    )


# ---------------------------------------------------------------------------
# 케이스 3c (회귀 방지): 표가 페이지 경계 근처가 아니면 인접 페이지를 보지 않아야 함
# ---------------------------------------------------------------------------

def test_middle_of_page_ignores_adjacent_caption():
    section("케이스 3c: 표가 페이지 중간에 있으면 옆 페이지의 무관한 캡션을 붙이면 안 됨")
    # 실제 버그 사례: GPT-3 논문에서 목차(ToC)가 표로 오인식된 케이스.
    # 표가 페이지 중간~전체를 차지하는데, 다음 페이지에 있는 완전히 무관한
    # "Figure 1.1" 캡션을 게이팅 없이는 잘못 채택했었음.

    texts = [
        FakeItem("caption", "Figure 9: 무관한 그림 설명", page_no=3, t=750, b=740),
    ]
    doc = FakeDoc(texts=texts, page_height=800.0)
    # 표가 페이지 중간을 차지(위/아래 15% 경계 밖) -> 인접 페이지 탐색 자체가 게이팅돼야 함
    table = FakeTable(page_no=2, t=600, b=200, captions=[])

    m = map_table_caption(doc, table, table_index=3)
    check("confidence == none (무관 캡션 채택 안 함)", m.confidence == "none", m.confidence)
    check("caption_text is None", m.caption_text is None, str(m.caption_text))
    check("cross_page == False", m.cross_page is False)


# ---------------------------------------------------------------------------
# 케이스 3d (회귀 방지): 본문 중간에 번호가 우연히 등장해도 캡션으로 오인하면 안 됨
# ---------------------------------------------------------------------------

def test_narrative_paragraph_not_mistaken_for_caption():
    section("케이스 3d: 본문 문단 중간에 'Figure N'이 나와도 캡션으로 오인하면 안 됨")
    # 실제 버그 사례: GPT-3 논문 부록 표(table[20], p47)에서 captions RefItem이
    # 비어있어 bbox fallback이 발동했는데, 근처의 서술형 문단
    # "This appendix contains the calculations ... Figure 2.2. As a simplifying ..."
    # 전체가 캡션으로 잘못 채택됨 (문단 중간에 "Figure 2.2"가 우연히 있었음).

    texts = [
        FakeItem(
            "text",
            "This appendix contains the calculations that were used to derive "
            "the approximate compute used to train the language models in "
            "Figure 2.2. As a simplifying assumption, we ignore the attention "
            "operation, as it typically uses less than 10% of the total compute.",
            page_no=5, t=390, b=350,
        ),
    ]
    doc = FakeDoc(texts=texts)
    table = FakeTable(page_no=5, t=400, b=200, captions=[])

    m = map_table_caption(doc, table, table_index=4)
    check("confidence == none (서술형 문단을 캡션으로 오인하지 않음)", m.confidence == "none", m.confidence)
    check("caption_text is None", m.caption_text is None, str(m.caption_text))


# ---------------------------------------------------------------------------
# 케이스 3e: 부록 전용 문자.숫자 캡션 번호 체계 ("Table C.1", "Figure G.4")
# ---------------------------------------------------------------------------

def test_appendix_letter_number_caption():
    section("케이스 3e: 부록 캡션 번호 체계 (Table C.1, Figure G.4) 인식")
    # 실제 버그 사례: GPT-3 논문 부록(45~63p, 표 35개)이 전부 이 형식을 쓰는데,
    # 숫자 앞 문자 하나를 정규식이 허용하지 않아 전부 "none"으로 빠졌었음.
    # table[18]은 심지어 captions RefItem까지 정확히 있었는데도 놓쳤던 케이스.

    texts = [
        FakeItem(
            "caption",
            "Table C.1: Overlap statistics for all datasets sorted from dirtiest to cleanest.",
            page_no=45, t=650, b=630,
        ),
    ]
    doc = FakeDoc(texts=texts)
    captions = [{"cref": "#/texts/0"}]
    table = FakeTable(page_no=45, t=620, b=400, captions=captions)

    m = map_table_caption(doc, table, table_index=0)
    check("confidence == direct", m.confidence == "direct", m.confidence)
    check(
        "Table C.1 캡션 텍스트 채택",
        m.caption_text == "Table C.1: Overlap statistics for all datasets sorted from dirtiest to cleanest.",
        str(m.caption_text),
    )


# ---------------------------------------------------------------------------
# 케이스 3f (3-13): captions RefItem이 그림(Figure) 캡션을 direct로 가리키는 경우
# ---------------------------------------------------------------------------

def test_direct_ref_pointing_to_figure_caption_is_downgraded():
    section("케이스 3f: captions RefItem이 그림 캡션을 가리키면 direct로 신뢰하면 안 됨")
    # 실제 버그 사례: ieee1.pdf에서 표의 captions RefItem이 "Fig. 3. Framework of
    # ISAC technologies..."를 가리키는데도 confidence="direct"로 확정돼버렸음.
    # interfaces.py의 EU가 이 잘못된 캡션을 모든 하위 유닛(행/문맥/주석)에 접두사로
    # 증폭시키면서 Recall -48.5pp 붕괴로 이어졌던 사고(3-11/3-12 참고).
    #
    # 수정: direct RefItem 검증은 _looks_like_table_caption(table/표 전용)을 쓴다.
    # 그림 캡션은 direct로 인정되지 않고 bbox fallback으로 넘어가며, fallback은
    # 기존 넓은 패턴을 그대로 쓰므로 텍스트 자체는 잃지 않고 confidence만
    # "direct" -> "inferred"로 정직해진다.

    texts = [
        FakeItem(
            "caption",
            "Fig. 3. Framework of ISAC technologies for future wireless systems.",
            page_no=1, t=420, b=410,
        ),
    ]
    doc = FakeDoc(texts=texts)
    captions = [{"cref": "#/texts/0"}]
    table = FakeTable(page_no=1, t=400, b=200, captions=captions)

    m = map_table_caption(doc, table, table_index=0)
    check(
        "confidence == inferred (direct로 잘못 신뢰하지 않음)",
        m.confidence == "inferred", m.confidence,
    )
    check(
        "caption_text는 bbox fallback으로 보존됨 (정보 손실 없음)",
        m.caption_text == "Fig. 3. Framework of ISAC technologies for future wireless systems.",
        str(m.caption_text),
    )
    check("caption_ref is None (direct 경로가 아니므로)", m.caption_ref is None, str(m.caption_ref))


def test_direct_ref_to_real_table_caption_unaffected():
    section("케이스 3g (회귀 방지): 정상 표 캡션은 3-13 이후에도 direct 유지")

    texts = [
        FakeItem("caption", "Table 5: 정상 표 캡션입니다.", page_no=1, t=420, b=410),
    ]
    doc = FakeDoc(texts=texts)
    captions = [{"cref": "#/texts/0"}]
    table = FakeTable(page_no=1, t=400, b=200, captions=captions)

    m = map_table_caption(doc, table, table_index=0)
    check("confidence == direct", m.confidence == "direct", m.confidence)
    check(
        "caption_text 그대로 채택",
        m.caption_text == "Table 5: 정상 표 캡션입니다.",
        str(m.caption_text),
    )


def test_multi_caption_excludes_figure_fragment():
    section("케이스 3h (3-13 부수 효과): 복수 캡션 중 그림 캡션 파편은 병합에서 제외")
    # 3-13 이전에는 "Table 7: 진짜 표 캡션" + "Fig 7a. 잘못 섞여 들어온 그림 참조"가
    # 둘 다 캡션처럼 보여서 caption_text에 함께 병합됐을 것. 이제는 표 패턴만 인정.

    texts = [
        FakeItem("caption", "Table 7: 진짜 표 캡션", page_no=1, t=420, b=410),
        FakeItem("caption", "Fig 7a. 잘못 섞여 들어온 그림 참조", page_no=1, t=410, b=400),
    ]
    doc = FakeDoc(texts=texts)
    captions = [{"cref": "#/texts/0"}, {"cref": "#/texts/1"}]
    table = FakeTable(page_no=1, t=400, b=200, captions=captions)

    m = map_table_caption(doc, table, table_index=0)
    check("confidence == direct", m.confidence == "direct", m.confidence)
    check(
        "그림 캡션 파편 없이 표 캡션만 채택",
        m.caption_text == "Table 7: 진짜 표 캡션",
        str(m.caption_text),
    )
    check("multi_caption == True (RefItem 자체는 2개였으므로)", m.multi_caption is True)


def main():
    test_no_caption()
    test_multi_caption()
    test_caption_on_previous_page()
    test_caption_on_next_page()
    test_middle_of_page_ignores_adjacent_caption()
    test_narrative_paragraph_not_mistaken_for_caption()
    test_appendix_letter_number_caption()
    test_direct_ref_pointing_to_figure_caption_is_downgraded()
    test_direct_ref_to_real_table_caption_unaffected()
    test_multi_caption_excludes_figure_fragment()

    section("결과")
    print(f"  PASS {PASS} / FAIL {FAIL}")
    if FAIL:
        sys.exit(1)
    print("\n[DONE] 08_caption_exception_tests.py complete")


if __name__ == "__main__":
    main()
