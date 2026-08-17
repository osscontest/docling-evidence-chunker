"""
캡션 예외처리 검증 — 캡션 없는 표 / 복수 캡션 / 다음(이전) 페이지 캡션.

배경:
    benchmarks/report_caption_mapping.py로 훑어본 실제 테스트 PDF(영어 논문 2 +
    한국어 보고서 + GPT-3, 표 23개) 어디에서도 multi_caption / cross_page 케이스가
    발동한 적이 없다. 실 데이터로 재현이 안 되므로 합성 문서를 만들어
    caption.map_table_caption()의 예외 처리 분기를 직접 검증한다.

caption.py가 parser.base.ParsedDoc/TextBlock/TableBlock을 직접 받으므로
합성 문서도 그 정식 모델로 조립한다 — DoclingDocument의 model_dump()
인터페이스를 모방하는 가짜 클래스가 필요 없다.

좌표는 전부 TOPLEFT(파서 경계에서 정규화된 값. parser/base.py 참고):
y가 작을수록 페이지 위쪽, 클수록 아래쪽. 원본 BOTTOMLEFT 테스트의 숫자를
그대로 뒤집는 대신, 각 케이스가 뜻하는 물리적 배치(표 위/아래, 페이지
경계)를 TOPLEFT로 새로 표현했다 — 산술 변환 실수보다 의미를 다시 쓰는
쪽이 검증하기 쉽다.
"""
from evidence_chunker.caption import map_table_caption
from evidence_chunker.parser.base import BBox, BlockLabel, ParsedDoc, TableBlock, TextBlock


# ---------------------------------------------------------------------------
# 빌더 헬퍼
# ---------------------------------------------------------------------------

def _text(index: int, text: str, label: str, page_no: int, t: float, b: float, l: float = 0.0, r: float = 100.0) -> TextBlock:
    return TextBlock(index=index, text=text, label=BlockLabel(label), page_no=page_no, bbox=BBox(l, t, r, b))


def _table(page_no: int, t: float, b: float, caption_refs: list[int] | None = None) -> TableBlock:
    return TableBlock(
        index=0, page_no=page_no, bbox=BBox(0.0, t, 100.0, b),
        cells=[], num_rows=0, num_cols=0, html=None,
        caption_refs=caption_refs or [],
    )


def _parsed(texts: list[TextBlock], page_height: float | None = None) -> ParsedDoc:
    page_sizes = {pg: (612.0, page_height) for pg in range(1, 6)} if page_height else {}
    return ParsedDoc(texts=texts, tables=[], picture_caption_refs=set(), page_sizes=page_sizes)


# ---------------------------------------------------------------------------
# 케이스 1: 캡션 없는 표
# ---------------------------------------------------------------------------

def test_no_caption():
    texts = [
        _text(0, "이 표는 실험 결과를 나타내지 않는다.", "text", page_no=1, t=350, b=380),
    ]
    parsed = _parsed(texts)
    table = _table(page_no=1, t=400, b=600, caption_refs=[])

    m = map_table_caption(parsed, table, table_index=0)
    assert m.confidence == "none"
    assert m.caption_text is None
    assert m.multi_caption is False


# ---------------------------------------------------------------------------
# 케이스 2: 복수 캡션 병합
# ---------------------------------------------------------------------------

def test_multi_caption():
    texts = [
        _text(0, "Table 1: 지역별 매출 비교", "caption", page_no=1, t=370, b=390),
        _text(1, "표 1. (전년 대비 증감률 포함)", "caption", page_no=1, t=390, b=400),
    ]
    parsed = _parsed(texts)
    table = _table(page_no=1, t=400, b=600, caption_refs=[0, 1])

    m = map_table_caption(parsed, table, table_index=0)
    assert m.confidence == "direct"
    assert m.multi_caption is True
    assert "지역별 매출 비교" in m.caption_text and "전년 대비 증감률" in m.caption_text
    assert "0" in m.caption_ref and "1" in m.caption_ref


# ---------------------------------------------------------------------------
# 케이스 3a: 캡션이 이전 페이지 맨 아래에 있는 경우
# ---------------------------------------------------------------------------

def test_caption_on_previous_page():
    """표가 페이지 최상단에서 시작(near_top 게이트) -> 같은 페이지 bbox
    fallback 실패 -> 이전 페이지 맨 아래(TOPLEFT: cy 최댓값)에서 재탐색."""
    texts = [
        _text(0, "본문 마지막 문단입니다.", "text", page_no=1, t=700, b=710),
        _text(1, "Table 2: 국가별 GDP 성장률 추이", "caption", page_no=1, t=720, b=740),
        _text(2, "표 아래 설명 텍스트", "text", page_no=2, t=390, b=400),
    ]
    parsed = _parsed(texts, page_height=800.0)
    # page_height 800 기준 near_top: table_top_y <= 120. 표가 페이지 2 최상단에서 시작.
    table = _table(page_no=2, t=20, b=370, caption_refs=[])

    m = map_table_caption(parsed, table, table_index=1)
    assert m.confidence == "inferred"
    assert m.cross_page is True
    assert m.caption_text == "Table 2: 국가별 GDP 성장률 추이"


# ---------------------------------------------------------------------------
# 케이스 3b: 캡션이 다음 페이지 맨 위에 있는 경우
# ---------------------------------------------------------------------------

def test_caption_on_next_page():
    """표가 페이지 최하단에서 끝남(near_bottom 게이트) -> 다음 페이지
    맨 위(TOPLEFT: cy 최솟값)에서 재탐색."""
    texts = [
        _text(0, "Figure 3: 손실 함수 수렴 곡선", "caption", page_no=3, t=40, b=60),
        _text(1, "본문 이어지는 내용", "text", page_no=3, t=90, b=110),
    ]
    parsed = _parsed(texts, page_height=800.0)
    # near_bottom: table_bot_y >= 680. 표가 페이지 2 최하단에서 끝남.
    table = _table(page_no=2, t=430, b=780, caption_refs=[])

    m = map_table_caption(parsed, table, table_index=2)
    assert m.confidence == "inferred"
    assert m.cross_page is True
    assert m.caption_text == "Figure 3: 손실 함수 수렴 곡선"


# ---------------------------------------------------------------------------
# 케이스 3c (회귀 방지): 표가 페이지 경계 근처가 아니면 인접 페이지를 보지 않아야 함
# ---------------------------------------------------------------------------

def test_middle_of_page_ignores_adjacent_caption():
    """실제 버그 사례: GPT-3 논문에서 목차(ToC)가 표로 오인식된 케이스.
    표가 페이지 중간~전체를 차지하는데, 다음 페이지에 있는 완전히 무관한
    캡션을 게이팅 없이는 잘못 채택했었음."""
    texts = [
        _text(0, "Figure 9: 무관한 그림 설명", "caption", page_no=3, t=40, b=60),
    ]
    parsed = _parsed(texts, page_height=800.0)
    # 표가 페이지 중간을 차지(near_top/near_bottom 둘 다 미충족) -> 게이팅돼야 함
    table = _table(page_no=2, t=200, b=600, caption_refs=[])

    m = map_table_caption(parsed, table, table_index=3)
    assert m.confidence == "none"
    assert m.caption_text is None
    assert m.cross_page is False


# ---------------------------------------------------------------------------
# 케이스 3d (회귀 방지): 본문 중간에 번호가 우연히 등장해도 캡션으로 오인하면 안 됨
# ---------------------------------------------------------------------------

def test_narrative_paragraph_not_mistaken_for_caption():
    """실제 버그 사례: GPT-3 논문 부록 표에서 captions 참조가 비어있어
    bbox fallback이 발동했는데, 근처의 서술형 문단 전체("... Figure 2.2 ...")가
    캡션으로 잘못 채택됨 (문단 중간에 번호가 우연히 있었음)."""
    texts = [
        _text(
            0,
            "This appendix contains the calculations that were used to derive "
            "the approximate compute used to train the language models in "
            "Figure 2.2. As a simplifying assumption, we ignore the attention "
            "operation, as it typically uses less than 10% of the total compute.",
            "text", page_no=5, t=350, b=390,
        ),
    ]
    parsed = _parsed(texts)  # page_height 없음 -> 인접 페이지 탐색 자체가 비활성
    table = _table(page_no=5, t=400, b=600, caption_refs=[])

    m = map_table_caption(parsed, table, table_index=4)
    assert m.confidence == "none"
    assert m.caption_text is None


# ---------------------------------------------------------------------------
# 케이스 3e: 부록 전용 문자.숫자 캡션 번호 체계 ("Table C.1", "Figure G.4")
# ---------------------------------------------------------------------------

def test_appendix_letter_number_caption():
    """실제 버그 사례: GPT-3 논문 부록(45~63p, 표 35개)이 전부 이 형식을 쓰는데,
    숫자 앞 문자 하나를 정규식이 허용하지 않아 전부 "none"으로 빠졌었음."""
    texts = [
        _text(0, "Table C.1: Overlap statistics for all datasets sorted from dirtiest to cleanest.",
              "caption", page_no=45, t=630, b=650),
    ]
    parsed = _parsed(texts)
    table = _table(page_no=45, t=650, b=850, caption_refs=[0])

    m = map_table_caption(parsed, table, table_index=0)
    assert m.confidence == "direct"
    assert m.caption_text == "Table C.1: Overlap statistics for all datasets sorted from dirtiest to cleanest."


# ---------------------------------------------------------------------------
# 케이스 3f (3-13): captions 참조가 그림(Figure) 캡션을 direct로 가리키는 경우
# ---------------------------------------------------------------------------

def test_direct_ref_pointing_to_figure_caption_is_downgraded():
    """실제 버그 사례: ieee1.pdf에서 표의 captions 참조가 "Fig. 3. Framework of
    ISAC technologies..."를 가리키는데도 confidence="direct"로 확정돼버렸음.
    EvidenceUnit이 이 잘못된 캡션을 모든 하위 유닛(행/문맥/주석)에 접두사로
    증폭시키면서 Recall -48.5pp 붕괴로 이어졌던 사고(3-11/3-12 참고).

    수정: direct 참조 검증은 _looks_like_table_caption(table/표 전용)을 쓴다.
    그림 캡션은 direct로 인정되지 않고 bbox fallback으로 넘어가며, fallback은
    기존 넓은 패턴을 그대로 쓰므로 텍스트 자체는 잃지 않고 confidence만
    "direct" -> "inferred"로 정직해진다."""
    texts = [
        _text(0, "Fig. 3. Framework of ISAC technologies for future wireless systems.",
              "caption", page_no=1, t=370, b=390),
    ]
    parsed = _parsed(texts)
    table = _table(page_no=1, t=400, b=600, caption_refs=[0])

    m = map_table_caption(parsed, table, table_index=0)
    # direct로 잘못 신뢰하지 않음
    assert m.confidence == "inferred"
    # caption_text는 bbox fallback으로 보존됨 (정보 손실 없음)
    assert m.caption_text == "Fig. 3. Framework of ISAC technologies for future wireless systems."
    # caption_ref is None (direct 경로가 아니므로)
    assert m.caption_ref is None


def test_direct_ref_to_real_table_caption_unaffected():
    """회귀 방지: 정상 표 캡션은 3-13 이후에도 direct 유지."""
    texts = [
        _text(0, "Table 5: 정상 표 캡션입니다.", "caption", page_no=1, t=370, b=390),
    ]
    parsed = _parsed(texts)
    table = _table(page_no=1, t=400, b=600, caption_refs=[0])

    m = map_table_caption(parsed, table, table_index=0)
    assert m.confidence == "direct"
    assert m.caption_text == "Table 5: 정상 표 캡션입니다."


def test_multi_caption_excludes_figure_fragment():
    """3-13 부수 효과: 복수 캡션 중 그림 캡션 파편은 병합에서 제외."""
    texts = [
        _text(0, "Table 7: 진짜 표 캡션", "caption", page_no=1, t=370, b=390),
        _text(1, "Fig 7a. 잘못 섞여 들어온 그림 참조", "caption", page_no=1, t=390, b=400),
    ]
    parsed = _parsed(texts)
    table = _table(page_no=1, t=400, b=600, caption_refs=[0, 1])

    m = map_table_caption(parsed, table, table_index=0)
    assert m.confidence == "direct"
    assert m.caption_text == "Table 7: 진짜 표 캡션"
    assert m.multi_caption is True
