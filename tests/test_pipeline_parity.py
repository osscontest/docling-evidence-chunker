"""
두 가지를 증명한다 (Stage 2 "빌더 통합" + "split 연결"의 실행 가능한 증거):

1. evidence_chunker.chunker.build_evidence_units() + split_oversized_units()를
   직접 합성한 결과가 chunk()가 낼 결과와 같은 모양이다 (빌더 결과를
   빠짐없이 포함 + 실제로 쪼개짐) — test_chunk_applies_split
2. SmartChunker.chunk()가 실제로 split_oversized_units를 호출한다 —
   test_chunk_wires_split (파싱 없이 배선만 검증)

두 개로 나눈 이유: chunk()는 경로를 받아 내부에서 직접 파싱하므로 외부
에서 미리 파싱한 doc을 주입할 방법이 없다. chunk()를 그대로 호출하면서
동시에 build_evidence_units()로 별도 비교군을 만들면 같은 프로세스 안에서
Docling 파싱이 두 번 일어나 메모리 누적으로 std::bad_alloc이 날 수 있다
(README "큰 PDF 처리 시 메모리 부족" 참고 — 실제로 전체 스위트 실행 중
재현됨). 그래서 1번은 파싱을 sample_doc 픽스처로 1회만 수행하고 chunk()를
직접 부르지 않으며, 2번은 아예 파싱을 monkeypatch로 걷어내고 배선만 본다.
"""
from evidence_chunker import SmartChunker
from evidence_chunker.chunker import build_evidence_units, split_oversized_units


def test_chunk_applies_split(sample_doc):
    raw = build_evidence_units(sample_doc)
    eus = split_oversized_units(raw)

    # 빌더 결과를 빠짐없이 포함 (두 경로가 갈라지지 않음)
    assert {e.eu_id for e in raw} <= {e.eu_id.rsplit("-s", 1)[0] for e in eus}

    # 이 PDF에는 512토큰 초과 표가 있다 — split이 실제로 쪼갬
    assert any(e.is_split for e in eus)
    assert len(eus) > len(raw)


def test_chunk_wires_split(monkeypatch):
    """chunk()가 split_oversized_units를 호출하는지 — 파싱 없이 배선만 검증."""
    called = []
    monkeypatch.setattr(
        "evidence_chunker.chunker.split_oversized_units",
        lambda eus, *a, **k: called.append(eus) or eus,
    )
    monkeypatch.setattr(SmartChunker, "_parse", lambda self, p: object())
    monkeypatch.setattr(
        "evidence_chunker.chunker.build_evidence_units",
        lambda doc, *a, **k: [],
    )

    SmartChunker().chunk("dummy.pdf")

    assert called, "chunk()가 split_oversized_units를 호출하지 않음"
