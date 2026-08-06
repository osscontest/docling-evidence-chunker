"""
Docling DocumentConverter 팩토리.
"""

from pathlib import Path


MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"


def make_converter(do_ocr: bool | None = None, use_local_models: bool = False):
    """DocumentConverter 생성.

    Parameters
    ----------
    do_ocr :
        None(기본)이면 **Docling 기본값을 그대로 따른다** — 우리가 값을 넘기지
        않는다. OCR 은 속도·품질 트레이드오프라 라이브러리가 대신 정할 문제가
        아니고, 여기서 True 로 박아두면 Docling 이 나중에 기본값을 바꿔도
        우리가 그걸 무시하게 된다. 켜거나 끄고 싶으면 호출부에서 명시할 것.
    use_local_models :
        `models/combined` 의 로컬 아티팩트 사용 여부. 기본 False —
        이 경로가 일부 PDF 7페이지에서 std::bad_alloc 으로 표를 통째로 날린다
        (benchmarks/measure_recall_single.py 주석). 필요할 때만 opt-in.
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    artifacts_path = None
    if use_local_models:
        # combined/ = layout(model.safetensors) + table(model_artifacts/) 병합본
        candidate = MODELS_DIR / "combined"
        artifacts_path = candidate if candidate.exists() else None

    # do_table_structure 만 고정한다. 표 구조 추출이 꺼지면 Evidence Unit 을
    # 만들 수 없어서 라이브러리 자체가 성립하지 않으므로 이건 '선호'가 아니라
    # '요구사항'이다. 반대로 do_ocr 은 선호라서 넘기지 않는다.
    opts = {"do_table_structure": True, "artifacts_path": artifacts_path}
    if do_ocr is not None:
        opts["do_ocr"] = do_ocr

    return DocumentConverter(
        format_options={
            "pdf": PdfFormatOption(pipeline_options=PdfPipelineOptions(**opts))
        }
    )


# ---------------------------------------------------------------------------
# DoclingParser — DoclingDocument -> ParsedDoc(Docling 무관 내부 모델)
#
# 아직 build_evidence_units() 등 알고리즘 쪽은 이걸 쓰지 않는다(그쪽은
# 여전히 raw DoclingDocument를 직접 만짐) — 파서 구현만 먼저 잠그고,
# 각 알고리즘 모듈을 ParsedDoc 기반으로 전환하는 건 뒤따르는 커밋들의 몫.
# ---------------------------------------------------------------------------

def _text_index_from_cref(cref: str) -> "int | None":
    """'#/texts/12' -> 12. Docling의 cref 인덱스가 doc.texts 리스트 위치와
    그대로 일치한다는 걸 resolve_ref()류의 경로 순회 로직으로 확인했음
    (parts=["texts","12"] -> doc.texts[12])."""
    try:
        parts = cref.strip("#/").split("/")
        if len(parts) == 2 and parts[0] == "texts":
            return int(parts[1])
    except (ValueError, AttributeError):
        pass
    return None


def _to_bbox(bbox_dict: dict, page_height: float) -> "BBox":
    """Docling BOTTOMLEFT bbox -> TOPLEFT 정규화.

    BOTTOMLEFT: t(위 모서리)가 b(아래 모서리)보다 큰 y값. origin은 페이지 하단.
    TOPLEFT:    origin이 페이지 상단이라 y가 아래로 갈수록 커짐 -> t < b가
                항상 성립해야 함(BBox 자체의 불변식).
    변환: new_y = page_height - old_y. 위 모서리(old t, 큰 값)가 origin에서
    멀리 있었으므로 빼면 작은 값(=TOPLEFT에서 위쪽)이 된다.
    """
    from .base import BBox

    l = bbox_dict.get("l", 0.0)
    old_t = bbox_dict.get("t", 0.0)
    r = bbox_dict.get("r", 0.0)
    old_b = bbox_dict.get("b", 0.0)
    return BBox(l=l, t=page_height - old_t, r=r, b=page_height - old_b)


class DoclingParser:
    """
    ParsedDoc을 만드는 기본 파서.

    make_converter(use_local_models=True) 는 쓰지 않는다 — 로컬 모델 경로가
    일부 PDF 7페이지에서 std::bad_alloc 으로 표를 통째로 날린다
    (measure_recall_single.py 주석 참고). 로컬 모델만 빼면 make_converter()
    와 동일한 설정이며, do_ocr 도 같은 값을 쓴다.

    do_ocr 는 기본이 None = Docling 기본값 위임. 라이브러리가 대신 정하지
    않는다(make_converter 참고).
    """

    def __init__(self, do_ocr: bool | None = None):
        self.do_ocr = do_ocr

    def parse(self, path: str) -> "ParsedDoc":
        doc = make_converter(do_ocr=self.do_ocr).convert(path).document
        return self.from_doc(doc)

    def from_doc(self, doc) -> "ParsedDoc":
        """이미 파싱된 DoclingDocument -> ParsedDoc. parse()의 변환부만 따로 뗀 것 —
        테스트가 매번 새로 PDF를 파싱하지 않고 이미 파싱된 doc(예: 세션 스코프
        픽스처)을 재사용할 수 있게 하기 위함(같은 프로세스 안에서 Docling
        파싱을 반복하면 메모리 누적으로 std::bad_alloc이 나는 이슈 회피)."""
        from .base import BlockLabel, ParsedDoc, TableBlock, TextBlock

        page_sizes: dict[int, tuple[float, float]] = {}
        if hasattr(doc, "pages"):
            for pg_key, pg_val in doc.pages.items():
                d = pg_val.model_dump() if hasattr(pg_val, "model_dump") else {}
                size = d.get("size") or {}
                page_sizes[int(pg_key)] = (size.get("width", 0.0), size.get("height", 0.0))

        def _page_height(page_no: int) -> float:
            return page_sizes.get(page_no, (0.0, 0.0))[1]

        _label_map = {
            "text": BlockLabel.TEXT,
            "list_item": BlockLabel.LIST_ITEM,
            "paragraph": BlockLabel.PARAGRAPH,
            "section_header": BlockLabel.SECTION_HEADER,
            "caption": BlockLabel.CAPTION,
        }

        texts: list[TextBlock] = []
        for i, item in enumerate(doc.texts):
            d = item.model_dump()
            prov = d.get("prov", [])
            page_no = prov[0].get("page_no", -1) if prov else -1
            bbox_dict = prov[0].get("bbox", {}) if prov else {}
            texts.append(TextBlock(
                index=i,
                text=d.get("text", "") or "",
                label=_label_map.get(d.get("label"), BlockLabel.OTHER),
                page_no=page_no,
                bbox=_to_bbox(bbox_dict, _page_height(page_no)),
            ))

        picture_caption_refs: set[int] = set()
        for pic in getattr(doc, "pictures", []) or []:
            pic_dict = pic.model_dump() if hasattr(pic, "model_dump") else {}
            for cap in pic_dict.get("captions", []) or []:
                cref = cap.get("cref", "") if isinstance(cap, dict) else getattr(cap, "cref", "")
                idx = _text_index_from_cref(cref)
                if idx is not None:
                    picture_caption_refs.add(idx)

        tables: list[TableBlock] = []
        for i, table in enumerate(doc.tables):
            d = table.model_dump()
            prov = d.get("prov", [])
            page_no = prov[0].get("page_no", -1) if prov else -1
            bbox_dict = prov[0].get("bbox", {}) if prov else {}
            data = d.get("data", {})

            caption_refs = []
            for cap in d.get("captions", []) or []:
                cref = cap.get("cref", "") if isinstance(cap, dict) else getattr(cap, "cref", "")
                idx = _text_index_from_cref(cref)
                if idx is not None:
                    caption_refs.append(idx)

            footnote_refs = []
            for fn in d.get("footnotes", []) or []:
                cref = fn.get("cref", "") if isinstance(fn, dict) else getattr(fn, "cref", "")
                idx = _text_index_from_cref(cref)
                if idx is not None:
                    footnote_refs.append(idx)

            try:
                html = table.export_to_html(doc) or None
            except Exception:
                html = None

            tables.append(TableBlock(
                index=i,
                page_no=page_no,
                bbox=_to_bbox(bbox_dict, _page_height(page_no)),
                cells=data.get("table_cells", []),
                num_rows=data.get("num_rows", 0),
                num_cols=data.get("num_cols", 0),
                html=html,
                caption_refs=caption_refs,
                footnote_refs=footnote_refs,
            ))

        return ParsedDoc(
            texts=texts,
            tables=tables,
            picture_caption_refs=picture_caption_refs,
            page_sizes=page_sizes,
        )
