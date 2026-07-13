# explore_docling.py
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("https://arxiv.org/pdf/2408.09869")
doc = result.document

print("=== 문서 내 표 개수 ===")
print(len(doc.tables))

print("\n=== 첫 번째 표 구조 ===")
table = doc.tables[0]

# 캡션 텍스트 실제로 뽑기
for cap_ref in table.captions:
    cap_item = doc.texts[int(cap_ref.cref.split("/")[-1])]
    print("캡션 텍스트:", cap_item.text)

print("bbox:", table.prov[0].bbox)
print("HTML:\n", table.export_to_html(doc=doc)[:500])