"""
Test PDF downloader - tables and captions heavy papers.
"""
import urllib.request
import os
import sys

# stdout UTF-8 for Windows console
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PDF_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "pdfs")
os.makedirs(PDF_DIR, exist_ok=True)

PDFS = [
    {
        "name": "attention_is_all_you_need.pdf",
        "url": "https://arxiv.org/pdf/1706.03762",
        "desc": "Transformer paper - many tables",
    },
    {
        "name": "bert_paper.pdf",
        "url": "https://arxiv.org/pdf/1810.04805",
        "desc": "BERT paper - rich comparison tables",
    },
    {
        "name": "gpt3_few_shot_learners.pdf",
        "url": "https://arxiv.org/pdf/2005.14165",
        "desc": "GPT-3 paper - huge appendix tables, likely multi-page/cross-page caption cases",
    },
    {
        "name": "bok_potential_growth_report_kr.pdf",
        "url": "https://hamancci.korcham.net/file/dext5uploaddata/2024/%ED%95%9C%EA%B5%AD%EC%9D%80%ED%96%89%20%E2%80%98%EC%9A%B0%EB%A6%AC%20%EA%B2%BD%EC%A0%9C%EC%9D%98%20%EC%9E%A0%EC%9E%AC%EC%84%B1%EC%9E%A5%EB%A5%A0%EA%B3%BC%20%ED%96%A5%ED%9B%84%20%EC%A0%84%EB%A7%9D%E2%80%99%20%EB%B3%B4%EA%B3%A0%EC%84%9C.pdf",
        "desc": "한국은행 잠재성장률 보고서 - 한국어 네이티브 텍스트, 표 다수",
    },
]


def download(name: str, url: str, desc: str) -> str:
    path = os.path.join(PDF_DIR, name)
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"[SKIP] {name} (already exists, {size:,} bytes) - {desc}")
        return path

    print(f"[DOWN] {name} - {desc}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(path, "wb") as f:
            f.write(resp.read())
        size = os.path.getsize(path)
        print(f"       done: {size:,} bytes -> {path}")
        return path
    except Exception as e:
        print(f"       failed: {e}")
        return ""


if __name__ == "__main__":
    print("=== Test PDF Download ===\n")
    success = []
    for item in PDFS:
        p = download(**item)
        if p:
            success.append(p)
    print(f"\n{len(success)}/{len(PDFS)} PDFs ready")
    for p in success:
        print(f"  - {p}")
