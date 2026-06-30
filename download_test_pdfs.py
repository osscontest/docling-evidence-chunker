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

PDF_DIR = os.path.join(os.path.dirname(__file__), "data", "pdfs")
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
