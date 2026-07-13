"""
W1 전체 탐색 파이프라인 실행기.
Usage:
    python run_all.py [path/to/file.pdf]

Runs steps 01~04 sequentially on the given PDF.
"""
import sys
import os
import subprocess
import time

if hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "scripts")
PDF_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "data", "pdfs", "attention_is_all_you_need.pdf"
)

PDF_PREFIX = os.path.splitext(os.path.basename(PDF_PATH))[0]

STEPS = [
    ("01_basic_parse.py",    [PDF_PATH]),
    ("02_explore_tables.py", [PDF_PATH]),
    ("03_explore_texts.py",  [PDF_PATH]),
    ("04_summary_report.py", [PDF_PREFIX]),
]


def run_step(script: str, args: list) -> bool:
    script_path = os.path.join(SCRIPTS_DIR, script)
    cmd = [sys.executable, script_path] + args
    print(f"\n{'#'*62}")
    print(f"# Running: {script}")
    print(f"{'#'*62}")
    start = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - start
    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"\n  [{status}] {script} ({elapsed:.1f}s)")
    return result.returncode == 0


if __name__ == "__main__":
    print(f"{'#'*62}")
    print(f"# W1 Docling API 탐색 파이프라인")
    print(f"# PDF: {PDF_PATH}")
    print(f"{'#'*62}")

    results = {}
    for script, args in STEPS:
        ok = run_step(script, args)
        results[script] = ok

    print(f"\n{'='*62}")
    print("  Pipeline Summary")
    print(f"{'='*62}")
    for script, ok in results.items():
        mark = "V" if ok else "X"
        print(f"  [{mark}] {script}")

    all_ok = all(results.values())
    print(f"\n  Report -> reports/W1_docling_api_report_{PDF_PREFIX}.txt")
    sys.exit(0 if all_ok else 1)
