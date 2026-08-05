"""
Docling 모델을 심볼릭링크 없이 로컬 디렉토리에 직접 다운로드.
Windows에서 Developer Mode 없이도 동작.

Usage:
    python download_models.py
"""
import os
import sys

if hasattr(sys.stdout, 'buffer'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

REPOS = [
    "docling-project/docling-models",
    "docling-project/docling-layout-heron",
]


def download_repo(repo_id: str) -> str:
    from huggingface_hub import snapshot_download
    name = repo_id.split("/")[-1]
    local_dir = os.path.join(MODELS_DIR, name)
    print(f"\n[DOWN] {repo_id}")
    print(f"       -> {local_dir}")
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,      # download directly, no symlinks
        local_dir_use_symlinks=False,  # force copy, not symlink
    )
    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(local_dir)
        for f in fns
    ) / 1024 / 1024
    print(f"       done: {size_mb:.1f} MB")
    return path


if __name__ == "__main__":
    print("=== Docling Model Download (no symlinks) ===\n")
    for repo in REPOS:
        try:
            download_repo(repo)
        except Exception as e:
            print(f"  ERROR: {e}")
    print(f"\nModels saved to: {MODELS_DIR}")
    print("Set DOCLING_ARTIFACTS_PATH or use artifacts_path= in scripts.")
