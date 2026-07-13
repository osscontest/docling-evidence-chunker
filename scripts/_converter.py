"""
Shared DocumentConverter factory — uses local model directory to avoid
Windows symlink permission issues with HuggingFace Hub cache.
"""
import os
from pathlib import Path


MODELS_DIR = Path(__file__).parent.parent / "models"


def make_converter():
    """Return a DocumentConverter configured to use local model artifacts."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    # combined/ has model.safetensors (layout) + model_artifacts/ (table) merged
    artifacts_path = MODELS_DIR / "combined"
    if not artifacts_path.exists():
        # Fallback: let docling use its default HF cache
        artifacts_path = None

    pipeline_options = PdfPipelineOptions(
        do_ocr=False,                   # PDF has native text — skip OCR
        do_table_structure=True,        # keep table cell detection
        artifacts_path=artifacts_path,  # use local models dir
    )

    return DocumentConverter(
        format_options={
            "pdf": PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
