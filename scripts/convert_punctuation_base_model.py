#!/usr/bin/env python3
"""Converts oliverguhr/fullstop-punctuation-multilingual-base (PyTorch) to a
quantized ONNX model and drops it where punctuation_restore.rs's load()
looks for it first: ~/.cache/squishi/models/fullstop-punctuation-multilingual-base/.

Why this exists: unlike the large XLM-RoBERTa model currently in
production (ldenoue/fullstop-punctuation-multilang-large, an existing ONNX
mirror on the Hub), no one has published an ONNX export of the base
variant. Investigated 2026-08-08: base measured 1.4x faster restore on a
real 6,662-word transcript with comparable quality (no accuracy
regression, unlike a DistilBERT alternative that was tried and rejected
for producing grammatically broken output) -- see
src/punctuation_restore.rs's module doc for the full comparison. Worth
doing for real corpora: ~500 videos at similar length is hours of restore
time either way, so a validated 1.4x+ (more once ONNX-quantized vs the
PyTorch number that was benchmarked) is real.

Usage:
    python3 -m venv .venv && source .venv/bin/activate
    pip install torch optimum-onnx onnx onnxruntime
    python3 scripts/convert_punctuation_base_model.py

Not run automatically by anything -- punctuation_restore.rs falls back to
the large model via hf_hub when this local path doesn't exist, so running
this script is an optional, one-time speedup, not a hard dependency.
"""

import shutil
import subprocess
import sys
from pathlib import Path

MODEL_ID = "oliverguhr/fullstop-punctuation-multilingual-base"
DEST = Path.home() / ".cache" / "squishi" / "models" / "fullstop-punctuation-multilingual-base"


def main() -> None:
    export_dir = Path("/tmp/squishi-punctuation-base-export")
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {MODEL_ID} to ONNX (fp32) ...")
    subprocess.run(
        [
            "optimum-cli", "export", "onnx",
            "--model", MODEL_ID,
            "--task", "token-classification",
            str(export_dir),
        ],
        check=True,
    )

    print("Quantizing to int8 ...")
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantized = export_dir / "model_quantized.onnx"
    quantize_dynamic(
        model_input=str(export_dir / "model.onnx"),
        model_output=str(quantized),
        weight_type=QuantType.QInt8,
    )

    DEST.mkdir(parents=True, exist_ok=True)
    for name in ("model_quantized.onnx", "tokenizer.json", "config.json"):
        shutil.copy(export_dir / name, DEST / name)
        print(f"  -> {DEST / name} ({(DEST / name).stat().st_size} bytes)")

    print(f"\nDone. punctuation_restore.rs will pick this up automatically from {DEST}")


if __name__ == "__main__":
    sys.exit(main())
