#!/usr/bin/env python
"""Download this directory's model.onnx and characters.txt.

model.onnx is PaddleOCR's PP-OCRv4 text recogniser (Chinese + English),
exactly as bundled inside the ``rapidocr_onnxruntime`` PyPI package (pinned
to a specific release below so this always fetches the same bytes, verified
by its checksum):
https://pypi.org/project/rapidocr-onnxruntime/

characters.txt is PaddleOCR's standard character dictionary, ppocr_keys_v1.txt:
https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/utils/ppocr_keys_v1.txt

    python download_model.py           # fetch anything missing
    python download_model.py --force   # re-download even if already present
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # examples/, for download_model_utils
from download_model_utils import fetch, fetch_zip_member  # noqa: E402

HERE = Path(__file__).parent

# A specific rapidocr_onnxruntime wheel release (just a .zip under the hood),
# pinned so this keeps working even after newer releases change what's bundled.
RAPIDOCR_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/ba/12/1e5497183bdbe782dbb91bad1d0d2297dba4d2831b2652657f7517bfc6df/rapidocr_onnxruntime-1.4.4-py3-none-any.whl"
)
RAPIDOCR_MEMBER = "rapidocr_onnxruntime/models/ch_PP-OCRv4_rec_infer.onnx"
MODEL_SHA256 = "48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b"

CHARACTERS_URL = "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/ppocr_keys_v1.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download even if already present")
    args = parser.parse_args()
    fetch_zip_member(RAPIDOCR_WHEEL_URL, RAPIDOCR_MEMBER, HERE / "model.onnx", sha256=MODEL_SHA256, force=args.force)
    fetch(CHARACTERS_URL, HERE / "characters.txt", force=args.force)  # content is stable, but no published checksum to pin


if __name__ == "__main__":
    main()
