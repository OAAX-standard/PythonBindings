#!/usr/bin/env python
"""Download this directory's model.onnx.

UltraFace (version-RFB-320, WIDERFace-trained, MIT licensed), pinned to a
specific commit of its source repository so this always fetches the same
bytes (verified by its checksum):
https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB

    python download_model.py           # fetch if missing
    python download_model.py --force   # re-download even if already present
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # examples/, for download_model_utils
from download_model_utils import fetch  # noqa: E402

HERE = Path(__file__).parent

# Pinned to a specific commit, not "master", so this keeps working even after
# the upstream repository changes what "master" points at.
COMMIT = "442599cebf11307bc231e2e5c3c6c869369fee48"
MODEL_URL = f"https://raw.githubusercontent.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/{COMMIT}/models/onnx/version-RFB-320_simplified.onnx"
MODEL_SHA256 = "faf6740b495e8b9508e6fda11e8c0368bbdde01143e8f0a3e511d0892c4d794e"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download even if already present")
    args = parser.parse_args()
    fetch(MODEL_URL, HERE / "model.onnx", sha256=MODEL_SHA256, force=args.force)


if __name__ == "__main__":
    main()
