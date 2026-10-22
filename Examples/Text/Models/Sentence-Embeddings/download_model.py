#!/usr/bin/env python
"""Download this directory's model.onnx and tokenizer.json.

all-MiniLM-L6-v2 (sentence-transformers), 8-bit quantised ONNX export, from
Hugging Face's Xenova org: https://huggingface.co/Xenova/all-MiniLM-L6-v2

    python download_model.py           # fetch anything missing
    python download_model.py --force   # re-download even if already present
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # Examples/, for download_model_utils
from download_model_utils import fetch  # noqa: E402

HERE = Path(__file__).parent
REPO = "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main"

# (file in the repo, local filename, expected sha256)
FILES = [
    ("onnx/model_quantized.onnx", "model.onnx", "afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1"),
    ("tokenizer.json", "tokenizer.json", None),  # no fixed checksum: this file is expected to track the repo's current revision
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download even if already present")
    args = parser.parse_args()
    for remote_name, local_name, sha256 in FILES:
        fetch(f"{REPO}/{remote_name}", HERE / local_name, sha256=sha256, force=args.force)


if __name__ == "__main__":
    main()
