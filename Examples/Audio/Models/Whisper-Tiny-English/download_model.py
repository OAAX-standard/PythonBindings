#!/usr/bin/env python
"""Download this directory's model and filterbank files.

encoder.onnx, decoder.onnx and tokenizer.json are Whisper tiny.en (OpenAI,
English-only), 8-bit quantised ONNX exports from Hugging Face's Xenova org:
https://huggingface.co/Xenova/whisper-tiny.en

mel_filters.npy is OpenAI's own published 80-channel mel filterbank,
transposed to [frequency_bins, mel_bins] to match this project's
``magnitudes @ mel_filters`` convention (see audio_utils.log_mel_spectrogram):
https://github.com/openai/whisper/blob/main/whisper/assets/mel_filters.npz

    python download_model.py           # fetch anything missing
    python download_model.py --force   # re-download even if already present
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # Examples/, for model_fetch
from model_fetch import fetch  # noqa: E402

HERE = Path(__file__).parent
HF_REPO = "https://huggingface.co/Xenova/whisper-tiny.en/resolve/main"
MEL_FILTERS_URL = "https://github.com/openai/whisper/raw/main/whisper/assets/mel_filters.npz"

# (file in the HF repo, local filename, expected sha256)
ONNX_FILES = [
    ("onnx/encoder_model_quantized.onnx", "encoder.onnx", "8cc3c6f8563d1b3fbd2c5af9f64c2bed8b020bc593c402d1ef53b9f08fbf1b90"),
    ("onnx/decoder_model_quantized.onnx", "decoder.onnx", "62b1761a6c23961cdabbf7caa25495525d78528a64baacf7e4283c83cb2941f1"),
    ("tokenizer.json", "tokenizer.json", None),  # no fixed checksum: this file is expected to track the repo's current revision
]


def fetch_mel_filters(force: bool) -> None:
    """mel_filters.npy isn't hosted anywhere directly; derive it from OpenAI's own asset."""
    destination = HERE / "mel_filters.npy"
    if destination.exists() and not force:
        print(f"Already present: {destination}")
        return
    print(f"Fetching {destination.name} (derived from {MEL_FILTERS_URL})")
    with urllib.request.urlopen(MEL_FILTERS_URL, timeout=60) as response:
        raw = response.read()
    with np.load(io.BytesIO(raw)) as data:
        mel_80 = data["mel_80"]  # OpenAI's filterbank, as [mel_bins, frequency_bins]
    np.save(destination, mel_80.T.astype(np.float32))  # -> [frequency_bins, mel_bins]
    print(f"  wrote {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download everything, even if already present")
    args = parser.parse_args()
    for remote_name, local_name, sha256 in ONNX_FILES:
        fetch(f"{HF_REPO}/{remote_name}", HERE / local_name, sha256=sha256, force=args.force)
    fetch_mel_filters(args.force)


if __name__ == "__main__":
    main()
