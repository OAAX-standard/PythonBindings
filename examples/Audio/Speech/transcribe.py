"""Transcribe a short speech clip with Whisper using any OAAX-compliant runtime.

    python transcribe.py <runtime_library> <model_directory> [--audio path/to.wav]

Defaults to the bundled speech.wav. Unlike the other examples this is not a
single inference: Whisper is two models. The encoder runs once and turns the
whole clip into a sequence of audio vectors; the decoder then runs once per
word piece, each time fed everything it has written so far plus those audio
vectors, until it emits its end-of-text token. <model_directory> holds both
models, the tokenizer and a model.json describing them
(see ../Models/Whisper-Tiny-English/model.json).

    pip install -e .[examples]   # from the repo root: the tokenizers package
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

SCRIPT_DIRECTORY = Path(__file__).parent
DEFAULT_AUDIO_PATH = SCRIPT_DIRECTORY / "speech.wav"

sys.path.insert(0, str(SCRIPT_DIRECTORY.parent))  # examples/Audio/, for audio_utils

from audio_utils import load_model_metadata, log_mel_spectrogram, read_wav  # noqa: E402

from oaax import Runtime  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("library", help="path to an OAAX runtime shared library")
    parser.add_argument("model", type=Path, help="directory holding encoder.onnx, decoder.onnx, tokenizer.json and model.json")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO_PATH, help=f"mono 16-bit 16 kHz WAV to transcribe (default: {DEFAULT_AUDIO_PATH.name})")
    parser.add_argument("--max-tokens", type=int, default=128, help="give up after this many word pieces if the model never signals the end (default: 128)")
    parser.add_argument("--config", action="append", default=[], metavar="KEY=VALUE", help="runtime init option, repeatable")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_init_options = dict(option.split("=", 1) for option in args.config)  # "key=value" strings -> {key: value}

    # Everything we need to know about the model - its two files, its tokenizer,
    # the tokens that start and end a transcript, how it wants audio prepared -
    # comes from the model.json in the model directory.
    metadata = load_model_metadata(args.model)
    tokenizer = Tokenizer.from_file(str(metadata.tokenizer_path))
    start_tokens = [tokenizer.token_to_id(token) for token in metadata.start_tokens]
    end_token = tokenizer.token_to_id(metadata.end_token)

    # Load the clip and turn it into the log-mel spectrogram the encoder expects.
    samples = read_wav(args.audio, metadata.audio.sample_rate)
    print(f"Audio: {args.audio.name}, {len(samples) / metadata.audio.sample_rate:.1f} s")
    if len(samples) > metadata.audio.chunk_samples:
        print(f"  longer than the model's {metadata.audio.chunk_seconds} s window; only the first {metadata.audio.chunk_seconds} s will be transcribed")
    features = log_mel_spectrogram(samples, metadata.audio)

    # Connect to the runtime. The "with" block takes care of shutting it down
    # cleanly afterwards, even if something below raises an error.
    with Runtime(args.library) as runtime:
        print(f"Runtime: {runtime.name} v{runtime.version}")
        runtime.init(runtime_init_options)  # apply any vendor-specific options requested via --config

        # --- Load both models in one call ---
        encoder_id, decoder_id = runtime.load_models([metadata.encoder_path, metadata.decoder_path])
        print(f"Model: {metadata.name} - encoder {metadata.encoder_path.name}, decoder {metadata.decoder_path.name}")

        # --- Encoder: once for the whole clip ---
        encoder_result = runtime.run(encoder_id, {"input_features": features[np.newaxis]}, timeout_ms=60_000)
        audio_vectors = encoder_result["last_hidden_state"].data  # [1, positions, features]: what the decoder listens to

        # --- Decoder: once per word piece ---
        # Each step sees the full token sequence so far and predicts the next one from
        # the scores at the last position. (A cached variant would only feed the newest
        # token; this way needs no cache handling and is fast enough for a sentence.)
        tokens = list(start_tokens)
        for decoder_runs in range(1, args.max_tokens + 1):  # noqa: B007 - its final value is reported after the loop
            decoder_result = runtime.run(
                decoder_id,
                {"input_ids": np.asarray([tokens], dtype=np.int64), "encoder_hidden_states": audio_vectors},
                timeout_ms=60_000,
            )
            next_token = int(decoder_result["logits"].data[0, -1].argmax())
            if next_token == end_token:
                break
            tokens.append(next_token)
        else:
            print(f"  stopped after {args.max_tokens} tokens without seeing {metadata.end_token}")

        # --- Parse the result ---
        # The tokenizer turns the word-piece ids back into text; the start tokens are dropped.
        transcript = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        print(f"\nTranscript ({len(tokens) - len(start_tokens)} tokens, {decoder_runs} decoder runs):\n  {transcript}")


if __name__ == "__main__":
    main()
