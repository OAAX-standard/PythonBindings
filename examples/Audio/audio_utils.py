"""Shared helpers for the OAAX audio examples.

Reading a model's ``model.json`` metadata, loading a WAV file, and turning
its samples into the log-mel spectrogram that speech models take as input.
Import this from a script under examples/Audio/<name>/ rather than
duplicating it.
"""

from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

METADATA_FILENAME = "model.json"


@dataclass(frozen=True)
class AudioSettings:
    """How the model wants its audio prepared; every field is read from model.json."""

    sample_rate: int
    n_fft: int
    hop_length: int
    chunk_seconds: int
    mel_filters: np.ndarray  # [frequency_bins, mel_bins]: the model's own filterbank, as data

    @property
    def chunk_samples(self) -> int:
        return self.sample_rate * self.chunk_seconds


@dataclass(frozen=True)
class ModelMetadata:
    name: str
    encoder_path: Path
    decoder_path: Path
    tokenizer_path: Path
    start_tokens: list[str]
    end_token: str
    audio: AudioSettings


def load_model_metadata(model_directory: Path) -> ModelMetadata:
    """Read the model.json in ``model_directory``; every file it names is relative to that directory."""
    metadata_path = model_directory / METADATA_FILENAME
    if not metadata_path.is_file():
        raise SystemExit(f"no {METADATA_FILENAME} in {model_directory}")
    raw = json.loads(metadata_path.read_text())
    audio = raw["audio"]
    return ModelMetadata(
        name=raw["name"],
        encoder_path=model_directory / raw["encoder"],
        decoder_path=model_directory / raw["decoder"],
        tokenizer_path=model_directory / raw["tokenizer"],
        start_tokens=raw["start_tokens"],
        end_token=raw["end_token"],
        audio=AudioSettings(
            sample_rate=audio["sample_rate"],
            n_fft=audio["n_fft"],
            hop_length=audio["hop_length"],
            chunk_seconds=audio["chunk_seconds"],
            mel_filters=np.load(model_directory / audio["mel_filters"]),
        ),
    )


def read_wav(path: Path, sample_rate: int) -> np.ndarray:
    """Mono 16-bit PCM WAV at ``sample_rate`` -> float32 samples in [-1, 1]. Anything else is refused rather than guessed at."""
    with wave.open(str(path)) as wav:
        channels, sample_width, rate = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
        if (channels, sample_width, rate) != (1, 2, sample_rate):
            raise SystemExit(f"{path.name}: need mono 16-bit PCM at {sample_rate} Hz, got {channels} channel(s), {8 * sample_width}-bit, {rate} Hz")
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def log_mel_spectrogram(samples: np.ndarray, audio: AudioSettings) -> np.ndarray:
    """Whisper-style input features, ``[mel_bins, frames]``, for exactly one chunk of audio.

    The clip is cut or padded with silence to ``chunk_seconds``, split into
    overlapping windows (one every ``hop_length`` samples), and each window's
    spectrum is mapped onto the mel scale with the model's filterbank and
    compressed with a log. The final scaling matches what the model was trained on.
    """
    samples = np.pad(samples[: audio.chunk_samples], (0, max(0, audio.chunk_samples - len(samples))))
    padded = np.pad(samples, audio.n_fft // 2, mode="reflect")  # so the first window is centred on sample 0
    window = np.hanning(audio.n_fft + 1)[:-1]  # periodic Hann window
    frames = np.lib.stride_tricks.sliding_window_view(padded, audio.n_fft)[:: audio.hop_length][:-1]
    magnitudes = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2  # [frames, frequency_bins]
    mel = magnitudes @ audio.mel_filters  # [frames, mel_bins]
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)  # clamp the dynamic range to 80 dB
    return ((log_spec + 4.0) / 4.0).T.astype(np.float32)
