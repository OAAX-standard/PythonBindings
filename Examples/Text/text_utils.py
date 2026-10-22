"""Shared helpers for the OAAX text examples.

Covers the parts every such example needs: reading a model's ``model.json``
metadata (its tensors, and which part of a tokenizer encoding fills each
input), turning a string into those input tensors with the model's own
tokenizer, and the small bits of arithmetic that turn outputs into answers
(softmax, mean pooling, cosine similarity). Import this from a script under
Examples/Text/<name>/ rather than duplicating it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tokenizers import Encoding, Tokenizer

from oaax import Tensor, TensorSpec
from oaax.enums import TensorElementType

METADATA_FILENAME = "model.json"
INPUT_ROLES = ("token_ids", "attention_mask", "token_type_ids")


@dataclass(frozen=True)
class ModelInput:
    """One model input, as described in model.json; ``role`` names the part of a tokenizer encoding that fills it."""

    spec: TensorSpec
    role: str


@dataclass(frozen=True)
class ModelOutput:
    """One model output; ``labels`` names the classes along a classifier's score axis."""

    spec: TensorSpec
    labels: list[str] | None = None


@dataclass(frozen=True)
class ModelMetadata:
    name: str
    tokenizer_path: Path
    inputs: list[ModelInput]
    outputs: list[ModelOutput]

    @property
    def sequence_length(self) -> int:
        """How many tokens every input holds; shorter texts are padded to this, longer ones truncated."""
        return self.inputs[0].spec.shape[1]


def _parse_tensor_spec(entry: dict) -> TensorSpec:
    return TensorSpec(entry["name"], TensorElementType[entry["element_type"]], tuple(entry["shape"]))


def _parse_model_input(entry: dict) -> ModelInput:
    spec = _parse_tensor_spec(entry)
    role = entry.get("role")
    if role not in INPUT_ROLES:
        raise SystemExit(f"input {spec.name!r}: 'role' must be one of {INPUT_ROLES}, got {role!r}")
    if len(spec.shape) != 2 or spec.shape[0] != 1:
        raise SystemExit(f"input {spec.name!r}: expected a (1, sequence_length) shape, got {spec.shape}")
    return ModelInput(spec, role)


def _parse_model_output(entry: dict) -> ModelOutput:
    spec = _parse_tensor_spec(entry)
    labels = entry.get("labels")
    if labels is not None and len(labels) != spec.shape[-1]:
        raise SystemExit(f"output {spec.name!r}: {len(labels)} labels for {spec.shape[-1]} classes")
    return ModelOutput(spec, labels)


def load_model_metadata(model_path: Path) -> ModelMetadata:
    """Read the model.json that sits next to ``model_path`` (one file describes every variant of a model)."""
    metadata_path = model_path.parent / METADATA_FILENAME
    if not metadata_path.is_file():
        raise SystemExit(f"no {METADATA_FILENAME} next to {model_path.name} (looked in {model_path.parent})")
    raw = json.loads(metadata_path.read_text())

    inputs = [_parse_model_input(entry) for entry in raw["inputs"]]
    if len({model_input.spec.shape[1] for model_input in inputs}) != 1:
        raise SystemExit(f"{metadata_path}: all inputs must share one sequence length")
    tokenizer_path = metadata_path.parent / raw["tokenizer"]
    if not tokenizer_path.is_file():
        raise SystemExit(f"{metadata_path}: tokenizer file not found: {tokenizer_path}")
    return ModelMetadata(raw["name"], tokenizer_path, inputs, [_parse_model_output(entry) for entry in raw["outputs"]])


def load_tokenizer(metadata: ModelMetadata) -> Tokenizer:
    """The model's own tokenizer, set up to always produce exactly ``sequence_length`` tokens."""
    tokenizer = Tokenizer.from_file(str(metadata.tokenizer_path))
    tokenizer.enable_padding(length=metadata.sequence_length)
    tokenizer.enable_truncation(metadata.sequence_length)
    return tokenizer


def build_model_inputs(metadata: ModelMetadata, tokenizer: Tokenizer, text: str) -> tuple[list[Tensor], Encoding]:
    """Tokenize ``text`` and give each model input the part of the encoding its role asks for.

    Also returns the encoding itself, for scripts that need to know which
    positions hold real tokens rather than padding.
    """
    encoding = tokenizer.encode(text)
    encoding_by_role = {
        "token_ids": encoding.ids,
        "attention_mask": encoding.attention_mask,
        "token_type_ids": encoding.type_ids,
    }
    model_inputs = [
        Tensor(model_input.spec.name, np.asarray(encoding_by_role[model_input.role], dtype=model_input.spec.dtype or np.int64)[np.newaxis, :])
        for model_input in metadata.inputs
    ]
    return model_inputs, encoding


def softmax(scores: np.ndarray) -> np.ndarray:
    """Turn a vector of raw class scores (logits) into probabilities that sum to 1."""
    shifted = np.exp(scores - scores.max())
    return shifted / shifted.sum()


def mean_pool(token_vectors: np.ndarray, attention_mask: list[int]) -> np.ndarray:
    """One vector for the whole text: the average of its ``[tokens, features]`` rows, ignoring padding."""
    weights = np.asarray(attention_mask, dtype=np.float32)[:, np.newaxis]
    return (token_vectors * weights).sum(axis=0) / weights.sum()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1.0 for vectors pointing the same way, 0.0 for unrelated ones."""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
