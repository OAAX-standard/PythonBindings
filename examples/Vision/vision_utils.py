"""Shared helpers for the OAAX vision examples.

Covers the parts every such example needs: reading a model's ``model.json``
metadata (its tensors and how to preprocess for it), preparing an image and
any other inputs accordingly, and decoding the two output conventions the
examples use - the "bboxes-format:xyxysc;..." detection tensors of the OAAX
illustrative examples, and CTC character scores from a text recogniser.
Import this from a script under examples/Vision/<name>/ rather than
duplicating it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from oaax import Tensor, TensorSpec
from oaax.enums import TensorElementType

METADATA_FILENAME = "model.json"
BOX_FORMAT_PREFIX = "bboxes-format:xyxysc;"
CHANNEL_ORDERS = ("RGB", "BGR")
RESIZE_MODES = ("stretch", "keep-aspect")

# (label, confidence score, (x1, y1, x2, y2))
Detection = tuple[str, float, tuple[float, float, float, float]]


@dataclass(frozen=True)
class ModelInput:
    """One model input, as described in model.json.

    The image input carries per-channel ``mean`` and ``normalization``; the
    image is fed to the model as ``(pixel - mean) / normalization`` with
    pixels in 0-255, after reordering channels to ``channel_order`` (RGB by
    default). ``resize`` is either "stretch" (resize to exactly the input's
    H x W) or "keep-aspect" (match the height, keep proportions, pad the
    right edge with the mean colour). Every other input carries a fixed
    ``value`` that is broadcast to the input's shape.
    """

    spec: TensorSpec
    mean: tuple[float, ...] | None = None
    normalization: tuple[float, ...] | None = None
    channel_order: str = "RGB"
    resize: str = "stretch"
    value: float | None = None

    @property
    def is_image(self) -> bool:
        return self.mean is not None


@dataclass(frozen=True)
class ModelOutput:
    """One model output; ``characters`` is set for a CTC text output (see :func:`ctc_decode`)."""

    spec: TensorSpec
    characters: list[str] | None = None


@dataclass(frozen=True)
class ModelMetadata:
    name: str
    inputs: list[ModelInput]
    outputs: list[ModelOutput]

    @property
    def image_input(self) -> ModelInput:
        return next(model_input for model_input in self.inputs if model_input.is_image)


def _parse_tensor_spec(entry: dict) -> TensorSpec:
    return TensorSpec(entry["name"], TensorElementType[entry["element_type"]], tuple(entry["shape"]))


def _parse_model_input(entry: dict) -> ModelInput:
    spec = _parse_tensor_spec(entry)
    has_mean, has_normalization = "mean" in entry, "normalization" in entry
    if has_mean != has_normalization:
        raise SystemExit(f"input {spec.name!r}: 'mean' and 'normalization' must be given together")
    if has_mean:
        if len(spec.shape) != 4 or len(entry["mean"]) != spec.shape[1] or len(entry["normalization"]) != spec.shape[1]:
            raise SystemExit(f"input {spec.name!r}: an image input needs an (N, C, H, W) shape and one mean/normalization per channel")
        channel_order = entry.get("channel_order", "RGB")
        if channel_order not in CHANNEL_ORDERS:
            raise SystemExit(f"input {spec.name!r}: 'channel_order' must be one of {CHANNEL_ORDERS}, got {channel_order!r}")
        resize = entry.get("resize", "stretch")
        if resize not in RESIZE_MODES:
            raise SystemExit(f"input {spec.name!r}: 'resize' must be one of {RESIZE_MODES}, got {resize!r}")
        return ModelInput(spec, mean=tuple(entry["mean"]), normalization=tuple(entry["normalization"]), channel_order=channel_order, resize=resize)
    if "value" not in entry:
        raise SystemExit(f"input {spec.name!r}: a non-image input needs a fixed 'value'")
    return ModelInput(spec, value=entry["value"])


def _parse_model_output(entry: dict, metadata_directory: Path) -> ModelOutput:
    spec = _parse_tensor_spec(entry)
    if "characters" not in entry:
        return ModelOutput(spec)
    characters = (metadata_directory / entry["characters"]).read_text(encoding="utf-8").splitlines()
    if len(characters) + 2 != spec.shape[-1]:
        raise SystemExit(
            f"output {spec.name!r}: {entry['characters']} lists {len(characters)} characters, but the output has "
            f"{spec.shape[-1]} classes; expected the characters plus a CTC blank and a space"
        )
    return ModelOutput(spec, characters)


def load_model_metadata(model_path: Path) -> ModelMetadata:
    """Read the model.json that sits next to ``model_path`` (one file describes every variant of a model)."""
    metadata_path = model_path.parent / METADATA_FILENAME
    if not metadata_path.is_file():
        raise SystemExit(f"no {METADATA_FILENAME} next to {model_path.name} (looked in {model_path.parent})")
    raw = json.loads(metadata_path.read_text())

    inputs = [_parse_model_input(entry) for entry in raw["inputs"]]
    if sum(model_input.is_image for model_input in inputs) != 1:
        raise SystemExit(f"{metadata_path}: exactly one input must carry 'mean'/'normalization' (the image)")
    outputs = [_parse_model_output(entry, metadata_path.parent) for entry in raw["outputs"]]
    return ModelMetadata(raw["name"], inputs, outputs)


def load_and_preprocess_image(path: Path, image_input: ModelInput) -> tuple[np.ndarray, Image.Image]:
    """Resize ``path`` to the model's input size and apply its channel order and mean/normalization.

    Returns an NCHW array ready to feed the model, plus the resized PIL image
    (the frame of reference for anything the model reports in image coordinates).
    """
    _batch, _channels, height, width = image_input.spec.shape
    source = Image.open(path).convert("RGB")
    if image_input.resize == "stretch":
        resized_image = source.resize((width, height))  # PIL size is (width, height)
    else:
        # keep-aspect: scale to the input height, pad the right edge with the mean colour
        # so the padding normalises to zero; an image too wide to fit is squashed instead.
        scaled_width = min(width, math.ceil(height * source.width / source.height))
        fill = tuple(int(round(value)) for value in image_input.mean)
        if image_input.channel_order == "BGR":
            fill = fill[::-1]  # the mean is in the model's channel order; PIL wants RGB
        resized_image = Image.new("RGB", (width, height), fill)
        resized_image.paste(source.resize((scaled_width, height)), (0, 0))

    pixel_array = np.asarray(resized_image, dtype=np.float32)  # HWC, RGB, values 0-255
    if image_input.channel_order == "BGR":
        pixel_array = pixel_array[..., ::-1]
    mean = np.asarray(image_input.mean, dtype=np.float32)
    normalization = np.asarray(image_input.normalization, dtype=np.float32)
    pixel_array = (pixel_array - mean) / normalization  # per channel, broadcast over the trailing C axis
    nchw_array = np.transpose(pixel_array, (2, 0, 1))[np.newaxis, ...]  # HWC -> CHW, then add the batch axis
    return nchw_array.astype(image_input.spec.dtype or np.float32), resized_image


def build_model_inputs(metadata: ModelMetadata, image_path: Path) -> tuple[list[Tensor], Image.Image]:
    """One Tensor per model input: the preprocessed image, and each fixed value broadcast to its shape."""
    model_inputs: list[Tensor] = []
    for model_input in metadata.inputs:
        if model_input.is_image:
            array, resized_image = load_and_preprocess_image(image_path, model_input)
        else:
            array = np.full(model_input.spec.shape, model_input.value, dtype=model_input.spec.dtype or np.float32)
        model_inputs.append(Tensor(model_input.spec.name, array))
    return model_inputs, resized_image


def parse_class_names(output_name: str) -> dict[int, str] | None:
    """'bboxes-format:xyxysc;0:face;1:...' -> {0: 'face', 1: ...}; None if the name doesn't follow that convention."""
    if not output_name.startswith(BOX_FORMAT_PREFIX):
        return None
    class_names_by_id = {}
    for entry in output_name[len(BOX_FORMAT_PREFIX) :].split(";"):
        class_id, _, class_name = entry.partition(":")
        if class_id.isdigit():
            class_names_by_id[int(class_id)] = class_name
    return class_names_by_id


def scale_up_normalized_boxes(rows: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """If every box in ``rows`` (columns [x1, y1, x2, y2, ...]) is a [0, 1] fraction rather than a
    pixel coordinate, scale it up to pixels of ``image_size`` (width, height); otherwise return as-is.
    """
    if rows.size == 0:
        return rows
    boxes = rows[:, :4]
    if not np.all((boxes >= 0.0) & (boxes <= 1.0)):
        return rows  # already in pixel coordinates
    width, height = image_size
    rows = rows.astype(np.float64, copy=True)
    rows[:, [0, 2]] *= width
    rows[:, [1, 3]] *= height
    return rows


def describe_output(output_tensor: Tensor, confidence_threshold: float, image_size: tuple[int, int]) -> list[Detection]:
    """Print what one output tensor contains; return any detections found in it.

    ``image_size`` is the (width, height) of the image actually fed to the
    model. Box coordinates are expected in pixels of that image, but some
    models report them normalized to [0, 1] instead; those are detected and
    scaled up automatically.
    """
    class_names_by_id = parse_class_names(output_tensor.name)
    if class_names_by_id is None or output_tensor.data.ndim != 2 or output_tensor.data.shape[1] < 6:
        # Not a bboxes-format tensor we know how to decode; report it as-is.
        print(f"  {output_tensor.name}: shape={output_tensor.data.shape} dtype={output_tensor.data.dtype}")
        return []

    rows = scale_up_normalized_boxes(output_tensor.data[:, :6], image_size)

    detections: list[Detection] = []
    for x1, y1, x2, y2, score, class_id in rows:
        if score < confidence_threshold:
            continue
        label = class_names_by_id.get(int(class_id), f"class {int(class_id)}")
        detections.append((label, float(score), (float(x1), float(y1), float(x2), float(y2))))

    if not detections:
        print(f"  {output_tensor.name}: no detections above {confidence_threshold:.0%} confidence")
    for label, score, (x1, y1, x2, y2) in detections:
        print(f"  {label:<20} {score:5.1%}   box=({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f})")
    return detections


def save_annotated_image(image: Image.Image, detections: list[Detection], output_path: Path) -> None:
    """Draw a box and label for each detection on a copy of ``image`` and save it to ``output_path``."""
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    for label, score, (x1, y1, x2, y2) in detections:
        draw.rectangle((x1, y1, x2, y2), outline="lime", width=2)
        draw.text((x1 + 2, y1 + 2), f"{label} {score:.0%}", fill="lime")
    annotated_image.save(output_path)


def ctc_decode(probabilities: np.ndarray, characters: list[str]) -> tuple[str, float]:
    """Turn a ``[positions, classes]`` matrix of character probabilities into ``(text, confidence)``.

    Follows the PaddleOCR convention for the class axis: class 0 is the CTC
    blank, classes 1..len(characters) are ``characters`` in order, and the
    last class is a space. Decoding keeps the most likely class at each
    position, collapses consecutive repeats, and drops blanks; the confidence
    is the mean probability of the characters kept.
    """
    best_classes = probabilities.argmax(axis=1)
    best_probabilities = probabilities.max(axis=1)

    keep = np.ones(len(best_classes), dtype=bool)
    keep[1:] = best_classes[1:] != best_classes[:-1]  # collapse repeats (a blank in between separates real doubles)
    keep &= best_classes != 0  # drop blanks
    if not keep.any():
        return "", 0.0

    class_labels = ["", *characters, " "]
    text = "".join(class_labels[class_id] for class_id in best_classes[keep])
    return text, float(best_probabilities[keep].mean())
