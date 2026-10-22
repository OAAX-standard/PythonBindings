"""Shared helpers for the OAAX vision examples.

Covers the parts every such example needs: reading a model's ``model.json``
metadata (its tensors and how to preprocess for it), preparing an image and
any other inputs accordingly, and decoding the two output conventions the
examples use - UltraFace's separate (scores, boxes) anchor-based detector
output, and CTC character scores from a text recogniser. Import this from a
script under examples/Vision/<name>/ rather than duplicating it.
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
CHANNEL_ORDERS = ("RGB", "BGR")
RESIZE_MODES = ("stretch", "keep-aspect")
OUTPUT_ROLES = ("scores", "boxes")

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
    """One model output.

    ``characters`` is set for a CTC text output (see :func:`ctc_decode`).
    ``role`` (``"scores"`` or ``"boxes"``) marks one half of UltraFace's
    detector output pair (see :func:`decode_ultraface_detections`); ``labels``
    names the scores output's classes, index 0 being background.
    """

    spec: TensorSpec
    characters: list[str] | None = None
    role: str | None = None
    labels: list[str] | None = None


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
    if "characters" in entry:
        characters = (metadata_directory / entry["characters"]).read_text(encoding="utf-8").splitlines()
        if len(characters) + 2 != spec.shape[-1]:
            raise SystemExit(
                f"output {spec.name!r}: {entry['characters']} lists {len(characters)} characters, but the output has "
                f"{spec.shape[-1]} classes; expected the characters plus a CTC blank and a space"
            )
        return ModelOutput(spec, characters=characters)

    role = entry.get("role")
    if role is not None and role not in OUTPUT_ROLES:
        raise SystemExit(f"output {spec.name!r}: 'role' must be one of {OUTPUT_ROLES}, got {role!r}")
    return ModelOutput(spec, role=role, labels=entry.get("labels"))


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


def outputs_by_role(metadata: ModelMetadata, inference_outputs: list[Tensor]) -> dict[str, Tensor]:
    """Pick out the ``"scores"``/``"boxes"`` tensors an anchor-based detector's model.json names by role."""
    tensors_by_name = {tensor.name: tensor for tensor in inference_outputs}
    return {output.role: tensors_by_name[output.spec.name] for output in metadata.outputs if output.role is not None}


def _non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy IoU-based NMS: keep the highest-scoring box, drop anything that overlaps it past
    ``iou_threshold``, repeat with what's left. ``boxes`` are corner-form (x1, y1, x2, y2).
    Returns the kept row indices, highest score first. Standard algorithm; this implementation
    follows the reference one in Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB (MIT licensed).
    """
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while order.size:
        current, rest = order[0], order[1:]
        keep.append(int(current))
        if rest.size == 0:
            break
        overlap_x1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        overlap_y1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        overlap_x2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        overlap_y2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        intersection = np.clip(overlap_x2 - overlap_x1, 0, None) * np.clip(overlap_y2 - overlap_y1, 0, None)
        area_current = (boxes[current, 2] - boxes[current, 0]) * (boxes[current, 3] - boxes[current, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = intersection / (area_current + area_rest - intersection + 1e-5)
        order = rest[iou <= iou_threshold]
    return np.array(keep, dtype=np.int64)


# --- UltraFace anchor decode -------------------------------------------
#
# UltraFace's "boxes" output is raw regression offsets relative to a fixed
# set of anchor ("prior") boxes - a plain Concat of the regression heads,
# no decode baked into the graph - not already-decoded coordinates. The
# prior boxes and the decode formula below are specific to this
# architecture at its 320x240 input resolution and come straight from its
# own reference implementation, MIT licensed:
# https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
# (vision/utils/box_utils.py's generate_priors()/convert_locations_to_boxes(),
# vision/ssd/config/fd_config.py's per-layer feature map and anchor sizes).

_PRIOR_FEATURE_MAP_SIZES = ((40, 30), (20, 15), (10, 8), (5, 4))  # (width, height) cells, per layer
_PRIOR_ANCHOR_SIZES_PX = ((10, 16, 24), (32, 48), (64, 96), (128, 192, 256))  # anchor box sizes, per layer
_PRIOR_IMAGE_SIZE = (320, 240)  # (width, height) this prior set is built for
_PRIOR_CENTER_VARIANCE = 0.1
_PRIOR_SIZE_VARIANCE = 0.2


def _generate_priors() -> np.ndarray:
    """The fixed anchor ("prior") boxes UltraFace's regression output is relative to.

    One row per anchor, ``(center_x, center_y, width, height)`` as a fraction
    of the image size. Always the same 4420 boxes for this architecture at
    320x240, so this only ever needs computing once (see module load below).
    """
    image_width, image_height = _PRIOR_IMAGE_SIZE
    priors = [
        (center_x, center_y, anchor_size / image_width, anchor_size / image_height)
        for (cells_wide, cells_high), anchor_sizes in zip(_PRIOR_FEATURE_MAP_SIZES, _PRIOR_ANCHOR_SIZES_PX)
        for row in range(cells_high)
        for col in range(cells_wide)
        for center_x, center_y in [((col + 0.5) / cells_wide, (row + 0.5) / cells_high)]
        for anchor_size in anchor_sizes
    ]
    return np.clip(np.array(priors, dtype=np.float64), 0.0, 1.0)


_PRIORS = _generate_priors()  # computed once at import time; every call below reuses it


def _decode_ultraface_boxes(raw_boxes: np.ndarray) -> np.ndarray:
    """Turn UltraFace's raw per-anchor regression output into normalized ``(x1, y1, x2, y2)`` boxes."""
    center = raw_boxes[:, :2] * _PRIOR_CENTER_VARIANCE * _PRIORS[:, 2:] + _PRIORS[:, :2]
    size = np.exp(raw_boxes[:, 2:] * _PRIOR_SIZE_VARIANCE) * _PRIORS[:, 2:]
    return np.concatenate([center - size / 2, center + size / 2], axis=1)


def decode_ultraface_detections(
    scores: Tensor,
    boxes: Tensor,
    labels: list[str] | None,
    confidence_threshold: float,
    image_size: tuple[int, int],
    iou_threshold: float = 0.3,
) -> list[Detection]:
    """Decode UltraFace's raw ``(scores, boxes)`` output pair into detections.

    ``scores`` is ``[1, num_priors, num_classes]`` (class 0 is background);
    ``boxes`` is ``[1, num_priors, 4]``, raw per-anchor regression output (see
    ``_decode_ultraface_boxes``). ``image_size`` is the (width, height) of the
    image actually fed to the model, to scale the normalized boxes to pixels.
    """
    class_scores = scores.data[0]  # drop the batch axis
    decoded_boxes = _decode_ultraface_boxes(boxes.data[0])
    width, height = image_size

    detections: list[Detection] = []
    for class_id in range(1, class_scores.shape[1]):  # 0 is background
        keep = class_scores[:, class_id] > confidence_threshold
        if not keep.any():
            continue
        candidate_boxes, candidate_scores = decoded_boxes[keep], class_scores[keep, class_id]
        picked = _non_max_suppression(candidate_boxes, candidate_scores, iou_threshold)
        label = labels[class_id] if labels and class_id < len(labels) else f"class {class_id}"
        for (x1, y1, x2, y2), score in zip(candidate_boxes[picked], candidate_scores[picked]):
            detections.append((label, float(score), (float(x1) * width, float(y1) * height, float(x2) * width, float(y2) * height)))

    if not detections:
        print(f"  no detections above {confidence_threshold:.0%} confidence")
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
