"""Detect faces (or other objects) in an image using any OAAX-compliant runtime.

    python detect_faces.py <runtime_library> <model_file> [--image path/to.jpg]

Defaults to the bundled faces.jpg. Works with any model whose input/output
tensors can be determined (via the runtime's runtime_get_info(), or, for a
bare .onnx file, the graph itself) and whose detection output follows the
"bboxes-format:xyxysc;<id>:<name>;..." naming convention used by the OAAX
illustrative examples: each output row is [x1, y1, x2, y2, score, class_id]
in pixel coordinates of the *resized* input image.

    pip install -e .[examples]   # from the repo root: Pillow for image I/O
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from oaax import Runtime, Tensor, TensorSpec
from oaax.onnx_introspect import model_io

HERE = Path(__file__).parent
DEFAULT_IMAGE = HERE / "faces.jpg"
BOX_FORMAT_PREFIX = "bboxes-format:xyxysc;"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("library", help="path to an OAAX runtime shared library")
    parser.add_argument("model", help="path to a model file that library can load")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help=f"image to analyse (default: {DEFAULT_IMAGE.name})")
    parser.add_argument("--output", type=Path, default=None, help="where to save an annotated copy (default: <image>.detections.jpg)")
    parser.add_argument("--threshold", type=float, default=0.5, help="minimum confidence to report a detection (default: 0.5)")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="scale pixels to [0, 1] before feeding the model, for a model that expects that instead of raw 0-255 values",
    )
    parser.add_argument("--config", action="append", default=[], metavar="KEY=VALUE", help="runtime init option, repeatable")
    return parser.parse_args()


def infer_image_size(shape: tuple[int, ...]) -> tuple[int, int]:
    """(height, width) from an NCHW shape; dynamic dimensions (-1) default to 224."""
    if len(shape) != 4:
        raise SystemExit(f"expected a 4D (N, C, H, W) image input, got shape {shape}")
    _, _, height, width = shape
    return (height if height > 0 else 224), (width if width > 0 else 224)


def load_image(path: Path, size: tuple[int, int], normalize: bool) -> tuple[np.ndarray, Image.Image]:
    """Load an image and resize it to ``size``; returns an NCHW float32 array plus the resized PIL image.

    Pixel scaling is genuinely model-specific: many detection graphs (this
    included) bake their own '/255' step in right after the input, so they
    expect raw 0-255 values, while others expect the caller to normalise to
    [0, 1] first. There's no general way to tell from the outside which one a
    given model wants; default to raw pixels and pass ``normalize=True``
    (``--normalize`` on the command line) for a model that needs the other
    convention. If detections come back empty or nonsensical, that's the
    first thing to try flipping.
    """
    height, width = size
    image = Image.open(path).convert("RGB").resize((width, height))
    array = np.asarray(image, dtype=np.float32)
    if normalize:
        array = array / 255.0
    return np.transpose(array, (2, 0, 1))[np.newaxis, ...], image


def placeholder_for(spec: TensorSpec) -> np.ndarray:
    """A neutral value for an input this example has no real data for (e.g. a mask or sensitivity knob)."""
    shape = tuple(d if d > 0 else 1 for d in spec.shape)
    dtype = spec.dtype or np.float32
    if dtype.kind == "b":
        return np.ones(shape, dtype=dtype)  # e.g. an all-true "don't mask anything" mask
    return np.full(shape, 0.5, dtype=dtype)  # e.g. a middle-of-the-road sensitivity/threshold


def build_inputs(input_specs: list[TensorSpec], image_path: Path, normalize: bool) -> tuple[list[Tensor], Image.Image]:
    """Feed the image to whichever input looks like one; fill any others with a placeholder."""
    image_spec = next((spec for spec in input_specs if len(spec.shape) == 4), input_specs[0])
    array, image = load_image(image_path, infer_image_size(image_spec.shape), normalize)
    tensors = [Tensor(image_spec.name, array)]
    for spec in input_specs:
        if spec.name == image_spec.name:
            continue
        print(f"  input {spec.name!r} isn't the image; using a placeholder value (this model may need real data here)")
        tensors.append(Tensor(spec.name, placeholder_for(spec)))
    return tensors, image


def parse_class_names(output_name: str) -> dict[int, str] | None:
    """'bboxes-format:xyxysc;0:face;1:...' -> {0: 'face', 1: ...}; None if the name doesn't follow that convention."""
    if not output_name.startswith(BOX_FORMAT_PREFIX):
        return None
    names = {}
    for pair in output_name[len(BOX_FORMAT_PREFIX) :].split(";"):
        index, _, name = pair.partition(":")
        if index.isdigit():
            names[int(index)] = name
    return names


Detection = tuple[str, float, tuple[float, float, float, float]]


def describe_output(tensor: Tensor, threshold: float) -> list[Detection]:
    """Print what one output tensor contains; return any detections found in it."""
    class_names = parse_class_names(tensor.name)
    if class_names is None or tensor.data.ndim != 2 or tensor.data.shape[1] < 6:
        print(f"  {tensor.name}: shape={tensor.data.shape} dtype={tensor.data.dtype}")
        return []

    detections: list[Detection] = []
    for x1, y1, x2, y2, score, class_id in tensor.data[:, :6]:
        if score < threshold:
            continue
        label = class_names.get(int(class_id), f"class {int(class_id)}")
        detections.append((label, float(score), (float(x1), float(y1), float(x2), float(y2))))

    if not detections:
        print(f"  {tensor.name}: no detections above {threshold:.0%} confidence")
    for label, score, (x1, y1, x2, y2) in detections:
        print(f"  {label:<20} {score:5.1%}   box=({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f})")
    return detections


def save_annotated(image: Image.Image, detections: list[Detection], path: Path) -> None:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for label, score, (x1, y1, x2, y2) in detections:
        draw.rectangle((x1, y1, x2, y2), outline="lime", width=2)
        draw.text((x1 + 2, y1 + 2), f"{label} {score:.0%}", fill="lime")
    annotated.save(path)


def main() -> None:
    args = parse_args()
    output_path = args.output or args.image.with_name(f"{args.image.stem}.detections.jpg")
    config = dict(pair.split("=", 1) for pair in args.config)

    with Runtime(args.library) as rt:
        print(f"Runtime: {rt.name} v{rt.version}")
        rt.init(config)

        model_id = rt.load_model(args.model)
        io, reason = model_io(rt, model_id, args.model)
        if io is None:
            raise SystemExit(f"Don't know what this model expects: {reason}")
        input_specs, _ = io

        print(f"Model: {len(input_specs)} input(s) -> {[spec.name for spec in input_specs]}")
        inputs, image = build_inputs(input_specs, args.image, args.normalize)

        result = rt.run(model_id, inputs, timeout_ms=30_000)

        print(f"\n{len(result.outputs)} output tensor(s):")
        detections = [d for tensor in result.outputs for d in describe_output(tensor, args.threshold)]

        if detections:
            save_annotated(image, detections, output_path)
            print(f"\nSaved an annotated copy to {output_path}")


if __name__ == "__main__":
    main()
