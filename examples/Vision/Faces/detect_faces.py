"""Detect faces in an image using any OAAX-compliant runtime.

    python detect_faces.py <runtime_library> <model_file> [--image path/to.jpg]

Defaults to the bundled faces.jpg. The model is described by a model.json in
the same directory as <model_file> (see ../Models/Face-Detection/model.json):
its input/output tensors, the mean/normalization to apply to the image, and
fixed values for any other inputs. The detection output is expected to follow
the "bboxes-format:xyxysc;<id>:<name>;..." naming convention used by the OAAX
illustrative examples: each output row is [x1, y1, x2, y2, score, class_id].

    pip install -e .[examples]   # from the repo root: Pillow for image I/O
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).parent
DEFAULT_IMAGE_PATH = SCRIPT_DIRECTORY / "faces.jpg"

sys.path.insert(0, str(SCRIPT_DIRECTORY.parent))  # examples/Vision/, for vision_utils

from vision_utils import build_model_inputs, describe_output, load_model_metadata, save_annotated_image  # noqa: E402

from oaax import Runtime  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("library", help="path to an OAAX runtime shared library")
    parser.add_argument("model", type=Path, help="path to a model file that library can load; a model.json must sit next to it")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH, help=f"image to analyse (default: {DEFAULT_IMAGE_PATH.name})")
    parser.add_argument("--output", type=Path, default=None, help="where to save an annotated copy (default: <image>.detections.jpg)")
    parser.add_argument("--threshold", type=float, default=0.5, help="minimum confidence to report a detection (default: 0.5)")
    parser.add_argument("--config", action="append", default=[], metavar="KEY=VALUE", help="runtime init option, repeatable")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotated_image_path = args.output or args.image.with_name(f"{args.image.stem}.detections.jpg")
    runtime_init_options = dict(option.split("=", 1) for option in args.config)  # "key=value" strings -> {key: value}

    # Everything we need to know about the model - its tensors and how to
    # preprocess for it - comes from the model.json next to the model file.
    metadata = load_model_metadata(args.model)

    # Connect to the runtime. The "with" block takes care of shutting it down
    # cleanly afterwards, even if something below raises an error.
    with Runtime(args.library) as runtime:
        print(f"Runtime: {runtime.name} v{runtime.version}")
        runtime.init(runtime_init_options)  # apply any vendor-specific options requested via --config

        # --- Load the model ---
        model_id = runtime.load_model(args.model)
        print(f"Model: {metadata.name} - inputs {[model_input.spec.name for model_input in metadata.inputs]}")

        # Resize the image to the model's input size and apply its mean/normalization;
        # any other inputs get the fixed value the metadata gives them.
        model_inputs, resized_image = build_model_inputs(metadata, args.image)

        # --- Run inference ---
        inference_result = runtime.run(model_id, model_inputs, timeout_ms=30_000)

        # --- Parse the result ---
        # Each output tensor is either a set of detected boxes or some other
        # data the model reports; describe_output() prints and decodes it.
        print(f"\n{len(inference_result.outputs)} output tensor(s):")
        detections = [
            detection
            for output_tensor in inference_result.outputs
            for detection in describe_output(output_tensor, args.threshold, resized_image.size)
        ]

        if detections:
            save_annotated_image(resized_image, detections, annotated_image_path)
            print(f"\nSaved an annotated copy to {annotated_image_path}")
        # else: nothing cleared the confidence threshold, so there's nothing to draw.


if __name__ == "__main__":
    main()
