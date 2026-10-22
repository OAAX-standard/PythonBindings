"""Read the text in an image of a single line using any OAAX-compliant runtime.

    python read_text.py <runtime_library> <model_file> [--image path/to.png]

Defaults to the bundled text.png. The model is described by a model.json in
the same directory as <model_file> (see ../Models/Text-Recognition/model.json):
its input/output tensors, how to prepare the image, and the character
dictionary its output is decoded with. The model is a CTC text recogniser: it
scores every character at each of a fixed number of positions along the
image, and decoding collapses those scores into a string.

    pip install -e .[examples]   # from the repo root: Pillow for image I/O
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).parent
DEFAULT_IMAGE_PATH = SCRIPT_DIRECTORY / "text.png"

sys.path.insert(0, str(SCRIPT_DIRECTORY.parent))  # Examples/Vision/, for vision_utils

from vision_utils import build_model_inputs, ctc_decode, load_model_metadata  # noqa: E402

from oaax import Runtime  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("library", help="path to an OAAX runtime shared library")
    parser.add_argument("model", type=Path, help="path to a model file that library can load; a model.json must sit next to it")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH, help=f"image of one line of text (default: {DEFAULT_IMAGE_PATH.name})")
    parser.add_argument("--config", action="append", default=[], metavar="KEY=VALUE", help="runtime init option, repeatable")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_init_options = dict(option.split("=", 1) for option in args.config)  # "key=value" strings -> {key: value}

    # Everything we need to know about the model - its tensors, how to prepare
    # the image, which characters it can output - comes from the model.json
    # next to the model file.
    metadata = load_model_metadata(args.model)

    # Connect to the runtime. The "with" block takes care of shutting it down
    # cleanly afterwards, even if something below raises an error.
    with Runtime(args.library) as runtime:
        print(f"Runtime: {runtime.name} v{runtime.version}")
        runtime.init(runtime_init_options)  # apply any vendor-specific options requested via --config

        # --- Load the model ---
        model_id = runtime.load_model(args.model)
        print(f"Model: {metadata.name} - inputs {[model_input.spec.name for model_input in metadata.inputs]}")

        # Scale the image to the model's input height, pad it to the input width
        # (keeping the text's proportions) and apply the model's mean/normalization.
        model_inputs, _resized_image = build_model_inputs(metadata, args.image)

        # --- Run inference ---
        inference_result = runtime.run(model_id, model_inputs, timeout_ms=30_000)

        # --- Parse the result ---
        # The output is one row of character scores per position along the image;
        # ctc_decode() turns those rows into a string using the model's dictionary.
        characters_by_output = {output.spec.name: output.characters for output in metadata.outputs}
        print(f"\n{len(inference_result.outputs)} output tensor(s):")
        for output_tensor in inference_result.outputs:
            characters = characters_by_output.get(output_tensor.name)
            if characters is None:
                print(f"  {output_tensor.name}: shape={output_tensor.data.shape} dtype={output_tensor.data.dtype}")
                continue
            text, confidence = ctc_decode(output_tensor.data[0], characters)  # [0]: the single image in the batch
            print(f"  {text!r}   ({confidence:.1%} confidence)")


if __name__ == "__main__":
    main()
