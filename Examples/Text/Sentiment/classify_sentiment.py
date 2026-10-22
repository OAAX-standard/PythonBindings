"""Classify the sentiment of sentences using any OAAX-compliant runtime.

    python classify_sentiment.py <runtime_library> <model_file> [--text "..."]...

Each --text is classified separately; without any, a few built-in sentences
are used. The model is described by a model.json in the same directory as
<model_file> (see ../Models/Sentiment-Classification/model.json): its
input/output tensors, the tokenizer that turns text into token ids, and the
label for each output class.

    pip install -e .[examples]   # from the repo root: the tokenizers package
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).parent
DEFAULT_TEXTS = [
    "I love this, it works beautifully.",
    "What a waste of time.",
    "The hotel was fine, but the breakfast was cold and the staff seemed bored.",
]

sys.path.insert(0, str(SCRIPT_DIRECTORY.parent))  # Examples/Text/, for text_utils

from text_utils import build_model_inputs, load_model_metadata, load_tokenizer, softmax  # noqa: E402

from oaax import Runtime  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("library", help="path to an OAAX runtime shared library")
    parser.add_argument("model", type=Path, help="path to a model file that library can load; a model.json must sit next to it")
    parser.add_argument("--text", action="append", default=None, help="a sentence to classify (repeatable; default: a few built-in ones)")
    parser.add_argument("--config", action="append", default=[], metavar="KEY=VALUE", help="runtime init option, repeatable")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    texts = args.text or DEFAULT_TEXTS
    runtime_init_options = dict(option.split("=", 1) for option in args.config)  # "key=value" strings -> {key: value}

    # Everything we need to know about the model - its tensors, its tokenizer,
    # what its output classes mean - comes from the model.json next to the model file.
    metadata = load_model_metadata(args.model)
    tokenizer = load_tokenizer(metadata)
    labels = metadata.outputs[0].labels
    if labels is None:
        raise SystemExit("this model's metadata doesn't name its output classes ('labels'), so there's nothing to report")

    # Connect to the runtime. The "with" block takes care of shutting it down
    # cleanly afterwards, even if something below raises an error.
    with Runtime(args.library) as runtime:
        print(f"Runtime: {runtime.name} v{runtime.version}")
        runtime.init(runtime_init_options)  # apply any vendor-specific options requested via --config

        # --- Load the model ---
        model_id = runtime.load_model(args.model)
        print(f"Model: {metadata.name} - inputs {[model_input.spec.name for model_input in metadata.inputs]}, classes {labels}\n")

        for text in texts:
            # Turn the sentence into token ids (padded to the model's fixed length) and the matching mask.
            model_inputs, _encoding = build_model_inputs(metadata, tokenizer, text)

            # --- Run inference ---
            inference_result = runtime.run(model_id, model_inputs, timeout_ms=30_000)

            # --- Parse the result ---
            # One raw score per class; softmax makes them probabilities, the largest wins.
            class_scores = inference_result.outputs[0].data[0]  # [0]: the single sentence in the batch
            probabilities = softmax(class_scores)
            best = int(probabilities.argmax())
            print(f"  {labels[best]:<9} {probabilities[best]:6.1%}   {text}")


if __name__ == "__main__":
    main()
