"""Compare sentences by meaning using any OAAX-compliant runtime.

    python compare_sentences.py <runtime_library> <model_file> [--sentence "..."]...

Each sentence is turned into an embedding - a vector that places texts with
similar meaning close together - and every pair is scored by how close they
are. Without any --sentence, a few built-in ones are used. The model is
described by a model.json in the same directory as <model_file> (see
../Models/Sentence-Embeddings/model.json): its input/output tensors and the
tokenizer that turns text into token ids.

    pip install -e .[examples]   # from the repo root: the tokenizers package
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).parent
DEFAULT_SENTENCES = [
    "The cat sits on the mat.",
    "A kitten is resting on a rug.",
    "Stock markets fell sharply today.",
    "Investors lost money as shares dropped.",
    "How do I bake sourdough bread?",
]

sys.path.insert(0, str(SCRIPT_DIRECTORY.parent))  # Examples/Text/, for text_utils

from text_utils import build_model_inputs, cosine_similarity, load_model_metadata, load_tokenizer, mean_pool  # noqa: E402

from oaax import Runtime  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("library", help="path to an OAAX runtime shared library")
    parser.add_argument("model", type=Path, help="path to a model file that library can load; a model.json must sit next to it")
    parser.add_argument("--sentence", action="append", default=None, help="a sentence to compare (repeatable, at least two; default: a few built-in ones)")
    parser.add_argument("--config", action="append", default=[], metavar="KEY=VALUE", help="runtime init option, repeatable")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sentences = args.sentence or DEFAULT_SENTENCES
    if len(sentences) < 2:
        raise SystemExit("give at least two --sentence values to compare")
    runtime_init_options = dict(option.split("=", 1) for option in args.config)  # "key=value" strings -> {key: value}

    # Everything we need to know about the model - its tensors and its tokenizer -
    # comes from the model.json next to the model file.
    metadata = load_model_metadata(args.model)
    tokenizer = load_tokenizer(metadata)

    # Connect to the runtime. The "with" block takes care of shutting it down
    # cleanly afterwards, even if something below raises an error.
    with Runtime(args.library) as runtime:
        print(f"Runtime: {runtime.name} v{runtime.version}")
        runtime.init(runtime_init_options)  # apply any vendor-specific options requested via --config

        # --- Load the model ---
        model_id = runtime.load_model(args.model)
        print(f"Model: {metadata.name} - inputs {[model_input.spec.name for model_input in metadata.inputs]}\n")

        embeddings = []
        for index, sentence in enumerate(sentences):
            # Turn the sentence into token ids (padded to the model's fixed length) and the matching mask.
            model_inputs, encoding = build_model_inputs(metadata, tokenizer, sentence)

            # --- Run inference ---
            inference_result = runtime.run(model_id, model_inputs, timeout_ms=30_000)

            # --- Parse the result ---
            # The model returns one vector per token position; averaging the real
            # (non-padding) ones gives a single vector that stands for the sentence.
            token_vectors = inference_result.outputs[0].data[0]  # [0]: the single sentence in the batch
            embeddings.append(mean_pool(token_vectors, encoding.attention_mask))
            print(f"  [{index}] {sentence}")

        # Score every pair: 1.0 means "same meaning", around 0 means unrelated.
        print("\nSimilarity:\n" + " " * 6 + "".join(f"{f'[{index}]':>6}" for index in range(len(sentences))))
        for row, embedding in enumerate(embeddings):
            scores = "".join(f"{cosine_similarity(embedding, other):6.2f}" for other in embeddings)
            print(f"  [{row}] {scores}")

        pairs = [(cosine_similarity(embeddings[a], embeddings[b]), a, b) for a in range(len(sentences)) for b in range(a + 1, len(sentences))]
        score, a, b = max(pairs)
        print(f"\nMost similar pair: [{a}] and [{b}] ({score:.2f})")


if __name__ == "__main__":
    main()
