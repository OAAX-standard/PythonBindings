# Similarity

A minimal example: hand sentences to any OAAX-compliant runtime and find out
which ones mean the same thing.

`compare_sentences.py` works with any runtime library (Intel, a generic ONNX
Runtime-backed CPU runtime, ...). Everything it needs to know about the
*model* comes from a `model.json` sitting next to the model file - see
[Model metadata](#model-metadata) below. The model lives in
`../Models/Sentence-Embeddings/`:

| File | Purpose |
| --- | --- |
| `model.onnx` | `all-MiniLM-L6-v2` (sentence-transformers, Apache-2.0), 8-bit quantised ONNX export from Hugging Face's `Xenova` org - not checked in, fetched by `download_model.py` |
| `tokenizer.json` | the model's own tokenizer, which turns text into the token ids it was trained on - also fetched by `download_model.py` |
| `model.json` | the metadata the script reads |
| `download_model.py` | fetches the two files above |

## Setup

```bash
pip install -e ".[examples]"                      # from the repo root: adds the tokenizers package
python ../Models/Sentence-Embeddings/download_model.py   # fetches model.onnx and tokenizer.json
```

Neither file is checked into the repo; the script downloads them and is
safe to re-run, it skips anything already present.

## Run

```bash
python compare_sentences.py <runtime_library> <model_file> [--sentence "..."]...
```

For example, with a generic ONNX Runtime-backed CPU runtime:

```bash
python compare_sentences.py ../../../RuntimeDownloads/CPU/libRuntimeLibrary.so ../Models/Sentence-Embeddings/model.onnx
```

Useful flags:

- `--sentence "..."` (repeatable, at least two) - the sentences to compare; without it, a few built-in ones are used.
- `--config key=value` (repeatable) - runtime-specific init options, e.g. `--config device_type=GPU`.

## Model metadata

```json
{
  "name": "Sentence-Embeddings",
  "tokenizer": "tokenizer.json",
  "inputs": [
    { "name": "input_ids",      "element_type": "INT64", "shape": [1, 128], "role": "token_ids" },
    { "name": "attention_mask", "element_type": "INT64", "shape": [1, 128], "role": "attention_mask" },
    { "name": "token_type_ids", "element_type": "INT64", "shape": [1, 128], "role": "token_type_ids" }
  ],
  "outputs": [
    { "name": "last_hidden_state", "element_type": "FLOAT", "shape": [1, 128, 384] }
  ]
}
```

Same scheme as the sentiment example (see its README for what `tokenizer`
and `role` mean); this model additionally wants `token_type_ids`, which the
tokenizer also produces. The output is not an answer but a 384-number vector
for each of the 128 token positions; the script averages the positions that
hold real tokens (the `attention_mask` says which) into one vector per
sentence, then compares sentences by the cosine of the angle between their
vectors: 1.0 means the same meaning, around 0 means unrelated.

## Output

```
Runtime: OAAX CPU Runtime v2.0.0
Model: Sentence-Embeddings - inputs ['input_ids', 'attention_mask', 'token_type_ids']

  [0] The cat sits on the mat.
  [1] A kitten is resting on a rug.
  [2] Stock markets fell sharply today.
  [3] Investors lost money as shares dropped.
  [4] How do I bake sourdough bread?

Similarity:
         [0]   [1]   [2]   [3]   [4]
  [0]   1.00  0.64  0.05 -0.00  0.08
  [1]   0.64  1.00  0.05  0.02 -0.04
  [2]   0.05  0.05  1.00  0.62  0.02
  [3]  -0.00  0.02  0.62  1.00  0.00
  [4]   0.08 -0.04  0.02  0.00  1.00

Most similar pair: [0] and [1] (0.64)
```
