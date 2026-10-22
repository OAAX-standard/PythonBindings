# Sentiment

A minimal example: hand sentences to any OAAX-compliant runtime and print
whether each one is positive or negative.

`classify_sentiment.py` works with any runtime library (Intel, a generic ONNX
Runtime-backed CPU runtime, ...). Everything it needs to know about the
*model* comes from a `model.json` sitting next to the model file - see
[Model metadata](#model-metadata) below. The model lives in
`../Models/Sentiment-Classification/`:

| File | Purpose |
| --- | --- |
| `model.onnx` | DistilBERT fine-tuned on SST-2 (`distilbert-base-uncased-finetuned-sst-2-english`, Apache-2.0), 8-bit quantised ONNX export from Hugging Face's `Xenova` org - not checked in, fetched by `download_model.py` |
| `tokenizer.json` | the model's own tokenizer, which turns text into the token ids it was trained on - also fetched by `download_model.py` |
| `model.json` | the metadata the script reads |
| `download_model.py` | fetches the two files above |

## Setup

```bash
pip install -e ".[examples]"                        # from the repo root: adds the tokenizers package
python ../Models/Sentiment-Classification/download_model.py   # fetches model.onnx and tokenizer.json
```

Neither file is checked into the repo; the script downloads them (see
[Model metadata](#model-metadata) for where from) and is safe to re-run, it
skips anything already present.

## Run

```bash
python classify_sentiment.py <runtime_library> <model_file> [--text "..."]...
```

For example, with a generic ONNX Runtime-backed CPU runtime:

```bash
python classify_sentiment.py ../../../runtime-downloads/CPU/libRuntimeLibrary.so ../Models/Sentiment-Classification/model.onnx --text "Best purchase I've made all year."
```

Useful flags:

- `--text "..."` (repeatable) - the sentences to classify; without it, a few built-in ones are used.
- `--config key=value` (repeatable) - runtime-specific init options, e.g. `--config device_type=GPU`.

## Model metadata

```json
{
  "name": "Sentiment-Classification",
  "tokenizer": "tokenizer.json",
  "inputs": [
    { "name": "input_ids",      "element_type": "INT64", "shape": [1, 128], "role": "token_ids" },
    { "name": "attention_mask", "element_type": "INT64", "shape": [1, 128], "role": "attention_mask" }
  ],
  "outputs": [
    { "name": "logits", "element_type": "FLOAT", "shape": [1, 2], "labels": ["NEGATIVE", "POSITIVE"] }
  ]
}
```

A text model doesn't take the text itself; it takes token ids. `tokenizer`
names the file (shipped with every Hugging Face model) that does that
conversion, and each input's `role` says which part of the tokenizer's output
fills it: the ids themselves, the `attention_mask` that marks which positions
are real tokens rather than padding, or (for models that want it) the
`token_type_ids`. The shape's second dimension is the fixed sequence length
every text is padded or truncated to - 128 tokens is plenty for a sentence
and keeps the tensor shapes static, which is what a compiled runtime variant
would need anyway.

The output's `labels` name its classes in order, taken from the model's
`config.json` on Hugging Face.

## Output

```
Runtime: OAAX CPU Runtime v2.0.0
Model: Sentiment-Classification - inputs ['input_ids', 'attention_mask'], classes ['NEGATIVE', 'POSITIVE']

  POSITIVE  100.0%   I love this, it works beautifully.
  NEGATIVE  100.0%   What a waste of time.
  NEGATIVE  100.0%   The hotel was fine, but the breakfast was cold and the staff seemed bored.
```

The two numbers the model returns are raw scores; the script runs them
through a softmax to get the probability shown.
