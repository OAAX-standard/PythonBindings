# OCR

A minimal example: load an image of one line of text, run it through any
OAAX-compliant runtime, and print the text it reads.

`read_text.py` works with any runtime library (Intel, a generic ONNX
Runtime-backed CPU runtime, ...). Everything it needs to know about the
*model* comes from a `model.json` sitting next to the model file - see
[Model metadata](#model-metadata) below. The recognition model lives in
`../Models/Text-Recognition/`:

| File | Purpose |
| --- | --- |
| `model.onnx` | PaddleOCR PP-OCRv4 text recogniser (Chinese + English, Apache-2.0), as shipped inside the `rapidocr_onnxruntime` PyPI package - not checked in, fetched by `download_model.py` |
| `characters.txt` | the 6623 characters it can output, one per line, in class order (PaddleOCR's `ppocr_keys_v1.txt`) - also fetched by `download_model.py` |
| `model.json` | the metadata the script reads |
| `download_model.py` | fetches the two files above |

This is recognition only: it expects an image that already contains just one
line of text, like the bundled `text.png`. Finding the lines in a larger image
is a separate detection model and a second inference, deliberately out of
scope here.

## Setup

```bash
pip install -e ".[examples]"                   # from the repo root: adds Pillow
python ../Models/Text-Recognition/download_model.py   # fetches model.onnx and characters.txt
```

Neither file is checked into the repo; the script downloads them and is
safe to re-run, it skips anything already present.

## Run

```bash
python read_text.py <runtime_library> <model_file>
```

For example, with a generic ONNX Runtime-backed CPU runtime:

```bash
python read_text.py ../../../runtime-downloads/CPU/libRuntimeLibrary.so ../Models/Text-Recognition/model.onnx
```

Useful flags:

- `--image path/to.png` - read a different image (default: the bundled `text.png`).
- `--config key=value` (repeatable) - runtime-specific init options, e.g. `--config device_type=GPU`.

## Model metadata

```json
{
  "name": "Text-Recognition",
  "inputs": [
    { "name": "x", "element_type": "FLOAT", "shape": [1, 3, 48, 320],
      "channel_order": "BGR", "resize": "keep-aspect",
      "mean": [127.5, 127.5, 127.5], "normalization": [127.5, 127.5, 127.5] }
  ],
  "outputs": [
    { "name": "softmax_11.tmp_0", "element_type": "FLOAT", "shape": [1, 40, 6625],
      "characters": "characters.txt" }
  ]
}
```

Compared with the face detector's metadata, the image input uses two more
fields:

- `channel_order: "BGR"` - the model was trained on OpenCV-style images, so
  the channels are swapped before `mean`/`normalization` are applied (which
  are given in the model's order). Default `"RGB"`.
- `resize: "keep-aspect"` - scale the image to the input height and keep its
  proportions, then pad the right edge up to the input width with the mean
  colour (so the padding normalises to zero). An image wider than that is
  squashed to fit. Default `"stretch"`, which resizes to the exact `(H, W)`.

The output carries a `characters` file: one character per line, in the order
of the output's class axis, following the PaddleOCR convention that class 0
is the CTC blank and the last class is a space (so the file has two fewer
lines than the class axis has entries; the script checks that).

## Output

The output tensor is `[1, 40, 6625]`: for each of 40 positions along the
image, a probability for every character. CTC decoding takes the most likely
character at each position, collapses consecutive repeats, and drops blanks;
the reported confidence is the mean probability of the characters kept.

```
Runtime: OAAX CPU Runtime v2.0.0
Model: Text-Recognition - inputs ['x']

1 output tensor(s):
  'Hello from OAAX 2.0'   (97.7% confidence)
```
