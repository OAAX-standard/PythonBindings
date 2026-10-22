# Faces

A minimal example: load an image, run it through any OAAX-compliant runtime,
and print the faces it found.

`detect_faces.py` works with any runtime library (Intel, a generic ONNX
Runtime-backed CPU runtime, ...). Everything it needs to know about the
*model* comes from a `model.json` sitting next to the model file - see
[Model metadata](#model-metadata) below. The face-detection model lives in
`../Models/Face-Detection/`:

| File | Purpose |
| --- | --- |
| `model.onnx` | UltraFace `version-RFB-320` (Linzaer, WIDERFace-trained, MIT licensed) - not checked in, fetched by `download_model.py` |
| `model.json` | the metadata the script reads |
| `download_model.py` | fetches `model.onnx` |

## Setup

```bash
pip install -e ".[examples]"       # from the repo root: adds Pillow
python ../Models/Face-Detection/download_model.py   # fetches model.onnx
```

`model.onnx` isn't checked into the repo; the script downloads it and is
safe to re-run, it skips the download if already present.

## Run

```bash
python detect_faces.py <runtime_library> <model_file>
```

For example, with a generic ONNX Runtime-backed CPU runtime:

```bash
python detect_faces.py ../../../runtime-downloads/CPU/libRuntimeLibrary.so ../Models/Face-Detection/model.onnx
```

Useful flags:

- `--image path/to.jpg` - analyse a different image (default: the bundled `faces.jpg`).
- `--threshold 0.3` - report lower-confidence detections too (default `0.5`).
- `--config key=value` (repeatable) - runtime-specific init options, e.g. `--config device_type=GPU`.
- `--output path/to.jpg` - where to save the annotated copy (default: next to the input image).

## Model metadata

The script never asks the runtime or the model file what the model's tensors
look like; it reads a `model.json` from the same directory as the model file
instead:

```json
{
  "name": "Face-Detection",
  "inputs": [
    { "name": "input", "element_type": "FLOAT", "shape": [1, 3, 240, 320],
      "mean": [127, 127, 127], "normalization": [128, 128, 128] }
  ],
  "outputs": [
    { "name": "scores", "element_type": "FLOAT", "shape": [1, 4420, 2], "role": "scores", "labels": ["BACKGROUND", "face"] },
    { "name": "boxes", "element_type": "FLOAT", "shape": [1, 4420, 4], "role": "boxes" }
  ]
}
```

- The one input carries `mean` and `normalization` (one value per channel).
  It is resized to the input's `(H, W)` and fed to the model as
  `(pixel - mean) / normalization`, with pixels in `0..255` and channels in
  RGB order (the default; `channel_order: "BGR"` is also supported, see the
  OCR example). Getting these two numbers wrong is the usual reason a
  detector returns garbage.
- `role: "scores"`/`role: "boxes"` mark the two halves of an anchor-based
  ("SSD-style") detector's raw output: per-anchor class scores, and
  per-anchor box regression offsets, neither NMS'd nor (for this model)
  decoded into coordinates yet. `labels` names the scores output's classes,
  index 0 always being background.
- `element_type` is a name from the ONNX/OAAX element-type enum (`FLOAT`,
  `UINT8`, `INT64`, ...).

## Output

The `scores`/`boxes` pair is decoded by `vision_utils.decode_ultraface_detections`:
per-anchor scores are thresholded per class, kept boxes are converted from
UltraFace's raw regression output into pixel coordinates using its own fixed
anchor layout (see the function's docstring for exactly where that layout
comes from), and overlapping boxes for the same class are collapsed with
standard IoU-based non-max suppression. This decode is specific to
UltraFace's architecture, unlike the rest of `vision_utils.py`, which stays
generic across models.

Textual detections go to stdout, one per line, and whenever at least one
clears the confidence threshold an annotated copy of the image is saved too:

```
Runtime: OAAX CPU Runtime v2.0.0
Model: Face-Detection - inputs ['input']

2 output tensor(s):
  face                 100.0%   box=(275, 12) - (313, 68)
  face                 100.0%   box=(209, 9) - (247, 72)
  face                 100.0%   box=(142, 11) - (177, 69)
  ...

Saved an annotated copy to faces.detections.jpg
```
