# Faces

A minimal example: load an image, run it through any OAAX-compliant runtime,
and print the faces it found.

`detect_faces.py` works with any runtime library (Intel, a generic ONNX
Runtime-backed CPU runtime, ...). Everything it needs to know about the
*model* comes from a `model.json` sitting next to the model file - see
[Model metadata](#model-metadata) below. The face-detection model itself lives
in `../Models/Face-Detection/`:

| File | Use with |
| --- | --- |
| `model.onnx` | an ONNX Runtime-backed runtime (e.g. the generic CPU runtime) |
| `model.fp32.intel.zip`, `model.fp16.intel.zip` | the Intel runtime (OpenVINO IR, full or half precision) |
| `model.json` | every variant above - the metadata the script reads |

## Setup

```bash
pip install -e ".[examples]"   # from the repo root: adds Pillow
```

## Run

```bash
python detect_faces.py <runtime_library> <model_file>
```

For example, with a generic ONNX Runtime-backed CPU runtime:

```bash
python detect_faces.py ../../../runtime-downloads/CPU/libRuntimeLibrary.so ../Models/Face-Detection/model.onnx
```

or with the Intel runtime and its OpenVINO IR variant:

```bash
python detect_faces.py ../../../runtime-downloads/INTEL/RuntimeLibrary.dll ../Models/Face-Detection/model.fp32.intel.zip
```

Useful flags:

- `--image path/to.jpg` - analyse a different image (default: the bundled `faces.jpg`).
- `--threshold 0.3` - report lower-confidence detections too (default `0.5`).
- `--config key=value` (repeatable) - runtime-specific init options, e.g. `--config device_type=GPU`.
- `--output path/to.jpg` - where to save the annotated copy (default: next to the input image).

## Model metadata

The script never asks the runtime or the model file what the model's tensors
look like; it reads a `model.json` from the same directory as the model file
instead. One file describes every variant of a model:

```json
{
  "name": "Face-Detection",
  "inputs": [
    { "name": "image-", "element_type": "FLOAT", "shape": [1, 3, 240, 320],
      "mean": [127, 127, 127], "normalization": [128, 128, 128] },
    { "name": "nms_sensitivity-", "element_type": "FLOAT", "shape": [1], "value": 0.5 }
  ],
  "outputs": [
    { "name": "bboxes-format:xyxysc;0:Face", "element_type": "FLOAT", "shape": [20, 6] }
  ]
}
```

- Exactly one input carries `mean` and `normalization` (one value per
  channel): that's the image. It is resized to the input's `(H, W)` and fed
  to the model as `(pixel - mean) / normalization`, with pixels in `0..255`
  and channels in RGB order. Getting these two numbers wrong is the usual
  reason a detector returns garbage.
- Every other input carries a fixed `value`, broadcast to its shape (here a
  neutral NMS sensitivity).
- `element_type` is a name from the ONNX/OAAX element-type enum (`FLOAT`,
  `UINT8`, `INT64`, ...).

## Output

Each output tensor named `bboxes-format:xyxysc;<id>:<name>;...` - the
convention used by the OAAX illustrative examples, one row per box as
`[x1, y1, x2, y2, score, class_id]` - is decoded into detections. Coordinates
may be pixels of the resized input image or `[0, 1]` fractions of it; the
latter are detected and scaled up. Any other output is still printed (name,
shape, dtype), just not decoded.

Textual detections go to stdout, one per line, and whenever at least one
clears the confidence threshold an annotated copy of the image is saved too:

```
Runtime: OAAX CPU Runtime v2.0.0
Model: Face-Detection - inputs ['image-', 'nms_sensitivity-']

1 output tensor(s):
  Face                 100.0%   box=(208, 10) - (247, 71)
  Face                 100.0%   box=(273, 180) - (309, 236)
  Face                 100.0%   box=(274, 12) - (311, 69)
  ...

Saved an annotated copy to faces.detections.jpg
```
