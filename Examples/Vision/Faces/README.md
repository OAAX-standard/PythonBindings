# Faces

A minimal example: load an image, run it through any OAAX-compliant runtime,
and print what came back.

`detect_faces.py` is intentionally generic. It works with:

- any runtime library (Intel, a generic ONNX Runtime-backed CPU runtime, ...);
- any detection model whose runtime reports its inputs (via
  `runtime_get_info()`, or - for a bare `.onnx` file - the graph itself);
- any output tensor named `bboxes-format:xyxysc;<id>:<name>;...`, the
  convention used by the OAAX illustrative examples, where each row is
  `[x1, y1, x2, y2, score, class_id]`. An output that doesn't follow this
  convention is still printed (name, shape, dtype), just not decoded into
  boxes.

If a model takes inputs besides the image (a sensitivity knob, a mask, ...),
the script fills them with a neutral placeholder and says so - it has no way
to know what real values that particular model expects there.

## Setup

```bash
pip install -e ".[examples]"   # from the repo root: adds Pillow
```

## Run

```bash
python detect_faces.py <runtime_library> <model_file>
```

For example, with the Intel runtime and an OpenVINO IR model:

```bash
python detect_faces.py ../../../RuntimeDownloads/INTEL/RuntimeLibrary.dll path/to/model.zip
```

or a generic ONNX Runtime-backed runtime loading a plain `.onnx` file directly:

```bash
python detect_faces.py ../../../RuntimeDownloads/GenericCPU/RuntimeLibrary.dll path/to/model.onnx
```

Useful flags:

- `--image path/to.jpg` - analyse a different image (default: the bundled `faces.jpg`).
- `--threshold 0.3` - report lower-confidence detections too (default `0.5`).
- `--normalize` - scale pixels to `[0, 1]` before feeding the model. Off by
  default (raw `0..255` values), since that's what worked for the models this
  was tested against; pixel scaling is genuinely model-specific (some graphs
  bake their own `/255` step in, others expect the caller to normalise), so
  if detections come back empty or nonsensical, this is the first thing to try.
- `--config key=value` (repeatable) - runtime-specific init options, e.g. `--config device_type=GPU`.
- `--output path/to.jpg` - where to save the annotated copy (default: next to the input image).

## Output

Textual detections on stdout, one per line. This is real output from a
generic ONNX Runtime-backed CPU runtime against an OAAX object-detection
model, on the bundled `faces.jpg` (a grid of face photos):

```
Runtime: OAAX CPU Runtime v2.0.0
Model: 3 input(s) -> ['image-', 'nms_sensitivity-', 'mask-']
  input 'nms_sensitivity-' isn't the image; using a placeholder value (this model may need real data here)
  input 'mask-' isn't the image; using a placeholder value (this model may need real data here)

1 output tensor(s):
  person               92.8%   box=(45, 223) - (128, 317)
  person               82.7%   box=(197, 3) - (259, 97)
  person               81.8%   box=(196, 222) - (265, 320)
  ...

Saved an annotated copy to faces.detections.jpg
```

Detections are labelled with whatever class names the model's output tensor
declares (COCO's `person` here, since this particular model is a general
object detector rather than a dedicated face detector - it still finds every
face in the grid). A model with a genuine `face` class would label them that
way instead. Whenever at least one detection clears the confidence threshold,
an annotated copy of the image is saved too, with boxes and labels drawn on it.
