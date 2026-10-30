# oaax

Python bindings for runtime libraries that implement the OAAX runtime interface
(OAAX 2.0). The package loads any compliant runtime library with `ctypes` and
lets you drive it with NumPy arrays: initialise, load models, enqueue inputs,
retrieve outputs, and clean up. It is not tied to any particular vendor's
runtime or model format.

## Installation

```bash
pip install oaax
```

You need an OAAX runtime shared library (`.dll` on Windows, `.so` on Linux)
and a model file that runtime can load. The package doesn't ship either.

## Usage

```python
import numpy as np
from oaax import Runtime

with Runtime("path/to/RuntimeLibrary.dll") as rt:
    print(rt.name, rt.version)
    rt.init({})  # runtime-specific options, if any

    model_id = rt.load_model("path/to/model.onnx")
    io = rt.model_io(model_id)  # None if the runtime doesn't report its inputs and outputs
    if io is not None:
        inputs, outputs = io
        print(inputs)

    image = np.random.rand(1, 3, 240, 320).astype(np.float32)
    result = rt.run(model_id, {"input": image}, timeout_ms=10_000)  # the model's real input name
    for tensor in result:
        print(tensor.name, tensor.data.shape, tensor.data.dtype)
```

Use the input names and shapes your model expects. If you pass an input name
the model doesn't have, `run()` raises `ValueError` when the runtime reports
its inputs; otherwise the request may time out.

For pipelined use, `rt.enqueue(model_id, inputs)` submits a request without
waiting, and `rt.retrieve(timeout_ms)` returns whichever request finishes
first.

Configuration keys passed to `rt.init(...)` and per-model options are defined
by each runtime vendor, not by the OAAX interface. A compliant runtime ignores
keys it doesn't recognise.

## Documentation and source

- Full API notes, memory ownership rules and the test suite:
  [README](https://github.com/OAAX-standard/PythonBindings/blob/main/README.md)
- Examples with sample models: [examples/](https://github.com/OAAX-standard/PythonBindings/tree/main/examples)
- Contributing: [CONTRIBUTING.md](https://github.com/OAAX-standard/PythonBindings/blob/main/CONTRIBUTING.md)
- Changes between versions: [CHANGELOG.md](https://github.com/OAAX-standard/PythonBindings/blob/main/CHANGELOG.md)
- Issues: [github.com/OAAX-standard/PythonBindings/issues](https://github.com/OAAX-standard/PythonBindings/issues)

## License

Apache License 2.0. See [LICENSE](https://github.com/OAAX-standard/PythonBindings/blob/main/LICENSE).
