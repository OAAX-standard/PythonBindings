# OAAX Python Bindings

Python bindings for shared libraries that implement the OAAX runtime interface
(`OAAX/Illustrative example/Runtime/include/interface.h`, OAAX 2.0). The
package loads any compliant runtime library with `ctypes` and lets you drive
it with NumPy arrays: initialise, load models, enqueue inputs, retrieve
outputs, and clean up. It is not tied to any particular vendor's runtime or
model format.

## Layout

| Path | Purpose |
| --- | --- |
| `src/oaax/` | The Python package (`import oaax`). |
| `tests/` | Pytest smoke test: load a library, feed random data, check the output. You supply the library and model. |
| `OAAX/` | The OAAX specification repository as a git submodule (branch `oaax-2.0`). |

The package lives under `src/` because Windows file systems are case
insensitive, so a top-level `oaax/` directory would collide with the `OAAX/`
submodule.

Nothing vendor-specific is checked in: no bundled runtime library, no bundled
model, no conversion tooling. Point the package and tests at whatever runtime
and model you're working with.

## Installation

```bash
pip install -e .          # package + numpy
pip install -e .[test]    # also pytest
```

Running from a checkout without installing also works: `tests/conftest.py`
adds `src/` to `sys.path`.

## Usage

```python
import numpy as np
from oaax import Runtime

with Runtime("path/to/RuntimeLibrary.dll") as rt:
    print(rt.name, rt.version)                      # available before init()
    rt.init({})                                      # pass runtime-specific options here, if any

    model_id = rt.load_model("path/to/model")
    inputs, outputs = rt.model_io(model_id)         # from runtime_get_info(), if reported
    print(inputs)                                   # e.g. [TensorSpec(name='image', element_type=FLOAT, shape=(1, 3, 240, 320))]

    image = np.random.rand(1, 3, 240, 320).astype(np.float32)
    result = rt.run(model_id, {"image": image}, timeout_ms=10_000)
    for tensor in result:
        print(tensor.name, tensor.data.shape, tensor.data.dtype)
```

Pipelined use mirrors the C interface directly:

```python
ids = [rt.enqueue(model_id, {"image": frame}) for frame in frames]   # non-blocking
while ids:
    result = rt.retrieve(timeout_ms=1000)          # any finished request, or None
    if result is not None:
        ids.remove(result.request_id)
```

`rt.wait_for(request_id)` blocks for one specific request and keeps results
for other requests aside for later `retrieve()` calls.

### API summary

| Python | C (`interface.h`) |
| --- | --- |
| `Runtime(path)` | loads the shared library, resolves the nine `runtime_*` symbols |
| `rt.name`, `rt.version` | `runtime_get_name`, `runtime_get_version` |
| `rt.init(config)` | `runtime_init(Config)` |
| `rt.load_models([...])`, `rt.load_model(...)` | `runtime_load_models` (paths, bytes, or `ModelConfig` with per-model options) |
| `rt.enqueue(model_id, inputs, request_id=None)` | `runtime_enqueue_input` |
| `rt.retrieve(timeout_ms)` | `runtime_retrieve_output` |
| `rt.info()`, `rt.raw_info()` | `runtime_get_info` (parsed JSON / raw string) |
| `rt.last_error()` | `runtime_get_error` |
| `rt.cleanup()` or leaving the `with` block | `runtime_cleanup` |

Inputs are a `{name: ndarray}` mapping, a `Tensor`, or a list of `Tensor`s.
Outputs come back as `InferenceResult(model_id, request_id, outputs)` where each
output is a `Tensor(name, data, element_type, shape)`. Element types with no
NumPy equivalent (BFLOAT16, FLOAT8, INT4, ...) are returned as raw `uint8`
bytes with `element_type` and `shape` filled in.

Any non-success status raises `oaax.RuntimeStatusError` carrying the
`RuntimeStatus` and the text of `runtime_get_error()`.

Configuration keys passed to `rt.init(...)` and per-model `config` are not
part of the OAAX interface itself; they're defined by each runtime vendor
(check that runtime's docs). A compliant runtime must ignore keys it doesn't
recognise.

### Memory ownership

The interface hands heap blocks across the library boundary: the runtime
`free()`s the inputs you enqueue, and you must `free()` the outputs it returns.
The bindings therefore copy input arrays into blocks allocated with the C
runtime's `malloc` (`ucrtbase.dll` on Windows, libc elsewhere) and release
outputs with the matching `free` after copying them into NumPy arrays. Your
own arrays are never handed to the runtime and stay valid.

## Tests

The suite takes no defaults: you must supply a runtime library and a model
file it can load.

```bash
pytest tests --library path/to/RuntimeLibrary.dll --model path/to/model
# or
OAAX_LIBRARY=path/to/RuntimeLibrary.dll OAAX_MODEL=path/to/model pytest tests
```

Tests skip with an explanation if `--library`/`--model` (or the matching
environment variables) aren't given, or point at a missing file.

The test reads the model's input names, types and shapes from
`runtime_get_info()`, which is optional per the interface: plenty of
compliant runtimes report little more than a model count. If the model you
handed the runtime is itself an `.onnx` file, the test falls back to reading
its input/output `value_info` directly (needs `pip install -e .[onnx]`).
Failing both, spell the inputs out yourself: `--input "name:dtype:d0,d1,..."`
(repeatable).

Pass runtime-specific init options with `--config key=value` (repeatable),
e.g. `--config device_type=GPU --config log_level=1`.

See `tests/conftest.py` for the full option list.
