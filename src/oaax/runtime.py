"""High-level, Pythonic wrapper around an OAAX runtime library."""

from __future__ import annotations

import ctypes
import itertools
import json
import os
import re
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import _native
from .enums import RuntimeStatus, TensorElementType
from .exceptions import RuntimeStatusError
from .tensors import Tensor, build_input_tensors, element_type_to_dtype, free_tensors, normalize_inputs, read_output_tensors

__all__ = ["ModelConfig", "InferenceResult", "TensorSpec", "Runtime"]


@dataclass
class ModelConfig:
    """Describes one model for :meth:`Runtime.load_models`.

    Provide either ``path`` (a file on disk) or ``data`` (the model bytes in
    memory). ``config`` holds runtime-specific key/value options for this model.
    """

    path: str | os.PathLike[str] | None = None
    data: bytes | bytearray | memoryview | None = None
    config: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.path is None) == (self.data is None):
            raise ValueError("ModelConfig needs exactly one of 'path' or 'data'")


@dataclass
class TensorSpec:
    """Name, type and shape of a model input or output, as reported by the runtime.

    ``element_type`` / ``dtype`` are ``None`` when the runtime used a type name
    this package does not recognise. Dynamic dimensions are reported as ``-1``.
    """

    name: str
    element_type: TensorElementType | None
    shape: tuple[int, ...]

    @property
    def dtype(self) -> np.dtype | None:
        return element_type_to_dtype(self.element_type) if self.element_type is not None else None


@dataclass
class InferenceResult:
    """Outputs of one inference request."""

    model_id: int
    request_id: int
    outputs: list[Tensor]

    def __getitem__(self, key: int | str) -> Tensor:
        if isinstance(key, int):
            return self.outputs[key]
        for tensor in self.outputs:
            if tensor.name == key:
                return tensor
        raise KeyError(key)

    def __iter__(self):
        return iter(self.outputs)

    def __len__(self) -> int:
        return len(self.outputs)

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.outputs]

    def as_dict(self) -> dict[str, np.ndarray]:
        """Outputs as ``{name: array}``."""
        return {t.name: t.data for t in self.outputs}


# Type names various runtimes may print in runtime_get_info(). Best effort.
_DTYPE_ALIASES: dict[str, TensorElementType] = {
    "float": TensorElementType.FLOAT,
    "float32": TensorElementType.FLOAT,
    "f32": TensorElementType.FLOAT,
    "fp32": TensorElementType.FLOAT,
    "double": TensorElementType.DOUBLE,
    "float64": TensorElementType.DOUBLE,
    "f64": TensorElementType.DOUBLE,
    "fp64": TensorElementType.DOUBLE,
    "float16": TensorElementType.FLOAT16,
    "half": TensorElementType.FLOAT16,
    "f16": TensorElementType.FLOAT16,
    "fp16": TensorElementType.FLOAT16,
    "bfloat16": TensorElementType.BFLOAT16,
    "bf16": TensorElementType.BFLOAT16,
    "uint8": TensorElementType.UINT8,
    "u8": TensorElementType.UINT8,
    "int8": TensorElementType.INT8,
    "i8": TensorElementType.INT8,
    "uint16": TensorElementType.UINT16,
    "u16": TensorElementType.UINT16,
    "int16": TensorElementType.INT16,
    "i16": TensorElementType.INT16,
    "uint32": TensorElementType.UINT32,
    "u32": TensorElementType.UINT32,
    "int32": TensorElementType.INT32,
    "i32": TensorElementType.INT32,
    "int": TensorElementType.INT32,
    "uint64": TensorElementType.UINT64,
    "u64": TensorElementType.UINT64,
    "int64": TensorElementType.INT64,
    "i64": TensorElementType.INT64,
    "bool": TensorElementType.BOOL,
    "boolean": TensorElementType.BOOL,
    "string": TensorElementType.STRING,
    "str": TensorElementType.STRING,
}


def parse_dtype_name(name: str | int | None) -> TensorElementType | None:
    """Map a runtime-reported type name (``"f32"``, ``"FLOAT"``, ``1`` ...) to an element type."""
    if name is None:
        return None
    if isinstance(name, int):
        return TensorElementType(name)
    key = str(name).strip().lower()
    key = re.sub(r"^(data_type_|tensor_)", "", key)
    if key in _DTYPE_ALIASES:
        return _DTYPE_ALIASES[key]
    try:
        return TensorElementType[key.upper()]
    except KeyError:
        return None


def _parse_shape(shape: Any) -> tuple[int, ...]:
    dims: list[int] = []
    for d in shape or ():
        try:
            value = int(d)
        except (TypeError, ValueError):
            value = -1  # symbolic / dynamic dimension such as "?" or "N"
        dims.append(value if value > 0 else -1)
    return tuple(dims)


def _make_config(options: Mapping[str, str] | None) -> tuple[_native.Config, list[object]]:
    """Build a ``Config`` struct; the returned list keeps its buffers alive."""
    items = [(str(k).encode("utf-8"), str(v).encode("utf-8")) for k, v in (options or {}).items()]
    if not items:
        return _native.Config(0, None, None), []
    keys = (ctypes.c_char_p * len(items))(*(k for k, _ in items))
    values = (ctypes.c_char_p * len(items))(*(v for _, v in items))
    return _native.Config(len(items), keys, values), [keys, values]


class Runtime:
    """One loaded OAAX runtime library.

    Typical use::

        with Runtime("RuntimeLibrary.dll") as rt:
            rt.init({"device_type": "CPU"})
            model_id = rt.load_model("model.zip")
            result = rt.run(model_id, {"image": np.zeros((1, 3, 240, 320), np.float32)})
            print(result[0].data.shape)

    The object is a context manager; leaving the ``with`` block calls
    :meth:`cleanup`. A runtime library holds process-global state, so create
    only one :class:`Runtime` per library within a process.
    """

    def __init__(self, library_path: str | os.PathLike[str], *, add_dll_directory: bool = True) -> None:
        self._lib = _native.load_library(library_path, add_dll_directory=add_dll_directory)
        self._initialized = False
        self._model_count = 0
        self._request_ids = itertools.count(1)
        self._lock = threading.Lock()
        # Results retrieved while waiting for a *different* request id.
        self._stash: dict[int, InferenceResult] = {}
        # model_io() results, cached per model_id so enqueue()'s input-name
        # check (see _validate_input_names) doesn't re-parse runtime_get_info()
        # on every call.
        self._model_io_cache: dict[int, tuple[list[TensorSpec], list[TensorSpec]] | None] = {}

    # ------------------------------------------------------------------ meta

    @property
    def library_path(self) -> Path:
        return self._lib.path

    @property
    def name(self) -> str:
        """``runtime_get_name()``; callable before :meth:`init`."""
        return (self._lib.runtime_get_name() or b"").decode("utf-8", errors="replace")

    @property
    def version(self) -> str:
        """``runtime_get_version()``; callable before :meth:`init`."""
        return (self._lib.runtime_get_version() or b"").decode("utf-8", errors="replace")

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def model_count(self) -> int:
        """Number of models loaded since :meth:`init`."""
        return self._model_count

    def last_error(self) -> str | None:
        """``runtime_get_error()``: the runtime's most recent error text, if any."""
        raw = self._lib.runtime_get_error()
        return raw.decode("utf-8", errors="replace") if raw else None

    def raw_info(self) -> str | None:
        """``runtime_get_info()`` as the unparsed JSON string (``None`` before init)."""
        raw = self._lib.runtime_get_info()
        return raw.decode("utf-8", errors="replace") if raw else None

    def info(self) -> dict[str, Any] | None:
        """``runtime_get_info()`` parsed into a dict. All keys are optional."""
        raw = self.raw_info()
        if raw is None:
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}
        return parsed if isinstance(parsed, dict) else {"raw": parsed}

    def model_io(self, model_id: int) -> tuple[list[TensorSpec], list[TensorSpec]] | None:
        """Input and output specs for ``model_id`` from :meth:`info`, if the runtime reports them.

        The interface leaves the contents of ``runtime_get_info()`` up to the
        runtime; this looks for a ``models`` list whose entries carry
        ``inputs`` / ``outputs`` lists of ``{name, dtype, shape}`` objects and
        returns ``None`` when that information is absent.
        """
        info = self.info() or {}
        models = info.get("models")
        if not isinstance(models, list) or not (0 <= model_id < len(models)):
            return None
        entry = models[model_id]
        if not isinstance(entry, dict):
            return None

        def specs(key: str) -> list[TensorSpec]:
            out = []
            for item in entry.get(key) or []:
                if not isinstance(item, dict):
                    continue
                dtype = item.get("dtype", item.get("type", item.get("data_type")))
                out.append(TensorSpec(str(item.get("name", "")), parse_dtype_name(dtype), _parse_shape(item.get("shape"))))
            return out

        inputs, outputs = specs("inputs"), specs("outputs")
        if not inputs and not outputs:
            return None
        return inputs, outputs

    # ------------------------------------------------------------- lifecycle

    def _check(self, function: str, code: int) -> RuntimeStatus:
        status = RuntimeStatus(code)
        if status is not RuntimeStatus.SUCCESS:
            raise RuntimeStatusError(function, status, self.last_error())
        return status

    def init(self, config: Mapping[str, str] | None = None) -> None:
        """``runtime_init()`` with optional key/value configuration.

        Keys are runtime-specific (for example ``device_type``, ``log_level``);
        unknown keys are ignored by a compliant runtime. Values must be strings.
        """
        struct, keepalive = _make_config(config)
        self._check("runtime_init", self._lib.runtime_init(struct))
        del keepalive
        self._initialized = True
        self._model_count = 0
        self._stash.clear()
        self._model_io_cache.clear()

    def load_models(self, models: Iterable[ModelConfig | str | os.PathLike[str] | bytes]) -> list[int]:
        """``runtime_load_models()`` for several models at once.

        Each item may be a :class:`ModelConfig`, a path, or raw model bytes.
        Returns the model ids, which are the indices into the given sequence
        (offset by the number of models loaded earlier).
        """
        configs = [self._coerce_model(m) for m in models]
        if not configs:
            raise ValueError("At least one model is required")

        array = (_native.ModelConfig * len(configs))()
        keepalive: list[object] = []
        for i, cfg in enumerate(configs):
            entry = array[i]
            if cfg.path is not None:
                path_bytes = os.fsencode(Path(cfg.path).expanduser())
                keepalive.append(path_bytes)
                entry.file_path = path_bytes
                entry.model_data = None
                entry.model_size = 0
            else:
                blob = (ctypes.c_ubyte * len(cfg.data)).from_buffer_copy(cfg.data)  # type: ignore[arg-type]
                keepalive.append(blob)
                entry.file_path = None
                entry.model_data = ctypes.cast(blob, ctypes.POINTER(ctypes.c_ubyte))
                entry.model_size = len(cfg.data)  # type: ignore[arg-type]
            entry.config, extra = _make_config(cfg.config)
            keepalive.extend(extra)

        self._check("runtime_load_models", self._lib.runtime_load_models(len(configs), array))
        del keepalive
        first = self._model_count
        self._model_count += len(configs)
        return list(range(first, first + len(configs)))

    def load_model(self, model: ModelConfig | str | os.PathLike[str] | bytes, config: Mapping[str, str] | None = None) -> int:
        """Load a single model and return its id."""
        cfg = self._coerce_model(model)
        if config:
            cfg = ModelConfig(path=cfg.path, data=cfg.data, config={**cfg.config, **config})
        return self.load_models([cfg])[0]

    @staticmethod
    def _coerce_model(model) -> ModelConfig:
        if isinstance(model, ModelConfig):
            return model
        if isinstance(model, (bytes, bytearray, memoryview)):
            return ModelConfig(data=model)
        return ModelConfig(path=model)

    def cleanup(self) -> None:
        """``runtime_cleanup()``: unload models and release runtime resources. Idempotent."""
        code = self._lib.runtime_cleanup()
        self._initialized = False
        self._model_count = 0
        self._stash.clear()
        self._model_io_cache.clear()
        self._check("runtime_cleanup", code)

    def __enter__(self) -> Runtime:
        return self

    def __exit__(self, *exc_info) -> None:
        self.cleanup()

    def __del__(self) -> None:  # best effort; explicit cleanup() is preferred
        try:
            if getattr(self, "_initialized", False):
                self._lib.runtime_cleanup()
        except Exception:
            pass

    # ------------------------------------------------------------- inference

    def _cached_model_io(self, model_id: int) -> tuple[list[TensorSpec], list[TensorSpec]] | None:
        """:meth:`model_io`, computed once per model and reused (cleared by :meth:`init`/:meth:`cleanup`)."""
        if model_id not in self._model_io_cache:
            self._model_io_cache[model_id] = self.model_io(model_id)
        return self._model_io_cache[model_id]

    def _validate_input_names(self, model_id: int, tensors: list[Tensor]) -> None:
        """Check ``tensors``' names against this model's own reported inputs, when it reports any.

        Reporting its inputs is optional for a runtime (most don't), so this
        silently does nothing when :meth:`model_io` is unavailable - there is
        nothing to check a name against. When it is available, catching a
        wrong or misspelled input name here, before it ever reaches the
        runtime, matters: the interface lets a runtime simply drop a request
        whose input names it doesn't recognise rather than reporting an
        error, which otherwise surfaces much later as a confusing timeout
        with no indication of the real cause (see :meth:`run`).
        """
        io = self._cached_model_io(model_id)
        if io is None or not io[0]:
            return
        expected_names = {spec.name for spec in io[0]}
        given_names = {tensor.name for tensor in tensors}
        if given_names == expected_names:
            return
        problems = []
        missing = expected_names - given_names
        if missing:
            problems.append(f"missing {sorted(missing)}")
        unexpected = given_names - expected_names
        if unexpected:
            problems.append(f"unexpected {sorted(unexpected)}")
        raise ValueError(
            f"model {model_id} expects inputs {sorted(expected_names)}, got {sorted(given_names)} ({'; '.join(problems)}). "
            "Sending the wrong name is accepted by the interface but silently dropped by some runtimes, producing "
            "a confusing timeout instead of a clear error - that's why this is caught here instead."
        )

    def enqueue(self, model_id: int, inputs, request_id: int | None = None) -> int:
        """``runtime_enqueue_input()``: submit inputs for asynchronous inference.

        Args:
            model_id: Id returned by :meth:`load_models`.
            inputs: A :class:`Tensor`, a sequence of Tensors, or a
                ``{name: numpy_array}`` mapping. Data is copied to the C heap;
                the caller keeps its arrays.
            request_id: Correlation id echoed back on the result. Auto-assigned
                (unique, increasing) when omitted.

        Returns:
            The request id.

        Raises:
            ValueError: if the runtime reports this model's expected input
                names (via ``runtime_get_info()``) and ``inputs`` doesn't
                name exactly those. Not every runtime reports this, so this
                check is best-effort, not a guarantee.
        """
        if request_id is None:
            request_id = next(self._request_ids)
        tensors = normalize_inputs(inputs)
        self._validate_input_names(model_id, tensors)
        ptr = build_input_tensors(tensors, request_id)
        code = self._lib.runtime_enqueue_input(int(model_id), ptr)
        status = RuntimeStatus(code)
        if status is not RuntimeStatus.SUCCESS:
            free_tensors(ptr)  # on failure ownership stays with us
            raise RuntimeStatusError("runtime_enqueue_input", status, self.last_error())
        return request_id

    def retrieve(self, timeout_ms: int = -1) -> InferenceResult | None:
        """``runtime_retrieve_output()``: fetch the next finished result from any model.

        Args:
            timeout_ms: ``0`` returns immediately, a negative value waits until
                a result is available, otherwise wait at most this long.

        Returns:
            The result, or ``None`` if nothing was ready within the timeout.
        """
        with self._lock:
            if self._stash:
                return self._stash.pop(next(iter(self._stash)))
        return self._retrieve_raw(timeout_ms)

    def _retrieve_raw(self, timeout_ms: int) -> InferenceResult | None:
        model_id = ctypes.c_int(-1)
        out = ctypes.POINTER(_native.Tensors)()
        code = self._lib.runtime_retrieve_output(ctypes.byref(model_id), ctypes.byref(out), int(timeout_ms))
        status = RuntimeStatus(code)
        if status in (RuntimeStatus.NO_OUTPUT_AVAILABLE, RuntimeStatus.TIMEOUT):
            if out:
                free_tensors(out)
            return None
        if status is not RuntimeStatus.SUCCESS:
            if out:
                free_tensors(out)
            raise RuntimeStatusError("runtime_retrieve_output", status, self.last_error())
        if not out:
            raise RuntimeStatusError("runtime_retrieve_output", RuntimeStatus.UNKNOWN_ERROR, "runtime returned SUCCESS with a NULL Tensors pointer")
        request_id, tensors = read_output_tensors(out)
        return InferenceResult(model_id.value, request_id, tensors)

    def wait_for(self, request_id: int, timeout_ms: int = -1) -> InferenceResult | None:
        """Retrieve the result for a specific request id.

        Results for other in-flight requests that arrive meanwhile are kept
        and handed out by later :meth:`retrieve` / :meth:`wait_for` calls.
        Returns ``None`` when the timeout elapses first.
        """
        deadline = None if timeout_ms < 0 else time.monotonic() + timeout_ms / 1000.0
        while True:
            with self._lock:
                if request_id in self._stash:
                    return self._stash.pop(request_id)
            if deadline is None:
                remaining = -1
            else:
                remaining = int(max(0.0, deadline - time.monotonic()) * 1000)
            result = self._retrieve_raw(remaining)
            if result is None:
                if deadline is not None and time.monotonic() >= deadline:
                    return None
                continue
            if result.request_id == request_id:
                return result
            with self._lock:
                self._stash[result.request_id] = result

    def run(self, model_id: int, inputs, timeout_ms: int = -1) -> InferenceResult:
        """Enqueue ``inputs`` and wait for that request's result.

        Raises:
            TimeoutError: when ``timeout_ms`` is non-negative and elapses first.
        """
        request_id = self.enqueue(model_id, inputs)
        result = self.wait_for(request_id, timeout_ms)
        if result is None:
            hint = (
                ""
                if self._cached_model_io(model_id) is not None
                else " (this runtime doesn't report this model's expected inputs, so enqueue() couldn't check "
                "them ahead of time: if this model doesn't actually need this long, a wrong input name silently "
                "dropped by the runtime is a common cause - double-check the names against the model itself)"
            )
            raise TimeoutError(f"No output for request {request_id} within {timeout_ms} ms{hint}")
        return result

    def __repr__(self) -> str:
        state = "initialized" if self._initialized else "not initialized"
        return f"<Runtime {self.name!r} v{self.version} ({state}) from {self._lib.path}>"
