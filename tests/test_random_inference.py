"""Smoke test: load a runtime library, feed a model random data, get output back.

Point it at a runtime library and a model it can load::

    pytest tests --library path/to/libRuntimeLibrary.so --model path/to/model.zip

See ``conftest.py`` for the full set of options (``--config``, ``--input``).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from oaax import Runtime, RuntimeStatus, RuntimeStatusError, Tensor, TensorElementType, TensorSpec, element_type_to_dtype, parse_dtype_name
from oaax.onnx_introspect import model_io

TIMEOUT_MS = 60_000


def random_tensor(spec: TensorSpec, rng: np.random.Generator) -> Tensor:
    """Random data matching ``spec``; dynamic dimensions become 1."""
    shape = tuple(d if d > 0 else 1 for d in spec.shape)
    dtype = spec.dtype
    if dtype is None:
        raise pytest.UsageError(f"input {spec.name!r} has unsupported type {spec.element_type}; pass --input to override")
    if dtype.kind == "f":
        data = rng.random(shape, dtype=np.float64).astype(dtype)
    elif dtype.kind in "biu":
        # A tight 0/1 range, not a wide span clamped to the dtype's own limits:
        # plenty of integer inputs are an index into a small lookup table (a
        # token_type_ids embedding table can have as few as 2 rows), and this
        # agnostic test has no way to know any given input's real valid range.
        # An out-of-range index can make the runtime's own inference engine
        # fail outright - and, worse, some runtimes don't surface that as an
        # error status, so the request just never completes and this test
        # hangs for the full timeout instead of failing fast. 0 or 1 stays a
        # valid index into any table with at least one row, so it still
        # exercises the real plumbing (shapes, dtypes, request/response
        # matching) without risking a runtime-side failure this test has no
        # way to anticipate or safely tell apart from a genuine bug.
        data = rng.integers(0, 2, size=shape).astype(dtype)
    elif dtype.kind == "c":
        data = (rng.random(shape) + 1j * rng.random(shape)).astype(dtype)
    else:  # pragma: no cover - the dtype table only contains the kinds above
        raise TypeError(f"no random generator for dtype {dtype}")
    return Tensor(spec.name, data)


def resolve_input_specs(rt: Runtime, model_id: int, model_path: Path, explicit) -> list[TensorSpec]:
    if explicit:
        specs = []
        for name, dtype, shape in explicit:
            element_type = parse_dtype_name(dtype)
            if element_type is None:
                raise pytest.UsageError(f"unknown dtype {dtype!r} in --input for {name!r}")
            specs.append(TensorSpec(name, element_type, shape))
        return specs
    io, reason = model_io(rt, model_id, model_path)
    if io is None or not io[0]:
        detail = reason if io is None else "runtime_get_info() and the model file both report zero input tensors"
        pytest.fail(
            f"Could not determine the model's input tensors automatically: {detail}.\n"
            "Specify them yourself with --input \"name:dtype:d0,d1,...\" (repeatable), "
            "e.g. --input \"image:float32:1,3,224,224\".",
            pytrace=False,
        )
    return io[0]


@pytest.fixture(scope="module")
def runtime(library_path: Path, runtime_config: dict[str, str]):
    with Runtime(library_path) as rt:
        rt.init(runtime_config)
        yield rt


@pytest.fixture(scope="module")
def model_id(runtime: Runtime, model_path: Path) -> int:
    return runtime.load_model(model_path)


def test_library_reports_name_and_version(library_path: Path) -> None:
    rt = Runtime(library_path)  # name/version must work before init()
    assert rt.name.strip(), "runtime_get_name() returned an empty string"
    assert rt.version.strip(), "runtime_get_version() returned an empty string"
    assert rt.version[0].isdigit(), f"version {rt.version!r} does not look like SemVer"


def test_info_is_json_after_init(runtime: Runtime) -> None:
    info = runtime.info()
    assert isinstance(info, dict) and info, "runtime_get_info() should return a JSON object once initialized"


def test_loading_missing_model_raises(runtime: Runtime, tmp_path: Path) -> None:
    # Deliberately runs on the shared `runtime` fixture, before `model_id` has
    # loaded a real model (this function must stay above any test that depends
    # on `model_id`; pytest runs a module's tests in file order by default).
    # A `Runtime` library models a single process-wide instance: once a model
    # is loaded, some runtimes reject any further runtime_load_models() call
    # outright, regardless of whether the given path exists. Running this
    # check first isolates the "file not found" failure from that unrelated
    # "already have a model loaded" failure, and it doubles as a check of the
    # interface's documented guarantee that a failed load can be retried
    # without a runtime_init() cycle - the tests after this one rely on it.
    with pytest.raises(RuntimeStatusError) as excinfo:
        runtime.load_model(tmp_path / "does-not-exist.zip")
    assert excinfo.value.status is not RuntimeStatus.SUCCESS


def test_random_input_produces_output(runtime: Runtime, model_id: int, model_path: Path, explicit_input_specs) -> None:
    rng = np.random.default_rng(seed=1234)
    specs = resolve_input_specs(runtime, model_id, model_path, explicit_input_specs)
    inputs = [random_tensor(spec, rng) for spec in specs]

    result = runtime.run(model_id, inputs, timeout_ms=TIMEOUT_MS)

    assert result.model_id == model_id
    assert result.outputs, "runtime returned zero output tensors"

    io, reason = model_io(runtime, model_id, model_path)
    if io is None:
        warnings.warn(
            f"Could not determine the model's output tensors automatically ({reason}); "
            "skipping the output name/shape cross-check. This does not fail the test: "
            "only the input side is required to run inference.",
            stacklevel=1,
        )
    reported_outputs = {spec.name: spec for spec in (io[1] if io else [])}

    for tensor in result.outputs:
        assert tensor.name, "output tensor has no name"
        assert tensor.element_type is not TensorElementType.UNDEFINED
        expected_dtype = element_type_to_dtype(tensor.element_type)
        if expected_dtype is not None:
            assert tensor.data.dtype == expected_dtype
            assert tensor.data.shape == tensor.shape
            if expected_dtype.kind in "fc":
                assert np.isfinite(tensor.data).all(), f"output {tensor.name!r} contains NaN/inf"

        spec = reported_outputs.get(tensor.name)
        if spec is not None:
            # Static dims must match exactly; dynamic dims (-1) may be anything,
            # including 0: a detector fed random noise legitimately finds nothing.
            assert len(tensor.shape) == len(spec.shape), f"rank mismatch for {tensor.name!r}"
            for actual, expected in zip(tensor.shape, spec.shape):
                assert expected < 0 or actual == expected, f"shape {tensor.shape} != {spec.shape} for {tensor.name!r}"
            if all(d > 0 for d in spec.shape):
                assert tensor.data.nbytes > 0, f"output {tensor.name!r} is empty"

    if reported_outputs:
        assert sorted(result.names) == sorted(reported_outputs)


def test_request_ids_are_echoed_for_pipelined_requests(runtime: Runtime, model_id: int, model_path: Path, explicit_input_specs) -> None:
    rng = np.random.default_rng(seed=99)
    specs = resolve_input_specs(runtime, model_id, model_path, explicit_input_specs)
    submitted = [runtime.enqueue(model_id, [random_tensor(s, rng) for s in specs]) for _ in range(4)]

    received = []
    for _ in submitted:
        result = runtime.retrieve(timeout_ms=TIMEOUT_MS)
        assert result is not None, "timed out waiting for a pipelined result"
        assert result.outputs
        received.append(result.request_id)

    assert sorted(received) == sorted(submitted)
    assert runtime.retrieve(timeout_ms=0) is None, "no further output should be pending"
