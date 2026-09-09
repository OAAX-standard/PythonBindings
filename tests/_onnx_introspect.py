"""Best-effort fallback: read a model's input/output tensor specs straight out
of an ONNX file, for use when a runtime's own ``runtime_get_info()`` doesn't
describe them (that's optional per the OAAX interface, and plenty of
compliant runtimes report little more than a model count).

This lives in ``tests/``, not in the ``oaax`` package: it's a convenience for
writing test input data, not part of the interface, and it keeps the core
package free of an ``onnx`` dependency. It only helps when the model you
handed the runtime is itself an ``.onnx`` file; a vendor's compiled archive or
IR (``.zip``, ``.xml``/``.bin``, ...) can't be introspected this way.
"""

from __future__ import annotations

from pathlib import Path

from oaax import TensorElementType
from oaax.runtime import TensorSpec

__all__ = ["read_onnx_io"]

IoSpecs = tuple[list[TensorSpec], list[TensorSpec]]


def read_onnx_io(path: Path) -> tuple[IoSpecs | None, str | None]:
    """Try to read ``(inputs, outputs)`` from an ONNX file.

    Returns ``(specs, None)`` on success or ``(None, reason)`` on failure,
    where ``reason`` is a short, user-facing explanation of why this fallback
    couldn't help - not a file-not-onnx or missing-dependency error, just
    information for whoever has to decide what to do next (typically: pass
    ``--input``/``--output`` explicitly).
    """
    if path.suffix.lower() != ".onnx":
        return None, f"{path.name} is not an .onnx file, so it can't be introspected"

    try:
        import onnx
    except ImportError:
        return None, (
            f"{path.name} is an .onnx file, but the 'onnx' package isn't installed "
            "(pip install -e '.[onnx]', or pip install onnx, to enable this fallback)"
        )

    try:
        model = onnx.load(str(path))
        # Graph inputs with an initializer have a default and aren't required
        # at runtime; interface.h notes tensor element types are borrowed
        # directly from the ONNX DataType enum, so onnx's elem_type ints need
        # no translation at all.
        initializer_names = {i.name for i in model.graph.initializer}

        def specs(value_infos) -> list[TensorSpec]:
            result = []
            for value_info in value_infos:
                if value_info.name in initializer_names:
                    continue
                tensor_type = value_info.type.tensor_type
                element_type = TensorElementType(tensor_type.elem_type)
                shape = tuple(
                    dim.dim_value if dim.HasField("dim_value") else -1  # symbolic/dynamic dimension
                    for dim in tensor_type.shape.dim
                )
                result.append(TensorSpec(value_info.name, element_type, shape))
            return result

        return (specs(model.graph.input), specs(model.graph.output)), None
    except Exception as exc:  # malformed/unreadable ONNX file, unexpected protobuf shape, ...
        return None, f"failed to read tensor info from {path.name}: {exc}"
