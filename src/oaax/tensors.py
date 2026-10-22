"""Conversion between NumPy arrays and the C ``Tensors`` structures."""

from __future__ import annotations

import ctypes
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from . import _allocator
from ._native import TensorDescriptor, Tensors
from .enums import TensorElementType

__all__ = [
    "Tensor",
    "element_type_to_dtype",
    "dtype_to_element_type",
    "normalize_inputs",
    "build_input_tensors",
    "read_output_tensors",
    "free_tensors",
]

# Element types that have an exact NumPy equivalent.
_ELEMENT_TO_DTYPE: dict[TensorElementType, np.dtype] = {
    TensorElementType.FLOAT: np.dtype(np.float32),
    TensorElementType.UINT8: np.dtype(np.uint8),
    TensorElementType.INT8: np.dtype(np.int8),
    TensorElementType.UINT16: np.dtype(np.uint16),
    TensorElementType.INT16: np.dtype(np.int16),
    TensorElementType.INT32: np.dtype(np.int32),
    TensorElementType.INT64: np.dtype(np.int64),
    TensorElementType.BOOL: np.dtype(np.bool_),
    TensorElementType.FLOAT16: np.dtype(np.float16),
    TensorElementType.DOUBLE: np.dtype(np.float64),
    TensorElementType.UINT32: np.dtype(np.uint32),
    TensorElementType.UINT64: np.dtype(np.uint64),
    TensorElementType.COMPLEX64: np.dtype(np.complex64),
    TensorElementType.COMPLEX128: np.dtype(np.complex128),
}
_DTYPE_TO_ELEMENT: dict[np.dtype, TensorElementType] = {v: k for k, v in _ELEMENT_TO_DTYPE.items()}


def element_type_to_dtype(element_type: TensorElementType | int) -> np.dtype | None:
    """Return the NumPy dtype for ``element_type``, or ``None`` if NumPy has none."""
    return _ELEMENT_TO_DTYPE.get(TensorElementType(element_type))


def dtype_to_element_type(dtype) -> TensorElementType:
    """Return the OAAX element type for a NumPy dtype.

    Raises:
        TypeError: for dtypes with no OAAX equivalent (for example object or str).
    """
    dt = np.dtype(dtype)
    try:
        return _DTYPE_TO_ELEMENT[dt]
    except KeyError:
        raise TypeError(f"NumPy dtype {dt} has no OAAX TensorElementType equivalent") from None


@dataclass(init=False)
class Tensor:
    """A named tensor: what you send to the runtime and what you get back.

    Args:
        name: Tensor name as the model knows it.
        data: The payload. For element types NumPy can represent this is a
            typed array whose shape is the tensor's shape. For exotic types
            (BFLOAT16, FLOAT8*, INT4, ...) it is a flat ``uint8`` array holding
            the raw bytes.
        element_type: Overrides the type inferred from ``data.dtype``. Set this
            when ``data`` carries raw bytes of a type NumPy cannot express.
        shape: Overrides the shape inferred from ``data.shape``. Only needed
            together with raw-bytes ``data``.
    """

    name: str
    data: np.ndarray
    element_type: TensorElementType
    shape: tuple[int, ...]

    def __init__(
        self,
        name: str,
        data,
        element_type: TensorElementType | int | None = None,
        shape: Iterable[int] | None = None,
    ) -> None:
        self.name = str(name)
        self.data = np.asarray(data)
        if element_type is None:
            self.element_type = dtype_to_element_type(self.data.dtype)
        else:
            self.element_type = TensorElementType(element_type)
        self.shape = tuple(int(d) for d in (self.data.shape if shape is None else shape))

    @property
    def is_raw(self) -> bool:
        """True when ``data`` holds raw bytes rather than typed elements."""
        return element_type_to_dtype(self.element_type) != self.data.dtype

    def __repr__(self) -> str:
        return f"Tensor(name={self.name!r}, element_type={self.element_type.name}, shape={self.shape}, dtype={self.data.dtype})"


def normalize_inputs(inputs) -> list[Tensor]:
    """Accept a Tensor, an iterable of Tensors, or a ``{name: array}`` mapping."""
    if isinstance(inputs, Tensor):
        return [inputs]
    if isinstance(inputs, Mapping):
        return [value if isinstance(value, Tensor) else Tensor(name, value) for name, value in inputs.items()]
    tensors = list(inputs)
    for t in tensors:
        if not isinstance(t, Tensor):
            raise TypeError(f"Expected Tensor instances, got {type(t).__name__}")
    return tensors


def build_input_tensors(tensors: Iterable[Tensor], request_id: int):
    """Deep-copy ``tensors`` into a heap-allocated ``Tensors`` struct.

    Every block (the struct, the descriptor array, each name, shape and data
    buffer) is allocated on the C heap so the runtime can ``free()`` them once
    inference completes. If ``runtime_enqueue_input`` fails the caller keeps
    ownership and must call :func:`free_tensors`.

    Returns:
        A ``ctypes.POINTER(Tensors)`` ready to pass to ``runtime_enqueue_input``.
    """
    tensors = list(tensors)
    if not tensors:
        raise ValueError("At least one input tensor is required")

    struct_ptr = 0
    desc_ptr = 0
    try:
        struct_ptr = _allocator.malloc(ctypes.sizeof(Tensors))
        desc_bytes = ctypes.sizeof(TensorDescriptor) * len(tensors)
        desc_ptr = _allocator.malloc(desc_bytes)
        ctypes.memset(desc_ptr, 0, desc_bytes)
        descriptors = ctypes.cast(desc_ptr, ctypes.POINTER(TensorDescriptor))

        header = ctypes.cast(struct_ptr, ctypes.POINTER(Tensors)).contents
        header.id = int(request_id)
        header.num_tensors = len(tensors)
        header.tensors = descriptors

        for i, tensor in enumerate(tensors):
            array = np.ascontiguousarray(tensor.data)
            desc = descriptors[i]
            desc.data_type = int(tensor.element_type)
            desc.rank = len(tensor.shape)
            desc.data_size = array.nbytes
            desc.name = _allocator.strdup(tensor.name)
            shape_c = (ctypes.c_int * max(len(tensor.shape), 1))(*tensor.shape)
            desc.shape = _allocator.copy_to_heap(shape_c, ctypes.sizeof(ctypes.c_int) * len(tensor.shape))
            desc.data = _allocator.copy_to_heap(array.ctypes.data, array.nbytes)
        return ctypes.cast(struct_ptr, ctypes.POINTER(Tensors))
    except Exception:
        if struct_ptr and desc_ptr:
            free_tensors(ctypes.cast(struct_ptr, ctypes.POINTER(Tensors)))
        else:
            _allocator.free(desc_ptr)
            _allocator.free(struct_ptr)
        raise


def _descriptor_to_tensor(desc: TensorDescriptor) -> Tensor:
    name = ctypes.string_at(desc.name).decode("utf-8", errors="replace") if desc.name else ""
    rank = max(int(desc.rank), 0)
    if desc.shape and rank:
        shape = tuple(int(d) for d in ctypes.cast(desc.shape, ctypes.POINTER(ctypes.c_int))[:rank])
    else:
        shape = ()
    element_type = TensorElementType(desc.data_type)
    size = int(desc.data_size)
    # string_at copies the payload into Python-owned memory.
    raw = ctypes.string_at(desc.data, size) if (desc.data and size) else b""

    dtype = element_type_to_dtype(element_type)
    element_count = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if dtype is not None and size == element_count * dtype.itemsize:
        data = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
        return Tensor(name, data, element_type, shape)
    # Unknown element type or inconsistent size: hand back the raw bytes.
    return Tensor(name, np.frombuffer(raw, dtype=np.uint8).copy(), element_type, shape)


def read_output_tensors(ptr) -> tuple[int, list[Tensor]]:
    """Copy a runtime-owned ``Tensors*`` into Python objects and free it.

    Returns ``(request_id, tensors)``. The C memory is released with the C
    allocator's ``free()`` as the interface requires, whether or not the
    conversion succeeds.
    """
    try:
        header = ptr.contents
        request_id = int(header.id)
        count = max(int(header.num_tensors), 0)
        tensors = [_descriptor_to_tensor(header.tensors[i]) for i in range(count)] if header.tensors else []
        return request_id, tensors
    finally:
        free_tensors(ptr)


def free_tensors(ptr) -> None:
    """Deep-free a ``Tensors*`` (names, shapes, data, descriptor array, struct)."""
    if not ptr:
        return
    header = ptr.contents
    if header.tensors:
        for i in range(max(int(header.num_tensors), 0)):
            desc = header.tensors[i]
            _allocator.free(desc.name)
            _allocator.free(desc.shape)
            _allocator.free(desc.data)
        _allocator.free(ctypes.cast(header.tensors, ctypes.c_void_p))
    _allocator.free(ctypes.cast(ptr, ctypes.c_void_p))
