"""Enumerations mirroring ``interface.h`` (OAAX runtime interface 2.0)."""

from __future__ import annotations

from enum import IntEnum


class RuntimeStatus(IntEnum):
    """Return codes of every ``runtime_*`` function."""

    SUCCESS = 0
    ERROR = 1
    NOT_INITIALIZED = 2
    ALREADY_INITIALIZED = 3
    MODEL_NOT_LOADED = 4
    INVALID_ARGUMENT = 5
    INVALID_MODEL = 6
    FILE_NOT_FOUND = 7
    OUT_OF_MEMORY = 8
    INVALID_TENSOR = 9
    TENSOR_SHAPE_MISMATCH = 10
    TENSOR_TYPE_MISMATCH = 11
    NO_OUTPUT_AVAILABLE = 12
    INFERENCE_ERROR = 13
    NOT_IMPLEMENTED = 14
    TIMEOUT = 15
    DEVICE_ERROR = 16
    UNKNOWN_ERROR = 17
    INVALID_MODEL_ID = 18

    @classmethod
    def _missing_(cls, value: object) -> "RuntimeStatus | None":
        # A newer runtime may return a code this binding does not know yet.
        # Map it onto a pseudo-member so callers still get a RuntimeStatus.
        if isinstance(value, int):
            member = int.__new__(cls, value)
            member._name_ = f"UNKNOWN_STATUS_{value}"
            member._value_ = value
            return member
        return None


class TensorElementType(IntEnum):
    """Element type of a tensor; values follow the ONNX ``DataType`` enum."""

    UNDEFINED = 0
    FLOAT = 1
    UINT8 = 2
    INT8 = 3
    UINT16 = 4
    INT16 = 5
    INT32 = 6
    INT64 = 7
    STRING = 8
    BOOL = 9
    FLOAT16 = 10
    DOUBLE = 11
    UINT32 = 12
    UINT64 = 13
    COMPLEX64 = 14
    COMPLEX128 = 15
    BFLOAT16 = 16
    FLOAT8E4M3FN = 17
    FLOAT8E4M3FNUZ = 18
    FLOAT8E5M2 = 19
    FLOAT8E5M2FNUZ = 20
    UINT4 = 21
    INT4 = 22
    FLOAT4E2M1 = 23
    FLOAT8E8M0 = 24
    UINT2 = 25
    INT2 = 26

    @classmethod
    def _missing_(cls, value: object) -> "TensorElementType | None":
        if isinstance(value, int):
            member = int.__new__(cls, value)
            member._name_ = f"UNKNOWN_TYPE_{value}"
            member._value_ = value
            return member
        return None
