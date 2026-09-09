"""ctypes declarations mirroring ``interface.h`` and the shared-library loader.

This module is the only place that knows the binary layout of the C structs
and the signatures of the nine ``runtime_*`` entry points. Everything else in
the package works through :func:`load_library`.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

from .exceptions import LibraryLoadError

__all__ = [
    "TensorDescriptor",
    "Tensors",
    "Config",
    "ModelConfig",
    "LoadedLibrary",
    "load_library",
]


class TensorDescriptor(ctypes.Structure):
    """``struct TensorDescriptor`` (40 bytes on 64-bit platforms).

    ``name``, ``shape`` and ``data`` are kept as raw addresses (``c_void_p``)
    rather than typed pointers so that ctypes never creates Python-owned
    copies behind our back; ownership of those blocks crosses the library
    boundary and they are always allocated with :mod:`oaax._allocator`.
    """

    _fields_ = [
        ("name", ctypes.c_void_p),  # char*  NUL-terminated tensor name
        ("data_type", ctypes.c_int),  # TensorElementType
        ("rank", ctypes.c_int),  # number of dimensions
        ("shape", ctypes.c_void_p),  # int*   `rank` dimension sizes
        ("data_size", ctypes.c_size_t),  # size of `data` in bytes
        ("data", ctypes.c_void_p),  # void*  tensor payload
    ]


class Tensors(ctypes.Structure):
    """``struct Tensors``: one inference request or response."""

    _fields_ = [
        ("id", ctypes.c_int),  # caller-assigned request id, echoed on output
        ("num_tensors", ctypes.c_int),
        ("tensors", ctypes.POINTER(TensorDescriptor)),
    ]


class Config(ctypes.Structure):
    """``struct Config``: key/value string pairs, caller-owned."""

    _fields_ = [
        ("length", ctypes.c_int),
        ("keys", ctypes.POINTER(ctypes.c_char_p)),
        ("values", ctypes.POINTER(ctypes.c_char_p)),
    ]


class ModelConfig(ctypes.Structure):
    """``struct ModelConfig``: where to find a model and how to configure it."""

    _fields_ = [
        ("file_path", ctypes.c_char_p),
        ("model_data", ctypes.POINTER(ctypes.c_ubyte)),
        ("model_size", ctypes.c_size_t),
        ("config", Config),
    ]


# (name, restype, argtypes) for every function exported per interface.h
_PROTOTYPES: tuple[tuple[str, object, list[object]], ...] = (
    ("runtime_init", ctypes.c_int, [Config]),
    ("runtime_load_models", ctypes.c_int, [ctypes.c_int, ctypes.POINTER(ModelConfig)]),
    ("runtime_enqueue_input", ctypes.c_int, [ctypes.c_int, ctypes.POINTER(Tensors)]),
    (
        "runtime_retrieve_output",
        ctypes.c_int,
        [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.POINTER(Tensors)), ctypes.c_int],
    ),
    ("runtime_cleanup", ctypes.c_int, []),
    ("runtime_get_error", ctypes.c_char_p, []),
    ("runtime_get_version", ctypes.c_char_p, []),
    ("runtime_get_name", ctypes.c_char_p, []),
    ("runtime_get_info", ctypes.c_char_p, []),
)


class LoadedLibrary:
    """A loaded OAAX runtime library with typed entry points.

    The C functions are exposed as attributes (``lib.runtime_init(...)``).
    The object also keeps the Windows DLL-directory handle alive so that the
    runtime's own dependencies (for example ``openvino.dll``) stay resolvable.
    """

    def __init__(self, path: Path, cdll: ctypes.CDLL, dll_directory_handle: object | None) -> None:
        self.path = path
        self._cdll = cdll
        self._dll_directory_handle = dll_directory_handle
        for name, restype, argtypes in _PROTOTYPES:
            try:
                fn = getattr(cdll, name)
            except AttributeError as exc:
                raise LibraryLoadError(f"{path} does not export required symbol '{name}'") from exc
            fn.restype = restype
            fn.argtypes = argtypes
            setattr(self, name, fn)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<LoadedLibrary {self.path}>"


def load_library(path: str | os.PathLike[str], *, add_dll_directory: bool = True) -> LoadedLibrary:
    """Load a shared library that implements ``interface.h``.

    Args:
        path: Path to the ``.dll`` / ``.so`` / ``.dylib``.
        add_dll_directory: On Windows, also register the library's directory
            as a DLL search path so its sibling dependencies resolve.

    Raises:
        LibraryLoadError: if the file cannot be loaded or lacks a symbol.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise LibraryLoadError(f"Runtime library not found: {resolved}")

    handle = None
    if sys.platform == "win32" and add_dll_directory:
        handle = os.add_dll_directory(str(resolved.parent))
    try:
        # CDLL (not PyDLL) releases the GIL around every call, so a blocking
        # runtime_retrieve_output() in one thread does not stall the others.
        cdll = ctypes.CDLL(str(resolved))
    except OSError as exc:
        if handle is not None:
            handle.close()
        raise LibraryLoadError(f"Could not load runtime library {resolved}: {exc}") from exc
    return LoadedLibrary(resolved, cdll, handle)
