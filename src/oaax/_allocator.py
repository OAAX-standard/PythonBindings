"""Access to the C heap shared with the runtime library.

The OAAX interface transfers ownership of heap blocks across the library
boundary in both directions: the runtime ``free()``s the input tensors the
caller hands to ``runtime_enqueue_input``, and the caller must ``free()`` the
output tensors returned by ``runtime_retrieve_output``. Every such block must
therefore come from the C allocator, never from Python's own allocator
(``ctypes.create_string_buffer`` and friends are *not* suitable).

On Windows the runtime libraries link the Universal C Runtime, so we use
``ucrtbase.dll``. On other platforms we use the process-wide libc.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes.util import find_library

__all__ = ["malloc", "free", "strdup", "copy_to_heap"]


def _load_c_runtime() -> ctypes.CDLL:
    candidates: list[str | None]
    if sys.platform == "win32":
        candidates = ["ucrtbase", "msvcrt"]
    elif sys.platform == "darwin":
        candidates = ["libSystem.B.dylib", "libc.dylib", None]
    else:
        candidates = [None, "libc.so.6", find_library("c")]
    last_error: Exception | None = None
    for name in candidates:
        try:
            lib = ctypes.CDLL(name)
            lib.malloc  # noqa: B018 - probe that the symbol resolves
            lib.free  # noqa: B018
            return lib
        except (OSError, AttributeError) as exc:  # pragma: no cover - platform dependent
            last_error = exc
    raise OSError(f"Could not locate the C runtime allocator: {last_error}")


_crt = _load_c_runtime()
_crt.malloc.restype = ctypes.c_void_p
_crt.malloc.argtypes = [ctypes.c_size_t]
_crt.free.restype = None
_crt.free.argtypes = [ctypes.c_void_p]


def malloc(size: int) -> int:
    """Allocate ``size`` bytes on the C heap and return the address.

    ``size`` of zero is rounded up to one byte so the result is never NULL.
    """
    ptr = _crt.malloc(max(int(size), 1))
    if not ptr:
        raise MemoryError(f"malloc({size}) returned NULL")
    return ptr


def free(ptr: int | ctypes.c_void_p | None) -> None:
    """Release a block previously obtained from the C heap. NULL is ignored."""
    if isinstance(ptr, ctypes.c_void_p):
        ptr = ptr.value
    if ptr:
        _crt.free(ptr)


def copy_to_heap(source, size: int) -> int:
    """Copy ``size`` bytes from ``source`` (an address or ctypes object) to a new heap block."""
    ptr = malloc(size)
    if size:
        ctypes.memmove(ptr, source, size)
    return ptr


def strdup(text: str | bytes) -> int:
    """Copy a string to the C heap as a NUL-terminated UTF-8 buffer."""
    raw = text.encode("utf-8") if isinstance(text, str) else bytes(text)
    raw += b"\0"
    return copy_to_heap(raw, len(raw))
