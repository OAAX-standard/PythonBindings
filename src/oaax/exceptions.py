"""Exception types raised by the OAAX Python bindings."""

from __future__ import annotations

from .enums import RuntimeStatus


class OaaxError(Exception):
    """Base class for all errors raised by this package."""


class LibraryLoadError(OaaxError):
    """The shared library could not be loaded or lacks a required symbol."""


class RuntimeStatusError(OaaxError):
    """A ``runtime_*`` call returned a status other than ``SUCCESS``.

    Attributes:
        function: Name of the C function that failed.
        status: The returned :class:`~oaax.enums.RuntimeStatus`.
        detail: The text of ``runtime_get_error()`` at the time of failure,
            or ``None`` if the runtime reported no message.
    """

    def __init__(self, function: str, status: RuntimeStatus, detail: str | None) -> None:
        self.function = function
        self.status = status
        self.detail = detail
        message = f"{function} failed with {status.name} ({int(status)})"
        if detail:
            message += f": {detail}"
        super().__init__(message)
