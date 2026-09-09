"""Shared pytest configuration.

This test suite is runtime-agnostic: it has no built-in library or model, and
none is bundled in the repo. You must point it at both::

    pytest tests --library path/to/RuntimeLibrary.dll --model path/to/model.zip
    OAAX_LIBRARY=... OAAX_MODEL=... pytest tests

Runtime-specific init options (device, precision, logging, ...) are not part
of the OAAX interface, so pass them explicitly, one ``--config key=value`` per
option::

    pytest tests --config device_type=GPU --config log_level=1

Runtimes that do not describe their models in ``runtime_get_info()`` need the
input specs spelled out too, one ``--input name:dtype:d0,d1,...`` per tensor::

    pytest tests --input "image:float32:1,3,240,320"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Allow running the tests straight from a checkout without `pip install -e .`.
_SRC = REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("oaax")
    group.addoption("--library", default=os.environ.get("OAAX_LIBRARY"), help="path to an OAAX runtime shared library (required)")
    group.addoption("--model", default=os.environ.get("OAAX_MODEL"), help="path to a model file that library can load (required)")
    group.addoption(
        "--input",
        action="append",
        default=None,
        help="input spec 'name:dtype:d0,d1,...' (repeatable); overrides what runtime_get_info() reports",
    )
    group.addoption(
        "--config",
        action="append",
        default=None,
        help="runtime init option 'key=value' (repeatable); these are runtime-specific, not part of the OAAX interface",
    )


@pytest.fixture(scope="session")
def library_path(request: pytest.FixtureRequest) -> Path:
    raw = request.config.getoption("--library")
    if not raw:
        pytest.skip("no runtime library given: pass --library <path> or set OAAX_LIBRARY")
    path = Path(raw).expanduser()
    if not path.is_file():
        pytest.skip(f"runtime library not found: {path}")
    return path.resolve()


@pytest.fixture(scope="session")
def model_path(request: pytest.FixtureRequest) -> Path:
    raw = request.config.getoption("--model")
    if not raw:
        pytest.skip("no model given: pass --model <path> or set OAAX_MODEL")
    path = Path(raw).expanduser()
    if not path.is_file():
        pytest.skip(f"model file not found: {path}")
    return path.resolve()


@pytest.fixture(scope="session")
def explicit_input_specs(request: pytest.FixtureRequest) -> list[tuple[str, str, tuple[int, ...]]] | None:
    raw = request.config.getoption("--input")
    if not raw:
        env = os.environ.get("OAAX_INPUTS")
        raw = env.split(";") if env else None
    if not raw:
        return None
    specs = []
    for item in raw:
        try:
            name, dtype, dims = item.rsplit(":", 2)
        except ValueError:
            raise pytest.UsageError(f"--input expects 'name:dtype:d0,d1,...', got {item!r}") from None
        shape = tuple(int(d) for d in dims.split(",") if d.strip())
        specs.append((name, dtype, shape))
    return specs


@pytest.fixture(scope="session")
def runtime_config(request: pytest.FixtureRequest) -> dict[str, str]:
    """Runtime init options, entirely from ``--config``/``OAAX_CONFIG``. Empty by default.

    The OAAX interface does not define any config keys itself; a compliant
    runtime must ignore keys it does not recognise, so this stays empty unless
    you ask for something specific.
    """
    raw = request.config.getoption("--config")
    if not raw:
        env = os.environ.get("OAAX_CONFIG")
        raw = env.split(";") if env else None
    config: dict[str, str] = {}
    for item in raw or []:
        try:
            key, value = item.split("=", 1)
        except ValueError:
            raise pytest.UsageError(f"--config expects 'key=value', got {item!r}") from None
        config[key] = value
    return config
