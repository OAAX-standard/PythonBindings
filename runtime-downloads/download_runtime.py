#!/usr/bin/env python
"""Download the OAAX CPU reference runtime for this machine.

Picks the prebuilt archive matching your OS and CPU architecture, downloads
it, and unpacks it into runtime-downloads/CPU/ (next to this script):

    python download_runtime.py                              # auto-detect this machine
    python download_runtime.py --os linux --arch aarch64     # fetch a specific build instead
    python download_runtime.py --force                       # re-download even if already present

Supported combinations (this is the generic ONNX Runtime-backed CPU runtime,
not a vendor-specific one like Intel's):

    linux   x86_64
    linux   aarch64
    windows x86_64

The result is ready to pass straight to Runtime(...) or pytest --library:
runtime-downloads/CPU/libRuntimeLibrary.so on Linux,
runtime-downloads/CPU/RuntimeLibrary.dll on Windows.
"""

from __future__ import annotations

import argparse
import platform
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET_DIR = HERE / "CPU"
BASE_URL = "https://oaax.nbg1.your-objectstorage.com/runtimes/oaax-v2/CPU"

# (os, arch) -> (URL path segment, the main library file to expect after extraction)
ARCHIVES: dict[tuple[str, str], tuple[str, str]] = {
    ("linux", "x86_64"): ("x86_64/Ubuntu", "libRuntimeLibrary.so"),
    ("linux", "aarch64"): ("aarch64/Ubuntu", "libRuntimeLibrary.so"),
    ("windows", "x86_64"): ("x86_64/Windows", "RuntimeLibrary.dll"),
}

# platform.machine() spells these differently across OSes and tools.
ARCH_ALIASES = {
    "amd64": "x86_64",
    "x86_64": "x86_64",
    "aarch64": "aarch64",
    "arm64": "aarch64",
}


def detect_os() -> str:
    system = platform.system().lower()
    if system in ("linux", "windows"):
        return system
    supported = sorted({os_name for os_name, _ in ARCHIVES})
    raise SystemExit(f"Unsupported OS: {platform.system()!r}. Supported: {supported}. Pass --os/--arch to fetch a build for a different machine.")


def detect_arch() -> str:
    machine = platform.machine().lower()
    arch = ARCH_ALIASES.get(machine)
    if arch is None:
        supported = sorted({arch for _, arch in ARCHIVES})
        raise SystemExit(f"Unsupported CPU architecture: {machine!r}. Supported: {supported}.")
    return arch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--os", choices=sorted({os_name for os_name, _ in ARCHIVES}), help="override the detected OS")
    parser.add_argument("--arch", choices=sorted({arch for _, arch in ARCHIVES}), help="override the detected CPU architecture")
    parser.add_argument("--force", action="store_true", help="re-download and overwrite even if already present")
    return parser.parse_args()


def download(url: str, destination: Path) -> None:
    """Stream ``url`` to ``destination``, printing a simple progress indicator."""
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0))
            written = 0
            with open(destination, "wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
                    written += len(chunk)
                    if total:
                        percent = written * 100 // total
                        print(f"\r  downloading... {percent:3d}% ({written // 1_000_000} / {total // 1_000_000} MB)", end="", flush=True)
                    else:
                        print(f"\r  downloading... {written // 1_000_000} MB", end="", flush=True)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"Download failed: {exc} ({url})") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Download failed: {exc.reason} ({url})") from exc
    print()


def extract(archive_path: Path, target_dir: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as tar:
        try:
            tar.extractall(target_dir, filter="data")  # Python 3.12+: refuse unsafe members
        except TypeError:
            tar.extractall(target_dir)  # older Python without the `filter` parameter


def main() -> None:
    args = parse_args()
    os_name = args.os or detect_os()
    arch = args.arch or detect_arch()

    key = (os_name, arch)
    if key not in ARCHIVES:
        supported = ", ".join(f"{o}/{a}" for o, a in ARCHIVES)
        raise SystemExit(f"No prebuilt runtime for {os_name}/{arch}. Supported combinations: {supported}.")
    url_segment, library_name = ARCHIVES[key]

    library_path = TARGET_DIR / library_name
    if library_path.exists() and not args.force:
        print(f"Already present: {library_path} (pass --force to re-download)")
        return

    url = f"{BASE_URL}/{url_segment}/library.tar.gz"
    print(f"Detected {os_name}/{arch}; downloading {url}")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = TARGET_DIR / "library.tar.gz"
    download(url, archive_path)

    print(f"Extracting to {TARGET_DIR}")
    extract(archive_path, TARGET_DIR)
    archive_path.unlink()

    if not library_path.exists():
        raise SystemExit(f"Extracted the archive, but {library_path} is missing; its layout may have changed upstream.")
    print(f"Done: {library_path}")


if __name__ == "__main__":
    sys.exit(main())
