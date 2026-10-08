"""Shared helper for a ``Models/<Name>/download_model.py`` script.

Not used by the inference examples themselves, only by the per-model
download scripts that fetch the (large, web-sourced) files their ``model.json``
points at. Each such script declares its own list of ``(url, local filename,
expected sha256)`` and calls :func:`fetch` or :func:`fetch_zip_member` for each.
"""

from __future__ import annotations

import hashlib
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

__all__ = ["ChecksumError", "fetch", "fetch_zip_member"]


class ChecksumError(Exception):
    """A downloaded file's SHA-256 didn't match what the download script expected."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    print(f"  downloading {url}")
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
                        print(f"\r    {percent:3d}% ({written // 1_000_000} / {total // 1_000_000} MB)", end="", flush=True)
                    else:
                        print(f"\r    {written // 1_000_000} MB", end="", flush=True)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"Download failed: {exc} ({url})") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Download failed: {exc.reason} ({url})") from exc
    print()


def fetch(url: str, destination: Path, *, sha256: str | None = None, force: bool = False) -> None:
    """Download ``url`` to ``destination``, skipping it if already present and correct.

    Args:
        url: Direct URL to the file.
        destination: Where to save it.
        sha256: Expected hash of the file content. When given, an existing
            file is re-downloaded if it doesn't match, and a freshly
            downloaded one that still doesn't match raises :class:`ChecksumError`
            rather than being left on disk silently wrong.
        force: Re-download even if ``destination`` already exists and matches.
    """
    if destination.exists() and not force:
        if sha256 is None or _sha256(destination) == sha256:
            print(f"Already present: {destination}")
            return
        print(f"{destination.name} exists but doesn't match the expected checksum; re-downloading")

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {destination.name}")
    _download(url, destination)

    if sha256 is not None:
        actual = _sha256(destination)
        if actual != sha256:
            destination.unlink()
            raise ChecksumError(f"{destination.name}: expected sha256 {sha256}, got {actual} (download corrupted, or the source changed)")


def fetch_zip_member(zip_url: str, member: str, destination: Path, *, sha256: str | None = None, force: bool = False) -> None:
    """Download a ``.zip``/``.whl`` from ``zip_url`` and extract just ``member`` to ``destination``.

    The archive itself is downloaded to a temporary file and discarded; only
    the one member's bytes end up on disk. Skips the download entirely if
    ``destination`` already exists and matches ``sha256`` (or just exists, if
    no checksum was given).
    """
    if destination.exists() and not force:
        if sha256 is None or _sha256(destination) == sha256:
            print(f"Already present: {destination}")
            return
        print(f"{destination.name} exists but doesn't match the expected checksum; re-downloading")

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {destination.name} (member {member!r} of {zip_url})")
    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / "archive.zip"
        _download(zip_url, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            with archive.open(member) as source, open(destination, "wb") as out:
                out.write(source.read())

    if sha256 is not None:
        actual = _sha256(destination)
        if actual != sha256:
            destination.unlink()
            raise ChecksumError(f"{destination.name}: expected sha256 {sha256}, got {actual} (download corrupted, or the source changed)")
