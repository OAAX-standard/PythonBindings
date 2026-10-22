#!/usr/bin/env python
"""Run every examples/*/Models/<Name>/download_model.py.

Finds each one under this directory and runs it with the current Python
interpreter, so a single command fetches every example's model files.

    python download_all_models.py           # fetch anything missing, for every model
    python download_all_models.py --force    # re-download everything, for every model

Any argument other than --force is passed through to each download_model.py
unchanged.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT_NAME = "download_model.py"


def find_download_scripts() -> list[Path]:
    return sorted(HERE.rglob(SCRIPT_NAME))


def main() -> int:
    extra_args = sys.argv[1:]
    scripts = find_download_scripts()
    if not scripts:
        raise SystemExit(f"No {SCRIPT_NAME} found under {HERE}")

    print(f"Found {len(scripts)} model(s) to fetch:")
    for script in scripts:
        print(f"  {script.relative_to(HERE)}")
    print()

    failures = []
    for script in scripts:
        name = script.parent.name
        print(f"=== {name} ===")
        result = subprocess.run([sys.executable, str(script), *extra_args])
        if result.returncode != 0:
            failures.append(name)
        print()

    if failures:
        print(f"Failed: {', '.join(failures)}")
        return 1
    print(f"All {len(scripts)} model(s) fetched successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
