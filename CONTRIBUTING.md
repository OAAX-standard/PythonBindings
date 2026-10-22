# Contributing to OAAX Python Bindings

Thank you for your interest in contributing! We welcome improvements to the
`oaax` package, the examples, the tests, and the documentation.

Before you start, please read the
[OAAX general CONTRIBUTING guide](https://github.com/OAAX-standard/OAAX/blob/main/CONTRIBUTING.md)
for commit message format, code style, and other project-wide guidelines.

## How to contribute

1. **Open an issue** to discuss your idea or bug report.
2. **Fork the repository** and create a branch for your changes.
3. **Make your changes**, add or update tests where relevant.
4. **Push** to your fork and **submit a pull request**.

Every commit must carry a `Signed-off-by` trailer (the Developer Certificate
of Origin the general guide asks for), with a name and email matching the
commit's own author. The easiest way is to pass `-s` to `git commit`:

```bash
git commit -s -m "..."
```

If you forgot on an already-made commit, `git commit --amend -s` (one commit)
or `git rebase --signoff <base>` (a whole branch) adds it after the fact, as
long as you haven't pushed yet.

---

## Development environment

```bash
pip install -e ".[test,onnx,examples,lint]"
```

This installs the package itself plus `pytest` (`test`), `onnx` for the
ONNX-introspection fallback (`onnx`), Pillow/`tokenizers` for the sample
scripts under `examples/` (`examples`), and `ruff` (`lint`). Running from a
checkout without installing also works: `tests/conftest.py` adds `src/` to
`sys.path`.

### Running the tests

The suite takes no defaults; point it at a runtime library and a model file:

```bash
pytest tests --library path/to/RuntimeLibrary.dll --model path/to/model
```

See the [Tests section of the README](README.md#tests) for the full set of
options (`--config`, `--input`, environment variable equivalents, ...).

### Linting

[`run-linter.yml`](.github/workflows/run-linter.yml) runs these same two
commands on every push and pull request; run them yourself first so CI isn't
the first place a problem shows up:

```bash
ruff check src/ tests/ examples/ runtime-downloads/
ruff format --check src/ tests/ examples/ runtime-downloads/   # drop --check to apply the formatting
```

Settings (line length, rule set, target Python version) live in
`pyproject.toml` under `[tool.ruff]`, not on the command line.

### Trying the examples

```bash
python runtime-downloads/download_runtime.py     # the generic CPU runtime, for your OS/arch
python examples/download_all_models.py           # every example's model files
python examples/Vision/Faces/detect_faces.py runtime-downloads/CPU/RuntimeLibrary.dll examples/Vision/Models/Face-Detection/model.onnx
```

Each example under `examples/` has its own README with the exact command for
that one, plus what its model expects and what its output means.

### Adding a new example

Follow the pattern an existing one already uses in its category (`vision_utils.py`,
`text_utils.py`, `audio_utils.py`): the example script stays generic, and
everything specific to one model lives in that model's `model.json`, read by
the shared helper. If the model's weights come from the web rather than being
authored for this repo, add a `download_model.py` next to its `model.json`
(see any existing one for the pattern) instead of committing the weights.

## License

By contributing to this project, you agree that your contributions will be
licensed under the [Apache License 2.0](LICENSE).
