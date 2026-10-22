# Changelog

All notable changes to this project are documented here. Versions are listed newest first.

---

## 0.1.0 — 2026-10-22

Initial release.

### Added
- `oaax` package: ctypes bindings for the OAAX runtime interface (`interface.h`, OAAX 2.0) — a `Runtime` class (init, load models, enqueue/retrieve, cleanup), `Tensor`/`TensorSpec`, the `RuntimeStatus`/`TensorElementType` enums, and the C-heap-safe tensor marshalling the interface's ownership rules require. Not tied to any particular vendor's runtime or model format.
- `oaax.onnx_introspect`: reads a model's input/output tensor specs directly from an `.onnx` file, for use when a runtime's own `runtime_get_info()` doesn't report them.
- `tests/`: a runtime-and-model-agnostic pytest suite (`--library`/`--model`, `--config`, `--input`) that loads a library, feeds a model random data matching its declared inputs, and checks the output tensors.
- `examples/`: five end-to-end samples across vision, text and audio — `Faces` and `OCR` (vision), `Sentiment` and `Similarity` (text), `Speech` (audio, Whisper tiny.en). Each is self-contained: a `model.json` next to the model describes its tensors and preprocessing, and (for a model sourced from the web rather than authored for this repo) a `download_model.py` fetches it on demand instead of the weights being committed.
- `examples/download_all_models.py`: runs every example's `download_model.py` in one command.
- `runtime-downloads/download_runtime.py`: downloads the prebuilt generic CPU runtime for the current OS/architecture.
- `LICENSE` (Apache-2.0), `CONTRIBUTING.md`, `CHANGELOG.md`, `VERSION`.

### Changed
- Top-level directories renamed to kebab-case (`Examples` -> `examples`, `RuntimeDownloads` -> `runtime-downloads`, `ModelDownloads` -> `model-downloads`), matching the naming convention of the other OAAX-standard repositories.
- `pyproject.toml`'s `version` and `oaax.__version__` are now both derived from `VERSION` at the repo root, rather than being hand-kept in sync across files.
