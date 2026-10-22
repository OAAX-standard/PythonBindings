# model-downloads

A gitignored scratch space for model files you're testing against, parallel
to `runtime-downloads/` for runtime libraries. Nothing in this repo reads
from here automatically; it's just a conventional place to put things like a
model you downloaded manually for `pytest --model ...` or one of the
`examples/` scripts' `--model`/model-directory argument, instead of scattering
them elsewhere in the working tree.

Anything you put here (other than this README and `.gitkeep`) stays
untracked, see `.gitignore`.
