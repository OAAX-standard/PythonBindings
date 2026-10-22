"""Python bindings for OAAX runtime libraries (``interface.h``, OAAX 2.0).

Load any shared library that implements the OAAX runtime interface and drive it
from Python with NumPy arrays::

    import numpy as np
    from oaax import Runtime

    with Runtime("RuntimeLibrary.dll") as rt:
        rt.init({"device_type": "CPU", "log_level": "3"})
        model_id = rt.load_model("model.zip")
        result = rt.run(model_id, {"image": np.random.rand(1, 3, 240, 320).astype(np.float32)})
        for tensor in result:
            print(tensor.name, tensor.data.shape, tensor.data.dtype)
"""

from importlib.metadata import PackageNotFoundError, version as _version
from pathlib import Path

from .enums import RuntimeStatus, TensorElementType
from .exceptions import LibraryLoadError, OaaxError, RuntimeStatusError
from .runtime import InferenceResult, ModelConfig, Runtime, TensorSpec, parse_dtype_name
from .tensors import Tensor, dtype_to_element_type, element_type_to_dtype

try:
    # The normal case: oaax is installed (editable or not), so setuptools has
    # already resolved VERSION into the package's own metadata.
    __version__ = _version("oaax")
except PackageNotFoundError:
    # Running from a raw checkout without `pip install -e .`: read the repo's
    # VERSION file directly. VERSION is the single source of truth either way.
    __version__ = (Path(__file__).resolve().parents[2] / "VERSION").read_text().strip()

__all__ = [
    "Runtime",
    "ModelConfig",
    "InferenceResult",
    "Tensor",
    "TensorSpec",
    "RuntimeStatus",
    "TensorElementType",
    "OaaxError",
    "LibraryLoadError",
    "RuntimeStatusError",
    "dtype_to_element_type",
    "element_type_to_dtype",
    "parse_dtype_name",
    "__version__",
]
