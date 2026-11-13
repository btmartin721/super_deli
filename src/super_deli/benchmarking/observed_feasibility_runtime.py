"""Keep scientific diagnostic execution separate from neural engineering fixtures."""

import sys
from contextlib import contextmanager
from importlib.abc import MetaPathFinder


class _NoTrainingLibraries(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in ("torch", "snpio"):
            raise ImportError("Observed-feature science must not import Torch or SNPio")
        return None


@contextmanager
def scientific_runtime():
    if any(n.split(".", 1)[0] in ("torch", "snpio") for n in sys.modules):
        raise ValueError("Run the benchmark in a fresh process without Torch/SNPio imports")
    finder = _NoTrainingLibraries()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
