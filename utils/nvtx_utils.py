"""Utilities for working with NVTX annotations."""
from contextlib import contextmanager

import torch

try:
    from torch.cuda import nvtx as torch_nvtx
except ImportError:  # pragma: no cover - NVTX is optional
    torch_nvtx = None


@contextmanager
def nvtx_range(label: str):
    """Context manager that creates an NVTX range when CUDA NVTX is available.

    Args:
        label: The label to apply to the NVTX range.
    """
    if torch_nvtx is not None and torch.cuda.is_available():
        with torch_nvtx.range(label):
            yield
    else:
        yield
