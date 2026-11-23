import os
import sys
import pytest
from pathlib import Path

def test_python_version():
    """Verify Python version is 3.10 or higher."""
    assert sys.version_info >= (3, 10), f"Expected Python 3.10+, got {sys.version}"

def test_directory_structure():
    """Verify all mandatory project directories exist."""
    base_dir = Path(__file__).resolve().parent.parent
    expected_dirs = [
        "data/raw",
        "data/processed",
        "models",
        "notebooks",
        "src/classification",
        "src/nlp",
        "src/rag",
        "src/agents",
        "src/guardrails",
        "api",
        "tests",
        "uploads"
    ]
    for rel_path in expected_dirs:
        dir_path = base_dir / rel_path
        assert dir_path.is_dir(), f"Missing required directory: {rel_path}"

def test_core_dependencies():
    """Verify core numerical and ML libraries import correctly."""
    import numpy as np
    import pandas as pd
    import sklearn

    assert np.__version__ is not None
    assert pd.__version__ is not None
    assert sklearn.__version__ is not None

def test_pytorch_tensor_operations():
    """Verify PyTorch installation and basic tensor math on available device."""
    import torch

    assert torch.__version__ is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create tensors
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
    
    # Matrix multiplication
    c = torch.matmul(a, b)
    
    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]], device=device)
    assert torch.allclose(c, expected), f"Matrix multiplication failed: {c} != {expected}"
