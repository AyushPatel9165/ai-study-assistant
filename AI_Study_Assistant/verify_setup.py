"""
Environment & PyTorch Verification Script
AI Study Assistant Project
"""

import os
import sys
import platform
from pathlib import Path

def print_separator(title=""):
    width = 65
    if title:
        print(f"\n{'=' * 5} {title} {'=' * (width - len(title) - 7)}")
    else:
        print("=" * width)

def verify_system():
    print_separator("1. System Information")
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Architecture: {platform.architecture()[0]} | Machine: {platform.machine()}")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version.split()[0]}")
    
    if sys.version_info < (3, 10):
        print("[WARNING] Python 3.10+ is recommended.")
        return False
    else:
        print("[OK] Python version meets requirements (>= 3.10).")
        return True

def verify_directories():
    print_separator("2. Directory Structure Verification")
    base_dir = Path(__file__).resolve().parent
    required_paths = [
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
    all_ok = True
    for p in required_paths:
        target = base_dir / p
        if target.is_dir():
            print(f"  [OK] Found directory: {p}")
        else:
            print(f"  [FAILED] Missing directory: {p}")
            all_ok = False
    return all_ok

def verify_packages():
    print_separator("3. Core Package Imports")
    packages = ["numpy", "pandas", "sklearn", "matplotlib", "seaborn", "pytest"]
    all_ok = True
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "installed")
            print(f"  [OK] {pkg:<15} : v{version}")
        except ImportError as e:
            print(f"  [FAILED] {pkg:<15} : Missing ({e})")
            all_ok = False
    return all_ok

def verify_pytorch():
    print_separator("4. PyTorch & Compute Device")
    try:
        import torch
        print(f"  [OK] PyTorch Version : {torch.__version__}")
        
        cuda_avail = torch.cuda.is_available()
        print(f"  [INFO] CUDA Available  : {cuda_avail}")
        
        if cuda_avail:
            print(f"  [INFO] GPU Device      : {torch.cuda.get_device_name(0)}")
            print(f"  [INFO] Memory Allocated: {torch.cuda.memory_allocated(0)/(1024**2):.1f} MB")
            device = torch.device("cuda")
        else:
            print("  [INFO] Compute Mode    : CPU (Optimized)")
            device = torch.device("cpu")
            
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        w = torch.tensor([[0.5, -0.5], [1.0, 2.0]], device=device)
        out = torch.matmul(x, w)
        print(f"  [OK] Tensor Math Test : Passed on {device.type.upper()}")
        return True
    except Exception as e:
        print(f"  [FAILED] PyTorch Verification Failed: {e}")
        return False

def main():
    print_separator("AI STUDY ASSISTANT - ENVIRONMENT VERIFICATION")
    s1 = verify_system()
    s2 = verify_directories()
    s3 = verify_packages()
    s4 = verify_pytorch()
    print_separator()
    if s1 and s2 and s3 and s4:
        print("[SUCCESS] ALL CHECKS PASSED. The environment is ready for Module 1.")
        sys.exit(0)
    else:
        print("[WARNING] Some checks failed. Please review the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
