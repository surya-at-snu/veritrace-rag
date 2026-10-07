import os

# torch and lightgbm both bring their own openmp on windows, this stops them crashing each other
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# optional: cap torch threads when two jobs run side by side
if os.environ.get("VT_THREADS"):
    try:
        import torch
        torch.set_num_threads(int(os.environ["VT_THREADS"]))
    except Exception:
        pass

__version__ = "1.0.0"
