"""Environment sanity check. Run after setup: python scripts/check_env.py"""

from __future__ import annotations

import platform
import sys

import numpy as np
import pandas as pd

from nidsrl.config import load_config
from nidsrl.utils.device import resolve_device
from nidsrl.utils.seed import seed_everything


def main() -> int:
    cfg = load_config("configs/default.yaml")
    print(f"python   : {sys.version.split()[0]} ({platform.platform()})")
    print(f"numpy    : {np.__version__}")
    print(f"pandas   : {pd.__version__}")

    try:
        import torch

        print(f"torch    : {torch.__version__}")
        print(f"cuda     : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"gpu      : {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    except ImportError:
        print("torch    : NOT INSTALLED (see README step 3)")

    print(f"device   : {resolve_device(cfg.project.device)}")

    first = seed_everything(cfg.project.seed).random(3)
    second = seed_everything(cfg.project.seed).random(3)
    ok = bool(np.array_equal(first, second))
    print(f"seeding  : {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
