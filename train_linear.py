"""Launch the linear experiment through the shared Hydra training entry point."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    command = [
        sys.executable,
        str(root / "train.py"),
        "--config-name=train_linear",
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(command, cwd=root))
