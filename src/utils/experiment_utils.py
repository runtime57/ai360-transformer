"""Device selection and result files for standalone experiment tools."""

import csv
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch


def resolve_device(name):
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is unavailable in {sys.executable} (PyTorch {torch.__version__}). "
            "Select a CUDA-enabled environment or use --device cpu."
        )
    return device


def autocast_context(device, dtype):
    if dtype == "float32":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=getattr(torch, dtype))


def save_results(output, report):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = report["results"]
    if rows:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
