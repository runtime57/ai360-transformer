import csv
from pathlib import Path


class ScalarHistory:
    def __init__(self, path, metric_names):
        self.path = Path(path)
        self.fields = list(
            dict.fromkeys(
                ["epoch", "step", "split", "scope", "elapsed_seconds", "learning_rate"]
                + list(metric_names)
            )
        )
        self.elapsed_offset = 0.0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open(encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                if reader.fieldnames != self.fields:
                    raise ValueError(f"Metric history columns changed: {self.path}")
                for row in reader:
                    self.elapsed_offset = max(self.elapsed_offset, float(row["elapsed_seconds"]))
        else:
            with self.path.open("w", encoding="utf-8", newline="") as file:
                csv.DictWriter(file, fieldnames=self.fields).writeheader()

    def write(self, *, epoch, step, split, scope, elapsed_seconds, metrics, learning_rate=None):
        row = dict(
            epoch=epoch,
            step=step,
            split=split,
            scope=scope,
            elapsed_seconds=self.elapsed_offset + elapsed_seconds,
            learning_rate=learning_rate,
            **metrics,
        )
        with self.path.open("a", encoding="utf-8", newline="") as file:
            csv.DictWriter(file, fieldnames=self.fields).writerow(row)
