import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))

from src.trainer import Inferencer

original_run = Inferencer.run_inference


def capture_run(self):
    logs = original_run(self)
    result = {
        'metrics': logs,
        'target_tokens': self.evaluation_metrics.count('loss'),
        'partitions': {
            name: {
                'stories': len(loader.dataset.stories),
                'sequences': len(loader.dataset),
                'batches': len(loader),
            }
            for name, loader in self.evaluation_dataloaders.items()
        },
    }
    Path(os.environ['INFERENCE_RESULT_PATH']).write_text(json.dumps(result, indent=2) + '\n')
    return logs


Inferencer.run_inference = capture_run

import runpy

runpy.run_path(str(Path.cwd() / "inference.py"), run_name="__main__")
