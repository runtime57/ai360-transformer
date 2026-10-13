import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.io_utils import ROOT_PATH


class Inferencer(BaseTrainer):
    def __init__(
        self,
        model,
        criterion,
        config,
        device,
        dataloaders,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
        datasets_context=None,
    ):
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = config.inferencer
        self.device = device
        self.model = model
        self.criterion = criterion
        self.batch_transforms = batch_transforms or {}
        self.evaluation_dataloaders = dict(dataloaders)
        self.metrics = (metrics.get("inference") or []) if metrics is not None else []
        self.evaluation_metrics = MetricTracker(
            "loss", *[metric.name for metric in self.metrics], writer=None
        )
        self.datasets_context = datasets_context or {}

        if not skip_model_load:
            self._from_pretrained(ROOT_PATH / config.inferencer.from_pretrained)

    def run_inference(self):
        return {
            part: self._inference_part(part, dataloader)
            for part, dataloader in self.evaluation_dataloaders.items()
        }

    def process_batch(self, batch, metrics):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)
        batch.update(self.model(**batch))

        token_count = (batch["target"] != self.criterion.ignore_class_id).sum().item()
        loss = self.criterion(**batch)
        for key, value in loss.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            metrics.update_avg(key, value, n=token_count)

        for metric in self.metrics:
            metric_result = metric(**batch)
            for key, value in metric_result.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                metrics.update_avg(metric.name, value, n=token_count)

        return batch

    def _inference_part(self, part, dataloader):
        self.is_train = False
        self.model.eval()
        self.criterion.eval()
        self.evaluation_metrics.reset()

        context = self.datasets_context.get(part)
        if context is not None:
            for component in [self.criterion, *self.metrics]:
                if hasattr(component, "_set_context"):
                    component._set_context(**context)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=part, total=len(dataloader), ncols=80):
                self.process_batch(batch, self.evaluation_metrics)

        if not self.evaluation_metrics.count("loss"):
            raise ValueError(f"No target tokens to evaluate in partition '{part}'")
        return self.evaluation_metrics.result()
