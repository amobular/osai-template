import torch
import transformers
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from src.conf import ModuleConfig
from src.losses.common import get_loss
from src.model.common import get_model
from src.util.weight_decay import set_weight_decay


def get_module(module_cfg: ModuleConfig) -> LightningModule:
    return GeneralModule(module_cfg)


class GeneralModule(LightningModule):
    def __init__(self, cfg: ModuleConfig):
        super().__init__()
        self.model = get_model(cfg.model)
        self.loss = get_loss(cfg.loss)
        self.learning_rate = cfg.learning_rate
        self.weight_decay = cfg.weight_decay
        self.warmup_percentage = cfg.warmup_percentage

    def num_steps(self) -> int:
        """Get number of steps"""
        # Accessing _data_source is flaky and might break
        dataset = self.trainer.fit_loop._data_source.dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, self.trainer.num_devices)
        num_steps = dataset_size * self.trainer.max_epochs // (self.trainer.accumulate_grad_batches * num_devices)
        return num_steps

    def forward(self, x):
        return self.model(x)

    def calculate_loss(self, x_hat, y, weight=None):
        return self.loss(x_hat, y, weight)

    def log_metrics(self, x_hat, y, loss, step_type):
        self.log(f"{step_type}_loss", loss)

    def shared_step(self, batch, step_type: str = "train"):
        x = batch["x"]
        y = batch["y"]
        x_hat = self.forward(x)
        weight = batch["weight"] if "weight" in batch else None
        loss = self.calculate_loss(x_hat, y, weight)
        self.log_metrics(x_hat, y, loss, step_type=step_type)
        return {"loss": loss, "x_hat": x_hat}

    def training_step(self, batch):
        return self.shared_step(batch, step_type="train")

    def validation_step(self, batch):
        return self.shared_step(batch, step_type="val")

    def test_step(self, batch):
        return self.shared_step(batch, step_type="test")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        param_groups = set_weight_decay(
            self.model,
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            norm_weight_decay=0.0,
            custom_keys_weight_decay=[("bias", 0.0), ("embedding.weight", 0.0)],
        )
        optimizer = torch.optim.AdamW(param_groups, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.warmup_percentage * self.num_steps(),
            num_training_steps=self.num_steps()
        )
        lr_scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]
