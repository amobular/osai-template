from pytorch_lightning import LightningModule

from src.conf import ModuleConfig
from src.losses.common import get_loss
from src.model.common import get_model


def get_module(module_cfg: ModuleConfig) -> LightningModule:
    return GeneralModule(module_cfg)


class GeneralModule(LightningModule):
    def __init__(self, cfg: ModuleConfig):
        self.model = get_model(cfg.model)
        self.loss = get_loss(cfg.loss)

    def forward(self, x):
        return self.model(x)

    def calculate_loss(self, x_hat, y, weight=None):
        return self.loss(x_hat, y, weight)

    def log_metrics(self, x_hat, y, loss, step_type):
        raise NotImplementedError()

    def shared_step(self, batch, step_type: str = "train"):
        x = batch["x"]
        y = batch["y"]
        x_hat = self.forward(x)
        weight = batch["weight"] if "weight" in batch else None
        loss = self.calculate_loss(x_hat, y, weight)
        self.log_metrics(x_hat, y, loss, step_type=step_type)
        return {"loss": loss, "x_hat": x_hat}

    def training_step(self, batch):
        self.shared_step(batch, step_type="train")

    def validation_step(self, batch):
        self.shared_step(batch, step_type="val")
