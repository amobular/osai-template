from torch.nn import Module

from src.conf import LossConfig
from src.losses.example import ExampleLoss


def get_loss(loss_cfg: LossConfig) -> Module:
    if loss_cfg.name in ["example", "example-model"]:
        return ExampleLoss(**loss_cfg.params)

    raise NotImplementedError(f"No loss implemented for type '{loss_cfg.name}'")
