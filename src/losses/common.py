from torch.nn import Module

from src.conf import LossConfig


def get_loss(loss_cfg: LossConfig) -> Module:
    raise NotImplementedError(f"No loss implemented for type '{loss_cfg.type}'")
