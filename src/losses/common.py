from torch.nn import Module

from src.conf import LossConfig

from . import *


def get_loss(loss_cfg: LossConfig) -> Module:
    for registration in get_registered_losses():
        namespace, init = registration
        if loss_cfg.name in namespace:
            return init(**loss_cfg.params)

    raise NotImplementedError(f"No loss implemented for type '{loss_cfg.name}'")
