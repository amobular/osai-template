from torch.nn import Module

from src.conf import ModelConfig

from . import *


def get_model(model_cfg: ModelConfig) -> Module:
    for registration in get_registered_models():
        namespace, init = registration
        if model_cfg.name in namespace:
            return init(**model_cfg.params)

    raise NotImplementedError(f"No model implemented for type '{model_cfg.name}'")
