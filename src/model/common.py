from torch.nn import Module

from src.conf import ModelConfig


def get_model(model_cfg: ModelConfig) -> Module:
    raise NotImplementedError(f"No model implemented for type '{model_cfg.type}'")
