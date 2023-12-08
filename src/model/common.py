from torch.nn import Module

from src.conf import ModelConfig
from src.model.example import ExampleModel


def get_model(model_cfg: ModelConfig) -> Module:
    if model_cfg.name in ["example", "example-model"]:
        return ExampleModel(**model_cfg.params)

    raise NotImplementedError(f"No model implemented for type '{model_cfg.name}'")
