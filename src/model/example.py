import torch
from torch.nn import Module, Parameter

from src.model import register_model


@register_model(["example", "example-model"])
class ExampleModel(Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param = Parameter(torch.rand(10))

    def forward(self, x):
        return x + self.param
