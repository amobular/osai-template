from torch.nn import Module
from torch.nn.functional import l1_loss

from src.losses import register_loss


@register_loss(["example", "example-loss"])
class ExampleLoss(Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, y, weight=None):
        return l1_loss(x, y)
