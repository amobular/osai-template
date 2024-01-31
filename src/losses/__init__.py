from os.path import dirname, basename, isfile, join
import glob
from typing import Union, List, Tuple

EXCLUDED = ["__init__.py", "common.py"]
modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = ["get_registered_losses", *[basename(f)[:-3] for f in modules if isfile(f) and not basename(f) in EXCLUDED]]

RegisteredLossesDatabase = List[Tuple[List[str], type]]
_registered_losses: RegisteredLossesDatabase = []


def register_loss(namespace: Union[str, List[str]]):
    if not isinstance(namespace, str) and not isinstance(namespace, list):
        raise ValueError(f"You need to register a loss with a namespace.")
    if isinstance(namespace, str):
        namespace = [namespace]

    def wrapper(loss_cls: type):
        if not isinstance(loss_cls, type):
            raise ValueError(f"Trying to register something other than a PyTorch loss. Type: '{type(loss_cls)}'")
        _registered_losses.append((namespace, loss_cls))
        return loss_cls
    return wrapper


def get_registered_losses() -> RegisteredLossesDatabase:
    return _registered_losses
