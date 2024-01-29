from os.path import dirname, basename, isfile, join
import glob
from typing import Union, List, Tuple

EXCLUDED = ["__init__.py", "common.py", "module.py"]
modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = ["get_registered_models", *[basename(f)[:-3] for f in modules if isfile(f) and not basename(f) in EXCLUDED]]

RegisteredModelsDatabase = List[Tuple[List[str], type]]
_registered_models: RegisteredModelsDatabase = []


def register_model(namespace: Union[str, List[str]]):
    if not isinstance(namespace, str) and not isinstance(namespace, list):
        raise ValueError(f"You need to register a model with a namespace.")
    if isinstance(namespace, str):
        namespace = [namespace]

    def wrapper(model_cls: type):
        if not isinstance(model_cls, type):
            raise ValueError(f"Trying to register something other than a PyTorch model. Type: '{type(model_cls)}'")
        _registered_models.append((namespace, model_cls))
        return model_cls
    return wrapper


def get_registered_models() -> RegisteredModelsDatabase:
    return _registered_models
