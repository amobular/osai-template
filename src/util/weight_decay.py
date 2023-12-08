import torch


def set_weight_decay(
        model: torch.nn.Module,
        weight_decay: float,
        learning_rate: float,
        norm_weight_decay=None,
        norm_classes=None,
        custom_keys_weight_decay=None,
):
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
        "_optim": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
        "_optim": 0.0,
    }
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for key in custom_keys:
                target_name = f"{prefix}.{name}" if prefix != "" and "." in key else name
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break

            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                elif hasattr(p, "_optim"):
                    params['_optim'].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            data = {"params": params[key], "weight_decay": params_weight_decay[key], "lr": learning_rate}
            if key == "_optim":
                data["lr"] = min([learning_rate, 0.001])
            param_groups.append(data)
    return param_groups
