from typing import List

import hydra
import pytorch_lightning as L
from pytorch_lightning import Callback

from src.conf import TrainConfig, CallbacksConfig
from src.data.common import get_data_module
from src.model.module import get_module


def get_callbacks(cfg: CallbacksConfig) -> List[Callback]:
    raise NotImplementedError()


def train(cfg: TrainConfig):
    data_module = get_data_module(dataset_cfg=cfg.dataset, dataloader_cfg=cfg.dataloader)
    module = get_module(module_cfg=cfg.module)

    trainer = L.Trainer(
        callbacks=get_callbacks(),
        **cfg.trainer,
    )

    trainer.fit(module, datamodule=data_module)


@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def main(cfg: TrainConfig):
    train(cfg)


if __name__ == '__main__':
    main()
