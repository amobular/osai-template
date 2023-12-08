from pytorch_lightning import LightningDataModule

from src.conf import DatasetConfig, DataloaderConfig
from src.data.module import DataModule


def get_data_module(dataset_cfg: DatasetConfig, dataloader_cfg: DataloaderConfig) -> LightningDataModule:
    return DataModule(dataset_cfg, dataloader_cfg)
