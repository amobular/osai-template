from pytorch_lightning import LightningDataModule

from src.conf import DatasetConfig, DataloaderConfig


def get_data_module(dataset_cfg: DatasetConfig, dataloader_cfg: DataloaderConfig) -> LightningDataModule:
    raise NotImplementedError()
