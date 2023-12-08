from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.conf import DatasetConfig, DataloaderConfig
from src.data.example import TrainingDataset, ValidationDataset, TestingDataset


class DataModule(LightningDataModule):
    def __init__(self, dataset_cfg: DatasetConfig, dataloader_cfg: DataloaderConfig):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.dataloader_cfg = dataloader_cfg

    def _setup_train_loader(self):
        self.train_dataset = TrainingDataset(**self.dataset_cfg.params, **self.dataset_cfg.train_params)

    def _setup_val_loader(self):
        self.val_dataset = ValidationDataset(**self.dataset_cfg.params, **self.dataset_cfg.val_params)

    def _setup_test_loader(self):
        self.test_dataset = TestingDataset(**self.dataset_cfg.params, **self.dataset_cfg.test_params)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self._setup_train_loader()
            self._setup_val_loader()
        elif stage == "validation":
            self._setup_val_loader()
        elif stage == "test":
            self._setup_test_loader()
        else:
            raise NotImplementedError(f"Unknown how to handle setup for data stage '{stage}'.")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.dataloader_cfg.params, **self.dataloader_cfg.train_params)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.dataloader_cfg.params, **self.dataloader_cfg.val_params)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.dataloader_cfg.params, **self.dataloader_cfg.test_params)
