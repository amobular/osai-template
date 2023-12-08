import torch
from torch.utils.data import Dataset


class TrainingDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __len__(self) -> int:
        return 1000

    def __getitem__(self, idx: int):
        return {
            "x": torch.rand(10),
            "y": torch.rand(10),
        }


class ValidationDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __len__(self) -> int:
        return 200

    def __getitem__(self, idx: int):
        return {
            "x": torch.rand(10),
            "y": torch.rand(10),
        }


class TestingDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __len__(self) -> int:
        return 200

    def __getitem__(self, idx: int):
        return {
            "x": torch.rand(10),
            "y": torch.rand(10),
        }
