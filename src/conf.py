from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DirConfig:
    data_dir: str
    processed_dir: str
    model_dir: str
    output_dir: str


@dataclass
class PrepareDataConfig:
    competition_name: str
    download: bool
    dir: DirConfig


@dataclass
class TrainConfig:
    dir: DirConfig
    dataset: DatasetConfig
    dataloader: DataloaderConfig
    module: ModuleConfig
    trainer: TrainerConfig


@dataclass
class DatasetConfig:
    pass


@dataclass
class DataloaderConfig:
    pass


@dataclass
class CallbacksConfig:
    pass


@dataclass
class ModuleConfig:
    model: ModelConfig
    loss: LossConfig


@dataclass
class ModelConfig:
    type: str


@dataclass
class LossConfig:
    type: str


@dataclass
class TrainerConfig:
    enable_progress_bar: bool
    max_epochs: int
    accelerator: str
    enable_checkpointing: bool
    check_val_every_n_epoch: int
    gradient_clip_val: float
    gradient_clip_algorithm: str
