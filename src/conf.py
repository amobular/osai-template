from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


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
    callbacks: CallbacksConfig
    trainer: TrainerConfig


@dataclass
class DatasetConfig:
    params: Dict[str, Any]
    train_params: Dict[str, Any]
    val_params: Dict[str, Any]
    test_params: Dict[str, Any]


@dataclass
class DataloaderConfig:
    params: Dict[str, Any]
    train_params: Dict[str, Any]
    val_params: Dict[str, Any]
    test_params: Dict[str, Any]


@dataclass
class CallbacksConfig:
    pass


@dataclass
class ModuleConfig:
    learning_rate: float
    weight_decay: float
    warmup_percentage: float
    model: ModelConfig
    loss: LossConfig


@dataclass
class ModelConfig:
    name: str
    params: Dict[str, Any]


@dataclass
class LossConfig:
    name: str
    params: Dict[str, Any]


@dataclass
class TrainerConfig:
    enable_progress_bar: bool
    max_epochs: int
    accelerator: str
    enable_checkpointing: bool
    check_val_every_n_epoch: int
    gradient_clip_val: float
    gradient_clip_algorithm: str
