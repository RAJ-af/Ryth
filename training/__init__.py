"""Ryth Training Engine — pure-PyTorch training for the Ryth model core.

Modular: config, precision, gradient, optimizer, scheduler, loss, dataloader,
curriculum, metrics, evaluator, logger, checkpoint, callbacks, profiler, trainer.

Usage:
    from training import TrainConfig, Trainer
    Trainer(TrainConfig(data_dir="rds_out", model_preset="ryth_30m")).train()
"""

from .config import TrainConfig
from .trainer import Trainer

__all__ = ["TrainConfig", "Trainer"]
