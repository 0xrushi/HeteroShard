"""
Training components for heterogeneous distributed training.
"""

from .coordinator import HeterogeneousTrainer
from .worker import RemoteWorker

__all__ = ['HeterogeneousTrainer', 'RemoteWorker']

