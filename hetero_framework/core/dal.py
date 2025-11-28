"""
Device Abstraction Layer (DAL)

Provides a unified interface for different hardware accelerators (CUDA, MPS, CPU).
"""

from typing import Literal

import torch

DeviceType = Literal["cuda", "mps", "cpu", "auto"]


class DeviceAbstractionLayer:
    """Abstraction layer for different compute devices."""

    def __init__(self, device: DeviceType = "auto"):
        """
        Initialize the Device Abstraction Layer.

        Args:
            device: Target device type. Use "auto" for automatic detection.
        """
        self.device = self._resolve_device(device)
        self.device_name = self._get_device_name()

    def _resolve_device(self, device: DeviceType) -> torch.device:
        """Resolve the actual device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)

    def _get_device_name(self) -> str:
        """Get a human-readable device name."""
        if self.device.type == "cuda":
            name = torch.cuda.get_device_name(0)
            return f"CUDA ({name})"
        elif self.device.type == "mps":
            return "Apple MPS"
        else:
            return "CPU"

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to the managed device."""
        return tensor.to(self.device)

    def to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to CPU for network transfer."""
        return tensor.cpu()

    def get_device_info(self) -> dict:
        """Get information about the current device."""
        info = {
            "type": self.device.type,
            "name": self.device_name,
        }

        if self.device.type == "cuda":
            info.update(
                {
                    "cuda_version": torch.version.cuda,
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "memory_allocated": torch.cuda.memory_allocated(0),
                    "memory_reserved": torch.cuda.memory_reserved(0),
                }
            )

        return info

    def __repr__(self) -> str:
        return f"DeviceAbstractionLayer(device={self.device}, name='{self.device_name}')"
