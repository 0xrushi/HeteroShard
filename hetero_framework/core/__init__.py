"""
Core components for heterogeneous training.
"""

from .dal import DeviceAbstractionLayer
from .protocol import Message, MessageType
from .transport import recv_tensor, send_tensor

__all__ = [
    "DeviceAbstractionLayer",
    "send_tensor",
    "recv_tensor",
    "Message",
    "MessageType",
]
