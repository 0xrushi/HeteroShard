"""
Core components for heterogeneous training.
"""

from .dal import DeviceAbstractionLayer
from .transport import send_tensor, recv_tensor
from .protocol import Message, MessageType

__all__ = ['DeviceAbstractionLayer', 'send_tensor', 'recv_tensor', 'Message', 'MessageType']

