"""
TCP/IP Tensor Transfer

Handles efficient serialization and network transfer of PyTorch tensors.
"""

import socket
import struct
import io
import torch
from typing import Optional


def send_tensor(sock: socket.socket, tensor: torch.Tensor) -> None:
    """
    Serialize and send a PyTorch tensor over a socket.
    
    Protocol:
        1. Send 8-byte header with tensor data size (big-endian unsigned long long)
        2. Send serialized tensor data
    
    Args:
        sock: Connected socket to send data through
        tensor: PyTorch tensor to send
    """
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    data = buffer.getvalue()
    
    # Header: 8 bytes (unsigned long long) representing size
    sock.sendall(struct.pack(">Q", len(data)))
    sock.sendall(data)


def recv_tensor(sock: socket.socket) -> Optional[torch.Tensor]:
    """
    Receive and deserialize a PyTorch tensor from a socket.
    
    Args:
        sock: Connected socket to receive data from
        
    Returns:
        Deserialized PyTorch tensor, or None if connection closed
    """
    # Read Header (8 bytes)
    raw_size = _recvall(sock, 8)
    if not raw_size:
        return None
    
    size = struct.unpack(">Q", raw_size)[0]
    
    # Read Data
    data = _recvall(sock, size)
    if not data:
        return None
    
    buffer = io.BytesIO(data)
    return torch.load(buffer)


def _recvall(sock: socket.socket, n: int) -> Optional[bytes]:
    """
    Helper function to receive exactly n bytes from a socket.
    
    Args:
        sock: Socket to receive from
        n: Number of bytes to receive
        
    Returns:
        Received bytes, or None if connection closed
    """
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

