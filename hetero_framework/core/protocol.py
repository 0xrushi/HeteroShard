"""
Message Protocol

Defines the message format for communication between coordinator and workers.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional
import json


class MessageType(Enum):
    """Types of messages in the protocol."""
    FORWARD = "forward"          # Forward pass data
    BACKWARD = "backward"        # Backward pass gradients
    CONTROL = "control"          # Control messages (shutdown, config, etc.)
    STATUS = "status"            # Status updates
    ERROR = "error"              # Error messages


@dataclass
class Message:
    """
    Message structure for communication.
    
    Attributes:
        type: Type of message
        payload: Message payload (can be tensor, dict, etc.)
        metadata: Optional metadata (step number, layer info, etc.)
    """
    type: MessageType
    payload: Any
    metadata: Optional[dict] = None
    
    def to_json(self) -> str:
        """Serialize message to JSON (for control messages)."""
        return json.dumps({
            "type": self.type.value,
            "payload": self.payload,
            "metadata": self.metadata
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """Deserialize message from JSON."""
        data = json.loads(json_str)
        return cls(
            type=MessageType(data["type"]),
            payload=data["payload"],
            metadata=data.get("metadata")
        )
    
    def __repr__(self) -> str:
        return f"Message(type={self.type.value}, metadata={self.metadata})"

