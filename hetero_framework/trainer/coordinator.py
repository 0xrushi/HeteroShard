"""
Coordinator

Main training coordinator class (similar to SFTTrainer).
Manages the distributed training process across heterogeneous devices.
"""

import socket
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
from ..core.dal import DeviceAbstractionLayer
from ..core.transport import send_tensor, recv_tensor
from ..visualizer.graph_gen import GraphGenerator


class HeterogeneousTrainer:
    """
    Main coordinator for heterogeneous distributed training.
    
    This class manages:
    - Local model shard execution
    - Network communication with remote workers
    - Training orchestration
    - Visualization generation
    """
    
    def __init__(
        self,
        local_model: nn.Module,
        remote_workers: List[Dict[str, Any]],
        device: str = "auto",
        visualize: bool = True
    ):
        """
        Initialize the heterogeneous trainer.
        
        Args:
            local_model: The model shard to run locally
            remote_workers: List of worker configs [{"ip": "192.168.1.X", "port": 9999}]
            device: Device to use for local computation ("auto", "cuda", "mps", "cpu")
            visualize: Whether to generate architecture visualization
        """
        self.dal = DeviceAbstractionLayer(device)
        self.local_model = local_model.to(self.dal.device)
        self.remote_workers = remote_workers
        self.visualize = visualize
        
        self.sockets: List[socket.socket] = []
        self.connected = False
        
        print(f"Coordinator initialized on {self.dal.device_name}")
    
    def connect_workers(self) -> bool:
        """
        Connect to all remote workers.
        
        Returns:
            True if all connections successful, False otherwise
        """
        print(f"\nConnecting to {len(self.remote_workers)} remote worker(s)...")
        
        for i, worker_config in enumerate(self.remote_workers):
            ip = worker_config["ip"]
            port = worker_config.get("port", 9999)
            
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((ip, port))
                self.sockets.append(sock)
                print(f"  ✓ Connected to Worker {i+1} at {ip}:{port}")
            except ConnectionRefusedError:
                print(f"  ✗ Failed to connect to {ip}:{port}")
                print(f"    Is the worker running? Try: python -m hetero_framework.trainer.worker")
                return False
            except Exception as e:
                print(f"  ✗ Error connecting to {ip}:{port}: {e}")
                return False
        
        self.connected = True
        print("All workers connected successfully!\n")
        return True
    
    def train_step(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Execute one training step across the distributed system.
        
        Args:
            input_data: Input tensor for the model
            
        Returns:
            Final output tensor from the last worker
        """
        if not self.connected:
            raise RuntimeError("Workers not connected. Call connect_workers() first.")
        
        # 1. Local forward pass
        print("--> Executing local model shard")
        intermediate = self.local_model(self.dal.to_device(input_data))
        
        # 2. Send to first remote worker and propagate through chain
        current_output = intermediate
        for i, sock in enumerate(self.sockets):
            print(f"--> Sending tensor {tuple(current_output.shape)} to Worker {i+1}")
            send_tensor(sock, self.dal.to_cpu(current_output))
            
            print(f"--> Waiting for result from Worker {i+1}...")
            current_output = recv_tensor(sock)
            
            if current_output is None:
                raise RuntimeError(f"Lost connection to Worker {i+1}")
        
        return current_output
    
    def generate_visualization(
        self,
        output_path: str = "architecture",
        format: str = "png"
    ) -> None:
        """
        Generate architecture visualization.
        
        Args:
            output_path: Output file path (without extension)
            format: Output format ("png", "svg", "mmd" for Mermaid)
        """
        if not self.visualize:
            return
        
        graph = GraphGenerator()
        
        # Add local node
        local_layers = self._get_layer_names(self.local_model)
        graph.add_node(
            node_id="local",
            label="Localhost (Coordinator)",
            node_type="local",
            layers=local_layers,
            device_info=self.dal.device_name
        )
        
        # Add remote workers
        prev_node = "local"
        for i, worker_config in enumerate(self.remote_workers):
            node_id = f"worker{i+1}"
            ip = worker_config["ip"]
            
            # Placeholder layer names (workers define their own)
            worker_layers = [f"Layer {j+len(local_layers)+1}" for j in range(2)]
            
            graph.add_node(
                node_id=node_id,
                label=f"Worker {i+1}",
                node_type="remote",
                layers=worker_layers,
                device_info=ip
            )
            
            # Add network edge
            graph.add_edge(
                from_node=prev_node,
                to_node=node_id,
                label="Ethernet (TCP)",
                edge_type="network"
            )
            
            prev_node = node_id
        
        # Generate diagram
        if format == "mmd":
            graph.generate_mermaid(f"{output_path}.mmd")
        else:
            graph.generate_graphviz(output_path, format)
    
    def _get_layer_names(self, model: nn.Module) -> List[str]:
        """Extract layer names from a model."""
        layers = []
        for name, module in model.named_children():
            layer_type = module.__class__.__name__
            layers.append(f"{name}\n({layer_type})")
        return layers if layers else ["Model"]
    
    def close(self) -> None:
        """Close all connections to remote workers."""
        for sock in self.sockets:
            try:
                sock.close()
            except:
                pass
        self.sockets.clear()
        self.connected = False
        print("All connections closed.")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

