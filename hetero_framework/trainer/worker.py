"""
Worker

Remote worker script that runs on GPU nodes.
Receives tensors, processes them through model shard, and returns results.
"""

import socket
import torch
import torch.nn as nn
from typing import Optional
from ..core.dal import DeviceAbstractionLayer
from ..core.transport import send_tensor, recv_tensor


class RemoteWorker:
    """
    Remote worker for distributed heterogeneous training.
    
    Runs on a remote machine and processes model shards.
    """
    
    def __init__(
        self,
        model_shard: nn.Module,
        listen_ip: str = "0.0.0.0",
        listen_port: int = 9999,
        device: str = "auto"
    ):
        """
        Initialize the remote worker.
        
        Args:
            model_shard: The model shard to execute on this worker
            listen_ip: IP address to listen on (0.0.0.0 for all interfaces)
            listen_port: Port to listen on
            device: Device to use ("auto", "cuda", "mps", "cpu")
        """
        self.dal = DeviceAbstractionLayer(device)
        self.model_shard = model_shard.to(self.dal.device)
        self.listen_ip = listen_ip
        self.listen_port = listen_port
        
        self.server_socket: Optional[socket.socket] = None
        self.client_socket: Optional[socket.socket] = None
        
        print(f"Worker initialized on {self.dal.device_name}")
        print(f"Device info: {self.dal.get_device_info()}")
    
    def start(self) -> None:
        """
        Start the worker and begin listening for connections.
        
        This method blocks until the connection is closed.
        """
        print(f"\n{'='*60}")
        print(f"REMOTE WORKER STARTED")
        print(f"Device: {self.dal.device_name}")
        print(f"Listening on {self.listen_ip}:{self.listen_port}")
        print(f"{'='*60}\n")
        
        # Create server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.listen_ip, self.listen_port))
        self.server_socket.listen(1)
        
        print("Waiting for coordinator connection...")
        
        # Accept connection
        self.client_socket, addr = self.server_socket.accept()
        print(f"✓ Connected to coordinator at {addr}\n")
        
        # Main processing loop
        self._process_loop()
    
    def _process_loop(self) -> None:
        """Main processing loop for handling incoming tensors."""
        step = 0
        
        while True:
            try:
                # 1. Receive input tensor
                print(f"[Step {step}] Waiting for data...")
                input_tensor = recv_tensor(self.client_socket)
                
                if input_tensor is None:
                    print("Connection closed by coordinator.")
                    break
                
                # 2. Move to device and process
                input_tensor = self.dal.to_device(input_tensor)
                print(f"[Step {step}] Processing tensor {tuple(input_tensor.shape)}...")
                
                with torch.no_grad():
                    output_tensor = self.model_shard(input_tensor)
                
                # 3. Send result back
                print(f"[Step {step}] Sending result {tuple(output_tensor.shape)} back...")
                send_tensor(self.client_socket, self.dal.to_cpu(output_tensor))
                
                print(f"[Step {step}] ✓ Complete\n")
                step += 1
                
            except Exception as e:
                print(f"Error in processing loop: {e}")
                break
    
    def stop(self) -> None:
        """Stop the worker and close connections."""
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        print("Worker stopped.")


def run_worker(
    model_shard: nn.Module,
    listen_ip: str = "0.0.0.0",
    listen_port: int = 9999,
    device: str = "auto"
) -> None:
    """
    Convenience function to run a worker.
    
    Args:
        model_shard: The model shard to execute
        listen_ip: IP address to listen on
        listen_port: Port to listen on
        device: Device to use
    """
    worker = RemoteWorker(
        model_shard=model_shard,
        listen_ip=listen_ip,
        listen_port=listen_port,
        device=device
    )
    
    try:
        worker.start()
    except KeyboardInterrupt:
        print("\n\nReceived interrupt signal...")
    finally:
        worker.stop()

