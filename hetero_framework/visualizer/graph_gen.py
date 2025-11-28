"""
Graph Generator

Generates visual representations of the distributed training architecture.
Supports both Graphviz and Mermaid formats.
"""

from typing import Literal

try:
    from graphviz import Digraph

    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False


class GraphGenerator:
    """Generate architecture diagrams for heterogeneous training setups."""

    def __init__(self):
        """Initialize the graph generator."""
        self.nodes: list[dict] = []
        self.edges: list[dict] = []

    def add_node(
        self,
        node_id: str,
        label: str,
        node_type: Literal["local", "remote"],
        layers: list[str],
        device_info: str | None = None,
    ) -> None:
        """
        Add a compute node to the graph.

        Args:
            node_id: Unique identifier for the node
            label: Display label (e.g., "Localhost", "Worker 1")
            node_type: Type of node (local or remote)
            layers: List of layer names on this node
            device_info: Optional device information string
        """
        self.nodes.append(
            {
                "id": node_id,
                "label": label,
                "type": node_type,
                "layers": layers,
                "device_info": device_info,
            }
        )

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        label: str | None = None,
        edge_type: Literal["local", "network"] = "network",
    ) -> None:
        """
        Add a connection between nodes.

        Args:
            from_node: Source node ID
            to_node: Destination node ID
            label: Optional label for the edge
            edge_type: Type of connection (local or network)
        """
        self.edges.append({"from": from_node, "to": to_node, "label": label, "type": edge_type})

    def generate_graphviz(self, output_path: str = "architecture", format: str = "png") -> bool:
        """
        Generate architecture diagram using Graphviz.

        Args:
            output_path: Output file path (without extension)
            format: Output format (png, svg, pdf, etc.)

        Returns:
            True if successful, False otherwise

        Raises:
            ImportError: If graphviz Python package is not installed
            RuntimeError: If graphviz system binary is not found
        """
        if not GRAPHVIZ_AVAILABLE:
            error_msg = (
                "Graphviz Python package not installed!\n"
                "Install with: pip install graphviz\n"
                "Also ensure graphviz system package is installed:\n"
                "  - Ubuntu/Debian: sudo apt-get install graphviz\n"
                "  - macOS: brew install graphviz\n"
                "  - Arch: sudo pacman -S graphviz"
            )
            raise ImportError(error_msg)

        dot = Digraph(comment="Heterogeneous Training Architecture")
        dot.attr(rankdir="LR", splines="ortho")

        # Add nodes grouped by type
        for node in self.nodes:
            cluster_name = f"cluster_{node['id']}"
            color = "#e6f3ff" if node["type"] == "local" else "#ffe6e6"

            with dot.subgraph(name=cluster_name) as c:
                label_parts = [node["label"]]
                if node["device_info"]:
                    label_parts.append(node["device_info"])
                c.attr(label="\\n".join(label_parts), style="filled", color=color)

                prev_layer = None
                for i, layer in enumerate(node["layers"]):
                    layer_id = f"{node['id']}_L{i}"
                    c.node(layer_id, layer)

                    if prev_layer:
                        c.edge(prev_layer, layer_id)
                    prev_layer = layer_id

        # Add edges between nodes
        for edge in self.edges:
            from_node_data = next(n for n in self.nodes if n["id"] == edge["from"])
            next(n for n in self.nodes if n["id"] == edge["to"])

            from_layer = f"{edge['from']}_L{len(from_node_data['layers']) - 1}"
            to_layer = f"{edge['to']}_L0"

            edge_attrs = {
                "label": edge["label"] or "Network",
                "color": "red" if edge["type"] == "network" else "blue",
                "penwidth": "2.0",
            }

            dot.edge(from_layer, to_layer, **edge_attrs)

        try:
            dot.render(output_path, format=format, cleanup=True)
            print(f"Architecture diagram saved to {output_path}.{format}")
            return True
        except FileNotFoundError as e:
            error_msg = (
                f"Graphviz system binary not found!\n"
                f"Please install graphviz system package:\n"
                f"  - Ubuntu/Debian: sudo apt-get install graphviz\n"
                f"  - macOS: brew install graphviz\n"
                f"  - Arch: sudo pacman -S graphviz\n"
                f"Original error: {e}"
            )
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Error generating Graphviz diagram: {e}"
            raise RuntimeError(error_msg) from e

    def generate_mermaid(self, output_path: str = "architecture.mmd") -> bool:
        """
        Generate architecture diagram using Mermaid syntax.

        Args:
            output_path: Output file path

        Returns:
            True if successful, False otherwise
        """
        lines = ["graph LR"]

        # Add subgraphs for each node
        for node in self.nodes:
            label = node["label"]
            if node["device_info"]:
                label += f" ({node['device_info']})"

            lines.append(f"    subgraph {node['id']}[\"{label}\"]")

            # Add layers
            for i, layer in enumerate(node["layers"]):
                layer_id = f"{node['id']}_L{i}"
                lines.append(f'        {layer_id}["{layer}"]')

                # Connect layers within node
                if i > 0:
                    prev_layer = f"{node['id']}_L{i-1}"
                    lines.append(f"        {prev_layer} --> {layer_id}")

            lines.append("    end")

        # Add edges between nodes
        for edge in self.edges:
            from_node_data = next(n for n in self.nodes if n["id"] == edge["from"])
            next(n for n in self.nodes if n["id"] == edge["to"])

            from_layer = f"{edge['from']}_L{len(from_node_data['layers']) - 1}"
            to_layer = f"{edge['to']}_L0"

            label = edge["label"] or "Network"
            style = "-.->|" if edge["type"] == "network" else "-->|"
            lines.append(f"    {from_layer} {style}{label}| {to_layer}")

        # Add styling
        lines.append("")
        lines.append("    classDef localNode fill:#e6f3ff,stroke:#333,stroke-width:2px")
        lines.append("    classDef remoteNode fill:#ffe6e6,stroke:#333,stroke-width:2px")

        try:
            with open(output_path, "w") as f:
                f.write("\n".join(lines))
            print(f"Mermaid diagram saved to {output_path}")
            return True
        except Exception as e:
            print(f"Error generating Mermaid diagram: {e}")
            return False

    def clear(self) -> None:
        """Clear all nodes and edges."""
        self.nodes.clear()
        self.edges.clear()
