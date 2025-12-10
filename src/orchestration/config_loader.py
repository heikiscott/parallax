"""Configuration loader for LangGraph workflows.

This module handles loading and parsing YAML workflow configurations.
Uses the unified config loader from config/ for YAML parsing and secrets injection.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from config.loader import load_yaml


@dataclass
class NodeConfig:
    """Configuration for a single node in the workflow."""
    name: str
    function: str
    config: Dict[str, Any]
    condition: Optional[str] = None


@dataclass
class EdgeConfig:
    """Configuration for an edge in the workflow."""
    from_node: str
    to_node: str
    condition: Optional[str] = None


@dataclass
class ConditionalEdgeConfig:
    """Configuration for a conditional edge in the workflow."""
    source: str  # Source node name
    router: str  # Router function name (from ROUTER_REGISTRY)
    destinations: Dict[str, str]  # {router_return_value: target_node_name}


@dataclass
class WorkflowConfig:
    """Configuration for an entire workflow."""
    name: str
    nodes: List[NodeConfig]
    edges: List[EdgeConfig]
    conditional_edges: List[ConditionalEdgeConfig]
    global_config: Dict[str, Any]


class ConfigLoader:
    """Loader for YAML workflow configurations."""

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize config loader.

        Args:
            config_dir: Directory containing workflow configs.
                       Defaults to project_root/config/workflows
        """
        if config_dir is None:
            # Default to project_root/config/workflows
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "config" / "workflows"

        self.config_dir = Path(config_dir)

    def load(self, config_name: str) -> WorkflowConfig:
        """Load a workflow configuration by name.

        Args:
            config_name: Name of the config file (without .yaml extension)
                        e.g., "agentic_hybrid"

        Returns:
            WorkflowConfig object

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
            ValueError: If config is missing required fields
        """
        config_path = self.config_dir / f"{config_name}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        raw_config = load_yaml(str(config_path))
        return self._parse_config(raw_config)

    def load_from_path(self, config_path: str) -> WorkflowConfig:
        """Load a workflow configuration from full path.

        Args:
            config_path: Full path to config file

        Returns:
            WorkflowConfig object
        """
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        raw_config = load_yaml(str(path))
        return self._parse_config(raw_config)

    def _parse_config(self, raw_config: Dict[str, Any]) -> WorkflowConfig:
        """Parse raw YAML config into WorkflowConfig object.

        Args:
            raw_config: Dictionary from YAML file

        Returns:
            WorkflowConfig object

        Raises:
            ValueError: If config is missing required fields
        """
        if 'workflow' not in raw_config:
            raise ValueError("Config must contain 'workflow' section")

        workflow_data = raw_config['workflow']

        # Parse name
        if 'name' not in workflow_data:
            raise ValueError("Workflow must have a 'name'")
        name = workflow_data['name']

        # Parse nodes
        nodes = []
        if 'nodes' not in workflow_data:
            raise ValueError("Workflow must have 'nodes' section")

        for node_data in workflow_data['nodes']:
            nodes.append(NodeConfig(
                name=node_data['name'],
                function=node_data['function'],
                config=node_data.get('config', {}),
                condition=node_data.get('condition')
            ))

        # Parse edges
        edges = []
        if 'edges' in workflow_data:
            for edge_data in workflow_data['edges']:
                edges.append(EdgeConfig(
                    from_node=edge_data.get('from', edge_data.get('from_node')),
                    to_node=edge_data.get('to', edge_data.get('to_node')),
                    condition=edge_data.get('condition')
                ))

        # Parse conditional edges
        conditional_edges = []
        if 'conditional_edges' in workflow_data:
            for cond_edge_data in workflow_data['conditional_edges']:
                conditional_edges.append(ConditionalEdgeConfig(
                    source=cond_edge_data['source'],
                    router=cond_edge_data['router'],
                    destinations=cond_edge_data.get('destinations', {})
                ))

        # Parse global config
        global_config = raw_config.get('global_config', {})

        return WorkflowConfig(
            name=name,
            nodes=nodes,
            edges=edges,
            conditional_edges=conditional_edges,
            global_config=global_config
        )

    def list_configs(self) -> List[str]:
        """List all available workflow configurations.

        Returns:
            List of config names (without .yaml extension)
        """
        if not self.config_dir.exists():
            return []

        return [
            f.stem for f in self.config_dir.glob("*.yaml")
        ]


def load_workflow_config(config_name: str) -> WorkflowConfig:
    """Convenience function to load a workflow config.

    Args:
        config_name: Name of the config file (without .yaml extension)

    Returns:
        WorkflowConfig object
    """
    loader = ConfigLoader()
    return loader.load(config_name)
