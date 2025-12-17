"""Evaluation Workflow Nodes.

Nodes are organized by stage:
- state.py: EvalState schema
- common.py: Shared utilities
- add_node.py: Add stage (index building) for V1/V2/V3
- add_node_v4.py: Add stage for V4 (fine-grained MemUnit extraction)
- cluster_node.py: Cluster stage (group event clustering)
- search_node.py: Search stage for V1/V2 (BM25 + Embedding hybrid)
- search_node_v3.py: Search stage for V3 (Pure ColBERT)
- search_node_v4.py: Search stage for V4 (Cross-Attention)
- answer_node.py: Answer stage (generate answers)
- evaluate_node.py: Evaluate stage (assess quality)
"""

from eval.core.nodes.state import EvalState

# Import nodes to trigger @register_node decorators
from eval.core.nodes.add_node import eval_add_stage_node
from eval.core.nodes.add_node_v4 import eval_add_stage_v4_node
from eval.core.nodes.cluster_node import eval_cluster_stage_node
from eval.core.nodes.search_node import eval_search_stage_node
from eval.core.nodes.search_node_v3 import eval_search_stage_v3_node
from eval.core.nodes.search_node_v4 import eval_search_stage_v4_node
from eval.core.nodes.answer_node import eval_answer_stage_node
from eval.core.nodes.evaluate_node import eval_evaluate_stage_node

__all__ = [
    "EvalState",
    "eval_add_stage_node",
    "eval_add_stage_v4_node",
    "eval_cluster_stage_node",
    "eval_search_stage_node",
    "eval_search_stage_v3_node",
    "eval_search_stage_v4_node",
    "eval_answer_stage_node",
    "eval_evaluate_stage_node",
]
