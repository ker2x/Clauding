"""
MCTS Node for tree search.
"""

import math
from typing import Dict, List, Optional, Tuple


class MCTSNode:
    """
    Node in the Monte Carlo Tree Search.

    Tracks visit counts, values, and children for the PUCT algorithm.
    Uses virtual loss for batched MCTS.
    """

    def __init__(self, prior: float, parent: Optional['MCTSNode'] = None):
        self.prior = prior
        self.parent = parent
        self.visit_count = 0
        self.value_sum = 0.0
        self.virtual_loss = 0
        self.children: Dict[int, 'MCTSNode'] = {}

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_value(self) -> float:
        effective_visits = self.visit_count + self.virtual_loss
        if effective_visits == 0:
            return 0.0
        return (self.value_sum - self.virtual_loss) / effective_visits

    def expand(self, actions: list, priors: list):
        for action, prior in zip(actions, priors):
            if action not in self.children:
                self.children[action] = MCTSNode(prior=prior, parent=self)

    def select_child(self, c_puct: float) -> Tuple[int, 'MCTSNode']:
        """Select child with highest PUCT score."""
        best_score = -float('inf')
        best_action = -1
        best_child = None

        sqrt_parent_visits = math.sqrt(self.visit_count + self.virtual_loss)

        for action, child in self.children.items():
            q_value = child.get_value()
            u_value = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count + child.virtual_loss)
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def backup(self, value: float):
        """Backpropagate value up the tree."""
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip for opponent
            node = node.parent

    def get_visit_counts(self) -> Dict[int, int]:
        return {action: child.visit_count for action, child in self.children.items()}

    def get_best_action(self) -> int:
        if not self.children:
            return -1
        return max(self.children.items(), key=lambda x: x[1].visit_count)[0]


def apply_virtual_loss(path: List[MCTSNode]):
    """Apply virtual loss to all nodes in path."""
    for node in path:
        node.virtual_loss += 1


def remove_virtual_loss(path: List[MCTSNode]):
    """Remove virtual loss from all nodes in path."""
    for node in path:
        node.virtual_loss -= 1
