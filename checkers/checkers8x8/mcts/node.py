"""
MCTS Node for tree search.
"""

import math
from typing import Dict, Optional, Tuple


class MCTSNode:
    """
    Node in the Monte Carlo Tree Search.

    Tracks visit counts, values, and children for the PUCT algorithm.
    """

    def __init__(self, prior: float, parent: Optional['MCTSNode'] = None):
        """
        Initialize MCTS node.

        Args:
            prior: Prior probability from network
            parent: Parent node (None for root)
        """
        self.prior = prior
        self.parent = parent

        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, 'MCTSNode'] = {}  # action → child node

    def is_leaf(self) -> bool:
        """Check if node is a leaf (no children expanded yet)."""
        return len(self.children) == 0

    def get_value(self) -> float:
        """Get average value (Q-value) of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, actions: list, priors: list):
        """
        Expand node by creating children for legal actions.

        Args:
            actions: List of legal action indices
            priors: Prior probabilities for each action
        """
        for action, prior in zip(actions, priors):
            if action not in self.children:
                self.children[action] = MCTSNode(prior=prior, parent=self)

    def select_child(self, c_puct: float) -> Tuple[int, 'MCTSNode']:
        """
        Select child with highest PUCT score.

        PUCT = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)

        Args:
            c_puct: Exploration constant

        Returns:
            (action, child_node) with highest PUCT score
        """
        best_score = -float('inf')
        best_action = -1
        best_child = None

        # Calculate exploration factor once
        sqrt_parent_visits = math.sqrt(self.visit_count)

        for action, child in self.children.items():
            # Q-value (exploitation)
            q_value = child.get_value()

            # U-value (exploration)
            u_value = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)

            # PUCT score
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def backup(self, value: float):
        """
        Backpropagate value up the tree.

        Args:
            value: Value to backpropagate (from current node's perspective)
        """
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip value for opponent
            node = node.parent

    def get_visit_counts(self) -> Dict[int, int]:
        """Get visit counts for all children."""
        return {action: child.visit_count for action, child in self.children.items()}

    def get_best_action(self) -> int:
        """Get action with highest visit count."""
        if not self.children:
            return -1

        return max(self.children.items(), key=lambda x: x[1].visit_count)[0]


# Testing
if __name__ == "__main__":
    print("Testing MCTS Node")
    print("=" * 60)

    # Create root node
    root = MCTSNode(prior=1.0)

    # Expand with some actions
    actions = [0, 1, 5, 10]
    priors = [0.4, 0.3, 0.2, 0.1]

    root.expand(actions, priors)

    print(f"Root expanded with {len(root.children)} children")
    print(f"Children: {list(root.children.keys())}")

    # Simulate some visits
    for _ in range(10):
        action, child = root.select_child(c_puct=1.0)
        child.backup(value=0.5)

    print(f"\nAfter 10 simulations:")
    print(f"Visit counts: {root.get_visit_counts()}")
    print(f"Best action: {root.get_best_action()}")

    print("\n✓ Node tests passed!")
