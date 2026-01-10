"""
MCTS Node structure for tree search.
"""

from typing import Dict, Optional
import numpy as np


class MCTSNode:
    """
    Node in the MCTS tree.

    Each node represents a game state and stores:
    - Visit count (N)
    - Total action value (W)
    - Mean action value (Q = W/N)
    - Prior probability from neural network (P)
    - Children nodes
    """

    def __init__(self, prior: float, parent: Optional['MCTSNode'] = None):
        """
        Initialize MCTS node.

        Args:
            prior: Prior probability from policy network
            parent: Parent node (None for root)
        """
        self.prior = prior
        self.parent = parent

        self.visit_count = 0
        self.total_value = 0.0
        self.mean_value = 0.0

        # Children: maps action index -> child node
        self.children: Dict[int, MCTSNode] = {}

        # Virtual loss for parallel search
        self.virtual_loss = 0

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children expanded)."""
        return len(self.children) == 0

    def is_root(self) -> bool:
        """Check if this is the root node."""
        return self.parent is None

    def get_value(self) -> float:
        """Get mean action value Q."""
        if self.visit_count == 0:
            return 0.0
        return self.mean_value

    def select_child(self, c_puct: float) -> tuple[int, 'MCTSNode']:
        """
        Select child with highest UCB score.

        UCB = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)

        Args:
            c_puct: Exploration constant

        Returns:
            (action, child_node)
        """
        best_score = float('-inf')
        best_action = -1
        best_child = None

        # Parent visit count for UCB calculation
        parent_visit_sqrt = np.sqrt(self.visit_count)

        for action, child in self.children.items():
            # Q value (with virtual loss for parallel search)
            if child.visit_count + child.virtual_loss > 0:
                q_value = (child.total_value - child.virtual_loss) / (
                    child.visit_count + child.virtual_loss
                )
            else:
                q_value = 0.0

            # UCB exploration term
            u_value = (
                c_puct * child.prior * parent_visit_sqrt /
                (1 + child.visit_count + child.virtual_loss)
            )

            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, actions: list[int], priors: np.ndarray):
        """
        Expand node by creating children for all legal actions.

        Args:
            actions: List of legal action indices
            priors: Prior probabilities for each action (full policy vector)
        """
        for action in actions:
            if action not in self.children:
                self.children[action] = MCTSNode(
                    prior=priors[action],
                    parent=self
                )

    def backup(self, value: float):
        """
        Propagate value up the tree from this node.

        Args:
            value: Value to backup (from current player's perspective)
        """
        node = self
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node.mean_value = node.total_value / node.visit_count

            # Flip value sign as we go up (alternating players)
            value = -value
            node = node.parent

    def add_virtual_loss(self):
        """Add virtual loss for parallel search."""
        self.virtual_loss += 1

    def remove_virtual_loss(self):
        """Remove virtual loss after evaluation."""
        self.virtual_loss -= 1
        assert self.virtual_loss >= 0

    def get_visit_counts(self) -> np.ndarray:
        """
        Get visit counts for all children as array.

        Returns:
            Array of visit counts (size = max action index + 1)
        """
        if not self.children:
            return np.array([])

        max_action = max(self.children.keys())
        visit_counts = np.zeros(max_action + 1, dtype=np.int32)

        for action, child in self.children.items():
            visit_counts[action] = child.visit_count

        return visit_counts

    def get_policy(self, temperature: float = 1.0) -> np.ndarray:
        """
        Get policy distribution based on visit counts.

        Args:
            temperature: Temperature for distribution
                        - temp=0: select most visited (greedy)
                        - temp=1: proportional to visits
                        - temp>1: more uniform

        Returns:
            Policy distribution over all legal actions
        """
        if not self.children:
            return np.array([])

        actions = list(self.children.keys())
        visit_counts = np.array([self.children[a].visit_count for a in actions])

        if temperature == 0:
            # Greedy: select most visited
            policy = np.zeros(len(actions))
            best_idx = np.argmax(visit_counts)
            policy[best_idx] = 1.0
        else:
            # Temperature scaling
            visit_counts = visit_counts ** (1.0 / temperature)
            policy = visit_counts / visit_counts.sum()

        # Create full policy vector
        max_action = max(actions)
        full_policy = np.zeros(max_action + 1)
        for i, action in enumerate(actions):
            full_policy[action] = policy[i]

        return full_policy

    def get_best_action(self) -> int:
        """Get action with highest visit count."""
        if not self.children:
            return -1

        return max(self.children.items(), key=lambda x: x[1].visit_count)[0]

    def __repr__(self) -> str:
        return (
            f"MCTSNode(N={self.visit_count}, W={self.total_value:.2f}, "
            f"Q={self.mean_value:.3f}, P={self.prior:.3f}, "
            f"children={len(self.children)})"
        )
