"""
Monte Carlo Tree Search for 9x9 Go.
"""

import numpy as np
import torch
from typing import Optional, List, Tuple

try:
    from .node import MCTSNode, apply_virtual_loss, remove_virtual_loss
    from ..engine.game import GoGame
    from ..network.resnet import GoNetwork
    from ..engine.action_encoder import NUM_ACTIONS
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from mcts.node import MCTSNode, apply_virtual_loss, remove_virtual_loss
    from engine.game import GoGame
    from network.resnet import GoNetwork
    from engine.action_encoder import NUM_ACTIONS


class MCTS:
    """Monte Carlo Tree Search with neural network guidance and batched evaluation."""

    def __init__(
        self,
        network: GoNetwork,
        c_puct: float = 1.0,
        num_simulations: int = 200,
        dirichlet_alpha: float = 0.03,
        dirichlet_epsilon: float = 0.25,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 1,
        early_term_threshold: float = 0.5
    ):
        self.network = network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.device = device
        self.batch_size = batch_size
        self.early_term_threshold = early_term_threshold
        self.root: Optional[MCTSNode] = None

    def search(self, game: GoGame, add_noise: bool = True) -> np.ndarray:
        """
        Run MCTS from current game state.

        Args:
            game: Current game state
            add_noise: Whether to add Dirichlet noise to root

        Returns:
            Policy distribution over all 82 actions based on visit counts
        """
        self.root = MCTSNode(prior=1.0, parent=None)

        legal_actions = game.get_legal_actions()
        if not legal_actions:
            return np.zeros(NUM_ACTIONS)

        # Evaluate root
        policy_probs, _ = self._evaluate_state(game, legal_actions)

        # Store clean prior (before noise) for surprise-weight computation
        self.root_prior = policy_probs.copy()

        # Add Dirichlet noise for exploration
        if add_noise and len(legal_actions) > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_actions))
            for i, action in enumerate(legal_actions):
                policy_probs[action] = (
                    (1 - self.dirichlet_epsilon) * policy_probs[action] +
                    self.dirichlet_epsilon * noise[i]
                )

        # Expand root
        priors = [policy_probs[action] for action in legal_actions]
        self.root.expand(legal_actions, priors)

        # Run batched simulations with early termination
        min_sims = int(self.num_simulations * self.early_term_threshold)
        sims_completed = 0
        while sims_completed < self.num_simulations:
            batch = min(self.batch_size, self.num_simulations - sims_completed)
            leaves = self._find_leaves(game, batch)

            # Separate terminal leaves from NN leaves
            terminal_leaves = []
            nn_leaves = []
            for leaf_info in leaves:
                if leaf_info['terminal']:
                    terminal_leaves.append(leaf_info)
                else:
                    nn_leaves.append(leaf_info)

            # Handle terminal leaves immediately
            for leaf in terminal_leaves:
                remove_virtual_loss(leaf['path'])
                leaf['node'].backup(leaf['value'])

            # Batch evaluate NN leaves
            if nn_leaves:
                self._batch_evaluate_and_backup(nn_leaves)

            sims_completed += len(leaves)

            # Early termination: stop if leader can't be overtaken
            if sims_completed >= min_sims and len(self.root.children) > 1:
                remaining = self.num_simulations - sims_completed
                visits = sorted(
                    (child.visit_count for child in self.root.children.values()),
                    reverse=True,
                )
                if visits[0] - visits[1] > remaining:
                    break

        return self._get_action_probs(temperature=1.0)

    def _find_leaves(self, game: GoGame, batch_size: int) -> List[dict]:
        """Find up to batch_size leaf nodes by traversing the tree with virtual loss."""
        leaves = []

        for _ in range(batch_size):
            sim_game = game.clone()
            node = self.root
            path = [node]

            # Traverse to leaf
            while not node.is_leaf() and not sim_game.is_terminal():
                action, node = node.select_child(self.c_puct)
                path.append(node)
                if not sim_game.make_action(action):
                    break

            # Apply virtual loss to discourage other paths from going here
            apply_virtual_loss(path)

            if sim_game.is_terminal():
                value = sim_game.get_result()
                leaves.append({
                    'node': node,
                    'path': path,
                    'terminal': True,
                    'value': value,
                })
            else:
                legal_actions = sim_game.get_legal_actions()
                if not legal_actions:
                    value = sim_game.get_result()
                    leaves.append({
                        'node': node,
                        'path': path,
                        'terminal': True,
                        'value': value,
                    })
                else:
                    state = sim_game.to_neural_input()
                    leaves.append({
                        'node': node,
                        'path': path,
                        'terminal': False,
                        'state': state,
                        'legal_actions': legal_actions,
                    })

        return leaves

    def _batch_evaluate_and_backup(self, nn_leaves: List[dict]):
        """Batch evaluate leaf states with the network, expand, and backup."""
        states = np.stack([leaf['state'] for leaf in nn_leaves])
        state_tensor = torch.from_numpy(states).to(self.device)

        with torch.no_grad():
            policy_logits, values, _ = self.network(state_tensor)

            # Build mask for all leaves at once
            mask = torch.full_like(policy_logits, float('-inf'))
            for i, leaf in enumerate(nn_leaves):
                for action in leaf['legal_actions']:
                    mask[i, action] = 0.0

            masked_logits = policy_logits + mask
            all_policies = torch.softmax(masked_logits, dim=1).cpu().numpy()
            all_values = values[:, 0].cpu().numpy()

        for i, leaf in enumerate(nn_leaves):
            policy = all_policies[i]
            value = float(all_values[i])
            legal_actions = leaf['legal_actions']

            priors = [policy[action] for action in legal_actions]
            leaf['node'].expand(legal_actions, priors)

            remove_virtual_loss(leaf['path'])
            leaf['node'].backup(value)

    def _evaluate_state(self, game: GoGame, legal_actions: list) -> tuple:
        """Evaluate game state with neural network (used for root evaluation)."""
        state = game.to_neural_input()
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value, _ = self.network(state_tensor)

            mask = torch.full_like(policy_logits, float('-inf'))
            for action in legal_actions:
                mask[0, action] = 0.0

            masked_logits = policy_logits + mask
            policy = torch.softmax(masked_logits, dim=1)[0].cpu().numpy()
            value = value[0, 0].item()

        return policy, value

    def _get_action_probs(self, temperature: float = 1.0) -> np.ndarray:
        """Get action probabilities from root visit counts."""
        if self.root is None or not self.root.children:
            return np.zeros(NUM_ACTIONS)

        visit_counts = self.root.get_visit_counts()
        policy = np.zeros(NUM_ACTIONS)

        if temperature == 0:
            best_action = max(visit_counts.items(), key=lambda x: x[1])[0]
            policy[best_action] = 1.0
        else:
            actions = list(visit_counts.keys())
            counts = np.array([visit_counts[a] for a in actions], dtype=np.float64)
            counts = counts ** (1.0 / temperature)
            probs = counts / counts.sum()
            for action, prob in zip(actions, probs):
                policy[action] = prob

        return policy

    def get_best_action(self) -> int:
        if self.root is None or not self.root.children:
            return -1
        return self.root.get_best_action()

    def sample_action(self, temperature: float = 1.0) -> int:
        policy = self._get_action_probs(temperature=temperature)
        legal_actions = np.where(policy > 0)[0]

        if len(legal_actions) == 0:
            return -1

        legal_probs = policy[legal_actions]
        legal_probs = legal_probs / legal_probs.sum()
        return int(np.random.choice(legal_actions, p=legal_probs))
