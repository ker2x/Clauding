"""
Monte Carlo Tree Search implementation for checkers.
"""

import numpy as np
import torch
from typing import Optional
from .node import MCTSNode
from ..engine.game import CheckersGame
from ..network.resnet import CheckersNetwork


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance.

    Uses PUCT algorithm for tree search with batched neural network evaluation.
    """

    def __init__(
        self,
        network: CheckersNetwork,
        c_puct: float = 1.0,
        num_simulations: int = 300,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        device: torch.device = torch.device("cpu")
    ):
        """
        Initialize MCTS.

        Args:
            network: Neural network for policy and value evaluation
            c_puct: Exploration constant for PUCT
            num_simulations: Number of simulations to run
            dirichlet_alpha: Alpha parameter for Dirichlet noise
            dirichlet_epsilon: Weight of Dirichlet noise vs network prior
            device: PyTorch device
        """
        self.network = network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.device = device

        self.root: Optional[MCTSNode] = None

    def search(
        self,
        game: CheckersGame,
        add_noise: bool = True
    ) -> np.ndarray:
        """
        Run MCTS from current game state.

        Args:
            game: Current game state
            add_noise: Whether to add Dirichlet noise to root (for exploration)

        Returns:
            Policy distribution over legal moves based on visit counts
        """
        # Initialize root node
        self.root = MCTSNode(prior=1.0, parent=None)

        # Evaluate root with network
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return np.array([])

        # Get policy and value from network
        policy_probs, _ = self._evaluate_state(game, legal_moves)

        # Add Dirichlet noise to root for exploration
        if add_noise and len(legal_moves) > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
            legal_actions = [i for i in range(len(legal_moves))]

            for i, action in enumerate(legal_actions):
                policy_probs[action] = (
                    (1 - self.dirichlet_epsilon) * policy_probs[action] +
                    self.dirichlet_epsilon * noise[i]
                )

        # Expand root
        legal_actions = [i for i in range(len(legal_moves))]
        self.root.expand(legal_actions, policy_probs)

        # Run simulations
        for _ in range(self.num_simulations):
            # Clone game state for simulation
            sim_game = game.clone()
            self._simulate(sim_game, self.root)

        # Get policy from visit counts
        policy = self._get_action_probs(temperature=1.0)
        return policy

    def _simulate(self, game: CheckersGame, node: MCTSNode):
        """
        Run one simulation from current node.

        Args:
            game: Current game state (will be modified)
            node: Current node in tree
        """
        # Selection: traverse tree until leaf
        path = [node]

        while not node.is_leaf() and not game.is_terminal():
            action, node = node.select_child(self.c_puct)
            path.append(node)

            # Apply action to game
            legal_moves = game.get_legal_moves()
            if action < len(legal_moves):
                game.make_move(legal_moves[action])

        # Check if game ended
        if game.is_terminal():
            # Use game result as value
            value = game.get_result()
        else:
            # Expansion and evaluation
            legal_moves = game.get_legal_moves()

            if not legal_moves:
                # No legal moves - terminal state
                value = game.get_result()
            else:
                # Evaluate with network
                policy_probs, value = self._evaluate_state(game, legal_moves)

                # Expand node
                legal_actions = [i for i in range(len(legal_moves))]
                node.expand(legal_actions, policy_probs)

        # Backup: propagate value up the tree
        node.backup(value)

    def _evaluate_state(
        self,
        game: CheckersGame,
        legal_moves: list
    ) -> tuple[np.ndarray, float]:
        """
        Evaluate game state with neural network.

        Args:
            game: Game state to evaluate
            legal_moves: List of legal moves

        Returns:
            (policy_probs, value): Policy probabilities and state value
        """
        # Convert game state to neural input
        state = game.to_neural_input()
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)

        # Create legal moves mask
        legal_mask = torch.zeros(1, 150, dtype=torch.bool, device=self.device)
        for i in range(min(len(legal_moves), 150)):
            legal_mask[0, i] = True

        # Evaluate with network
        with torch.no_grad():
            policy_logits, value = self.network(state_tensor)

            # Mask illegal moves
            masked_logits = policy_logits.clone()
            masked_logits[~legal_mask] = float('-inf')

            # Softmax to get probabilities
            policy = torch.softmax(masked_logits, dim=1)[0].cpu().numpy()
            value = value[0, 0].item()

        return policy, value

    def _get_action_probs(self, temperature: float = 1.0) -> np.ndarray:
        """
        Get action probabilities from root visit counts.

        Args:
            temperature: Temperature for sampling

        Returns:
            Policy distribution (size = 150)
        """
        if self.root is None or not self.root.children:
            return np.zeros(150)

        # Get visit counts
        actions = list(self.root.children.keys())
        visits = np.array([self.root.children[a].visit_count for a in actions])

        if temperature == 0:
            # Greedy
            probs = np.zeros(len(actions))
            best_idx = np.argmax(visits)
            probs[best_idx] = 1.0
        else:
            # Temperature scaling
            visits = visits ** (1.0 / temperature)
            probs = visits / visits.sum()

        # Create full policy vector (size 150)
        policy = np.zeros(150)
        for i, action in enumerate(actions):
            policy[action] = probs[i]

        return policy

    def get_best_action(self) -> int:
        """Get action with highest visit count."""
        if self.root is None or not self.root.children:
            return -1

        return self.root.get_best_action()

    def sample_action(self, temperature: float = 1.0) -> int:
        """
        Sample action from policy distribution.

        Args:
            temperature: Temperature for sampling

        Returns:
            Sampled action index
        """
        policy = self._get_action_probs(temperature=temperature)
        legal_actions = np.where(policy > 0)[0]

        if len(legal_actions) == 0:
            return -1

        # Sample from distribution
        legal_probs = policy[legal_actions]
        legal_probs = legal_probs / legal_probs.sum()  # Renormalize

        action = np.random.choice(legal_actions, p=legal_probs)
        return action


# Example usage and testing
if __name__ == "__main__":
    import sys
    import os
    # Add parent directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    sys.path.insert(0, parent_dir)

    from checkers.engine.game import CheckersGame
    from checkers.network.resnet import CheckersNetwork
    from checkers.mcts.node import MCTSNode

    print("Testing MCTS...")

    # Create network and game
    network = CheckersNetwork(num_filters=128, num_res_blocks=6, policy_size=150)
    network.eval()

    game = CheckersGame()

    # Create MCTS
    mcts = MCTS(
        network=network,
        c_puct=1.0,
        num_simulations=100,  # Reduced for testing
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25
    )

    print(f"Initial game state:")
    print(game.render())
    print(f"Legal moves: {len(game.get_legal_moves())}")

    # Run MCTS
    print(f"\nRunning MCTS with {mcts.num_simulations} simulations...")
    policy = mcts.search(game, add_noise=True)

    print(f"Policy shape: {policy.shape}")
    print(f"Policy sum: {policy.sum():.4f}")
    print(f"Non-zero actions: {np.count_nonzero(policy)}")

    # Get best action
    best_action = mcts.get_best_action()
    print(f"Best action: {best_action}")
    print(f"Best action probability: {policy[best_action]:.4f}")

    # Sample action
    sampled_action = mcts.sample_action(temperature=1.0)
    print(f"Sampled action: {sampled_action}")

    # Apply action and run again
    legal_moves = game.get_legal_moves()
    if best_action < len(legal_moves):
        print(f"\nApplying move: {legal_moves[best_action]}")
        game.make_move(legal_moves[best_action])

        print(f"\nAfter move:")
        print(game.render())

        # Run MCTS again
        policy2 = mcts.search(game, add_noise=True)
        best_action2 = mcts.get_best_action()
        print(f"Best action: {best_action2}")

    print("\n✓ MCTS tests passed!")
