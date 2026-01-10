"""
Monte Carlo Tree Search for 8x8 Checkers with Fixed Action Space.

Key improvement: Uses fixed action space directly, no need to map between
move objects and action indices during search.
"""

import numpy as np
import torch
from typing import Optional

try:
    from .node import MCTSNode
    from ..engine.game import CheckersGame
    from ..network.resnet import CheckersNetwork
    from ..engine.action_encoder import NUM_ACTIONS
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from mcts.node import MCTSNode
    from engine.game import CheckersGame
    from network.resnet import CheckersNetwork
    from engine.action_encoder import NUM_ACTIONS

try:
    import checkers_cpp
    USE_CPP = True
except ImportError:
    USE_CPP = False
    print("Warning: checkers_cpp not found, using slow Python MCTS")


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance.

    Uses PUCT algorithm with fixed action space.
    """

    def __init__(
        self,
        network: CheckersNetwork,
        c_puct: float = 1.0,
        num_simulations: int = 100,
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
            Policy distribution over all 128 actions based on visit counts
        """
        if USE_CPP:
            return self._search_cpp(game, add_noise)
            
        # Initialize root node
        self.root = MCTSNode(prior=1.0, parent=None)

        # Evaluate root with network
        legal_actions = game.get_legal_actions()
        if not legal_actions:
            return np.zeros(NUM_ACTIONS)

        # Get policy and value from network
        policy_probs, _ = self._evaluate_state(game, legal_actions)

        # Add Dirichlet noise to root for exploration
        if add_noise and len(legal_actions) > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_actions))

            # Mix noise with network priors
            for i, action in enumerate(legal_actions):
                policy_probs[action] = (
                    (1 - self.dirichlet_epsilon) * policy_probs[action] +
                    self.dirichlet_epsilon * noise[i]
                )

        # Expand root
        priors = [policy_probs[action] for action in legal_actions]
        self.root.expand(legal_actions, priors)

        # Run simulations
        for _ in range(self.num_simulations):
            # Clone game state for simulation
            sim_game = game.clone()
            self._simulate(sim_game, self.root)

        # Get policy from visit counts
        policy = self._get_action_probs(temperature=1.0)
        return policy

    def _search_cpp(self, game, add_noise: bool) -> np.ndarray:
        """Run MCTS using C++ extension."""
        
        # 1. Create/Copy C++ Game
        if isinstance(game, checkers_cpp.Game):
            # Already C++ game, usage proper copy constructor
            cpp_game = checkers_cpp.Game(game)
        else:
            # Convert Python Game to C++ Game
            cpp_game = checkers_cpp.Game()
            cpp_game.player_men = game.player_men
            cpp_game.player_kings = game.player_kings
            cpp_game.opponent_men = game.opponent_men
            cpp_game.opponent_kings = game.opponent_kings
            cpp_game.current_player = game.current_player
            cpp_game.move_count = game.move_count
            
            # Copy position history (critical for draw detection)
            # PyBind11 handles List -> std::vector conversion
            # Convert Python dict {key: count} to C++ vector [key, key, ...]
            history_list = []
            for key, count in game.position_history.items():
                history_list.extend([key] * count)
            cpp_game.position_history = history_list
        
        # 2. Setup C++ MCTS
        # If noise disabled, set epsilon to 0
        epsilon = self.dirichlet_epsilon if add_noise else 0.0
        
        cpp_mcts = checkers_cpp.MCTS(
            self.c_puct, 
            self.num_simulations, 
            self.dirichlet_alpha, 
            epsilon
        )
        
        cpp_mcts.start_search(cpp_game)
        
        
        # 3. Batch Loop
        batch_size = 16
        
        while not cpp_mcts.is_finished():
            # Get batch of leaves
            batch_id, inputs = cpp_mcts.find_leaves(batch_size)
            
            if batch_id == -1:
                break
                
            # Inputs is list of flat floats. 
            input_tensor = torch.tensor(inputs, dtype=torch.float32, device=self.device)
            input_tensor = input_tensor.view(-1, 8, 8, 8)
            
            with torch.no_grad():
                policy_logits, values = self.network(input_tensor)
                
            # Softmax
            policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()
            values = values.cpu().numpy().flatten()
            
            cpp_mcts.process_results(batch_id, policy_probs.tolist(), values.tolist())
            
        # 4. Get Result
        return np.array(cpp_mcts.get_policy(1.0))

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
            if not game.make_action(action):
                # Invalid action (shouldn't happen)
                break

        # Check if game ended
        if game.is_terminal():
            # Use game result as value
            value = game.get_result()
        else:
            # Expansion and evaluation
            legal_actions = game.get_legal_actions()

            if not legal_actions:
                # No legal moves - terminal state
                value = game.get_result()
            else:
                # Evaluate with network
                policy_probs, value = self._evaluate_state(game, legal_actions)

                # Expand node
                priors = [policy_probs[action] for action in legal_actions]
                node.expand(legal_actions, priors)

        # Backup: propagate value up the tree
        node.backup(value)

    def _evaluate_state(
        self,
        game: CheckersGame,
        legal_actions: list
    ) -> tuple:
        """
        Evaluate game state with neural network.

        Args:
            game: Game state to evaluate
            legal_actions: List of legal action indices

        Returns:
            (policy_probs, value): Policy probabilities (128,) and state value
        """
        # Convert game state to neural input
        state = game.to_neural_input()
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)

        # Evaluate with network
        with torch.no_grad():
            policy_logits, value = self.network(state_tensor)

            # Mask illegal actions
            mask = torch.full_like(policy_logits, float('-inf'))
            for action in legal_actions:
                mask[0, action] = 0.0

            masked_logits = policy_logits + mask

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
            Policy distribution (size = 128)
        """
        if self.root is None or not self.root.children:
            return np.zeros(NUM_ACTIONS)

        # Get visit counts
        visit_counts = self.root.get_visit_counts()

        # Create policy vector
        policy = np.zeros(NUM_ACTIONS)

        if temperature == 0:
            # Greedy: put all probability on most visited action
            best_action = max(visit_counts.items(), key=lambda x: x[1])[0]
            policy[best_action] = 1.0
        else:
            # Temperature scaling
            actions = list(visit_counts.keys())
            counts = np.array([visit_counts[a] for a in actions])

            counts = counts ** (1.0 / temperature)
            probs = counts / counts.sum()

            for action, prob in zip(actions, probs):
                policy[action] = prob

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


# Testing
if __name__ == "__main__":
    print("Testing MCTS with Fixed Action Space")
    print("=" * 60)

    # Create network and game
    network = CheckersNetwork(num_filters=128, num_res_blocks=6, policy_size=128)
    network.eval()

    game = CheckersGame()

    # Create MCTS
    mcts = MCTS(
        network=network,
        c_puct=1.0,
        num_simulations=50,  # Reduced for testing
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25
    )

    print(f"Initial game state:")
    print(game.render())
    print(f"Legal actions: {game.get_legal_actions()}")

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

    # Apply action and run again
    if game.make_action(best_action):
        print(f"\nAfter move:")
        print(game.render())

        # Run MCTS again
        policy2 = mcts.search(game, add_noise=True)
        best_action2 = mcts.get_best_action()
        print(f"Best action: {best_action2}")

    print("\n✓ MCTS tests passed!")
