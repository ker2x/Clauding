"""
GTP command handlers for 9x9 Go engine.
"""

import torch
import numpy as np
from typing import Optional

try:
    from ..engine.game import GoGame, BLACK, WHITE
    from ..engine.action_encoder import (
        gtp_to_action, action_to_gtp, PASS_ACTION, BOARD_SIZE
    )
    from ..engine.scoring import score_game
    from ..network.resnet import GoNetwork
    from ..mcts.mcts import MCTS
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from going.engine.game import GoGame, BLACK, WHITE
    from going.engine.action_encoder import (
        gtp_to_action, action_to_gtp, PASS_ACTION, BOARD_SIZE
    )
    from going.engine.scoring import score_game
    from going.network.resnet import GoNetwork
    from going.mcts.mcts import MCTS


class GTPEngine:
    """GTP protocol command handler."""

    SUPPORTED_COMMANDS = [
        "protocol_version", "name", "version", "known_command",
        "list_commands", "boardsize", "clear_board", "komi",
        "play", "genmove", "quit", "showboard", "final_score",
        "time_settings", "time_left"
    ]

    def __init__(self, network: GoNetwork, config, device: torch.device):
        self.network = network
        self.config = config
        self.device = device
        self.game = GoGame()
        self.game.komi = config.KOMI

        self.mcts = MCTS(
            network=self.network,
            c_puct=config.C_PUCT,
            num_simulations=config.MCTS_SIMS_EVAL,
            dirichlet_alpha=config.DIRICHLET_ALPHA,
            dirichlet_epsilon=0.0,  # No noise for play
            device=device,
            batch_size=getattr(config, 'MCTS_BATCH_SIZE', 1)
        )

    def handle_command(self, command: str) -> tuple:
        """
        Handle a GTP command.

        Args:
            command: Full GTP command string

        Returns:
            (success, response): success is bool, response is string
        """
        parts = command.strip().split()
        if not parts:
            return True, ""

        # Check for optional command ID
        cmd_id = None
        if parts[0].isdigit():
            cmd_id = parts[0]
            parts = parts[1:]

        if not parts:
            return True, ""

        cmd = parts[0].lower()
        args = parts[1:]

        handler = getattr(self, f"cmd_{cmd}", None)
        if handler is None:
            return False, f"unknown command: {cmd}"

        try:
            result = handler(args)
            return True, result
        except Exception as e:
            return False, str(e)

    def cmd_protocol_version(self, args) -> str:
        return "2"

    def cmd_name(self, args) -> str:
        return "going"

    def cmd_version(self, args) -> str:
        return "0.1"

    def cmd_known_command(self, args) -> str:
        if args and args[0].lower() in self.SUPPORTED_COMMANDS:
            return "true"
        return "false"

    def cmd_list_commands(self, args) -> str:
        return "\n".join(self.SUPPORTED_COMMANDS)

    def cmd_boardsize(self, args) -> str:
        if not args:
            raise ValueError("boardsize requires an argument")
        size = int(args[0])
        if size != 9:
            raise ValueError(f"unsupported board size: {size} (only 9 supported)")
        self.game = GoGame()
        self.game.komi = self.config.KOMI
        return ""

    def cmd_clear_board(self, args) -> str:
        self.game = GoGame()
        self.game.komi = self.config.KOMI
        return ""

    def cmd_komi(self, args) -> str:
        if not args:
            raise ValueError("komi requires an argument")
        self.game.komi = float(args[0])
        return ""

    def cmd_play(self, args) -> str:
        """Handle 'play color vertex' command."""
        if len(args) < 2:
            raise ValueError("play requires color and vertex")

        color_str = args[0].lower()
        vertex = args[1].lower()

        if color_str in ("b", "black"):
            color = BLACK
        elif color_str in ("w", "white"):
            color = WHITE
        else:
            raise ValueError(f"invalid color: {color_str}")

        # Verify it's the right player's turn
        if color != self.game.current_player:
            # GTP allows playing out of turn - we just play the move
            # Some controllers expect this to work
            self.game.current_player = color

        action = gtp_to_action(vertex)
        if not self.game.make_action(action):
            raise ValueError(f"illegal move: {vertex}")

        return ""

    def cmd_genmove(self, args) -> str:
        """Generate and play a move."""
        if not args:
            raise ValueError("genmove requires a color argument")

        color_str = args[0].lower()
        if color_str in ("b", "black"):
            color = BLACK
        elif color_str in ("w", "white"):
            color = WHITE
        else:
            raise ValueError(f"invalid color: {color_str}")

        # Set current player
        if color != self.game.current_player:
            self.game.current_player = color

        # Check if game is over
        if self.game.is_terminal():
            return "pass"

        # Run MCTS
        policy = self.mcts.search(self.game, add_noise=False)
        action = self.mcts.get_best_action()

        if action == -1 or action == PASS_ACTION:
            self.game.make_action(PASS_ACTION)
            return "pass"

        # Play the move
        move_str = action_to_gtp(action)
        if not self.game.make_action(action):
            # Fallback: pass
            self.game.make_action(PASS_ACTION)
            return "pass"

        return move_str

    def cmd_showboard(self, args) -> str:
        return self.game.render()

    def cmd_final_score(self, args) -> str:
        return self.game.get_score_string()

    def cmd_time_settings(self, args) -> str:
        # Accept but ignore time settings
        return ""

    def cmd_time_left(self, args) -> str:
        # Accept but ignore time left
        return ""

    def cmd_quit(self, args) -> str:
        return ""
