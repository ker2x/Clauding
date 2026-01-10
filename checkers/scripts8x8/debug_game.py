
import sys
import os
import torch
import numpy as np
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from checkers8x8.mcts.mcts import MCTS
from checkers8x8.network.resnet import CheckersNetwork
from config8x8 import Config
import checkers_cpp

def debug_game():
    print("Initializing components for Debug Game...")
    
    # Setup Network
    device = torch.device("cpu") # Debug on CPU
    network = CheckersNetwork(
        num_filters=Config.NUM_FILTERS,
        num_res_blocks=Config.NUM_RES_BLOCKS,
        policy_size=Config.POLICY_SIZE
    ).to(device)
    network.eval()
    
    # Setup Game (C++)
    print("Using C++ Game Engine")
    game = checkers_cpp.Game()
    
    # Setup MCTS
    mcts = MCTS(
        network=network,
        c_puct=Config.C_PUCT,
        num_simulations=50, # Enough for testing
        dirichlet_alpha=Config.DIRICHLET_ALPHA,
        dirichlet_epsilon=Config.DIRICHLET_EPSILON,
        device=device
    )
    
    print("\nStarting Game Loop...")
    step = 0
    max_steps = 100
    
    while not game.is_terminal() and step < max_steps:
        step += 1
        print(f"\n[Move {step}] Player {game.current_player}")
        print(f"Board Men: P1={game.player_men:x} Opp={game.opponent_men:x}")
        
        # MCTS
        t0 = time.time()
        policy = mcts.search(game, add_noise=(step < 15))
        dt = time.time() - t0
        
        
        # Select action
        legal_actions = game.get_legal_actions()
        print(f"Legal actions ({len(legal_actions)}): {legal_actions}")
        
        # Sample directly from policy array like self_play.py does
        legal_indices = np.where(policy > 0)[0]
        if len(legal_indices) == 0:
            print(f"GAME OVER: Policy is all zeros! Sum={np.sum(policy)}")
            break
            
        # Greedy for debug
        action = legal_indices[np.argmax(policy[legal_indices])]
        
        print(f"Selected Action: {action}")
        print(f"MCTS Time: {dt:.3f}s")
        
        # Apply
        game.make_action(action)
        
    print(f"\nGame Finished in {step} moves.")
    print(f"Result: {game.get_result()}")
    
    if step < 5:
        print("⚠ WARNING: Game was suspiciously short!")

if __name__ == "__main__":
    debug_game()
