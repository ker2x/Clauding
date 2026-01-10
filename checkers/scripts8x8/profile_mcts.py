
import sys
import os
import time
import cProfile
import pstats
from pstats import SortKey
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from checkers8x8.engine.game import CheckersGame
from checkers8x8.network.resnet import CheckersNetwork
from checkers8x8.mcts.mcts import MCTS
from config8x8 import Config

def profile_run():
    print("Setting up MCTS Profiling...")
    
    # Setup
    device = torch.device("cpu") # Test on CPU first as that's what self-play uses
    network = CheckersNetwork(
        num_filters=Config.NUM_FILTERS, 
        num_res_blocks=Config.NUM_RES_BLOCKS, 
        policy_size=Config.POLICY_SIZE
    ).to(device)
    network.eval()
    
    game = CheckersGame()
    
    mcts = MCTS(
        network=network,
        c_puct=Config.C_PUCT,
        num_simulations=100, # Run enough to get good data
        dirichlet_alpha=Config.DIRICHLET_ALPHA,
        dirichlet_epsilon=Config.DIRICHLET_EPSILON,
        device=device
    )
    
    print("\nStarting Profiling Run (100 simulations)...")
    start_time = time.time()
    
    # Run profiling
    with cProfile.Profile() as pr:
        mcts.search(game, add_noise=True)
        
    end_time = time.time()
    print(f"\nTotal Time: {end_time - start_time:.4f} seconds")
    
    # Analyze results
    stats = pstats.Stats(pr)
    stats.sort_stats(SortKey.TIME)
    
    print("\nTop 20 functions by time spent:")
    stats.print_stats(20)
    
    # Specific checks
    print("\nBreakdown by Key Components:")
    stats.print_stats('clone')
    stats.print_stats('resnet') # Network forward pass
    stats.print_stats('evaluate_state')
    stats.print_stats('make_action')

if __name__ == "__main__":
    profile_run()
