#!/usr/bin/env python3
"""
GTP entry point for 9x9 Go engine.

Usage in Sabaki:
    Engine path: .venv/bin/python
    Engine args: go9x9/scripts/gtp_play.py --model checkpoints/best_model.pt

Usage with gogui-twogtp:
    gogui-twogtp -black '.venv/bin/python go9x9/scripts/gtp_play.py --model checkpoints/best_model.pt' \
                 -white 'katago gtp' -games 10 -size 9
"""

import argparse
import sys
import os
import queue
import threading
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from going.network.resnet import GoNetwork
from going.gtp.engine import GTPEngine
from going.gtp.controller import GTPController
from config import Config


def main():
    # Start buffering stdin immediately before slow imports/model loading,
    # so Sabaki's handshake commands don't time out while we're initializing.
    cmd_queue = queue.Queue()
    def _stdin_reader():
        while True:
            line = sys.stdin.readline()
            if not line:
                cmd_queue.put(None)  # EOF
                break
            cmd_queue.put(line)
    threading.Thread(target=_stdin_reader, daemon=True).start()

    parser = argparse.ArgumentParser(description="9x9 Go GTP Engine")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--simulations", type=int, default=None,
                        help="Override MCTS simulations")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for inference (cpu/mps)")
    args = parser.parse_args()

    os.chdir(project_root)

    # Setup device
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Create network
    network = GoNetwork(
        num_filters=Config.NUM_FILTERS,
        num_res_blocks=Config.NUM_RES_BLOCKS,
        policy_size=Config.POLICY_SIZE,
        input_planes=Config.INPUT_PLANES,
        global_pool_freq=Config.GLOBAL_POOL_FREQ,
    )

    # Load model weights
    if args.model and Path(args.model).exists():
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
        network.load_state_dict(checkpoint['network_state_dict'])
        sys.stderr.write(f"Loaded model from {args.model}\n")
    else:
        sys.stderr.write("No model loaded, using random weights\n")

    network.to(device)
    network.eval()

    # Override simulations if specified
    if args.simulations:
        Config.MCTS_SIMS_EVAL = args.simulations

    # Create GTP engine and controller
    engine = GTPEngine(network, Config, device)
    controller = GTPController(engine, cmd_queue=cmd_queue)

    sys.stderr.write("GTP engine ready\n")
    controller.run()


if __name__ == "__main__":
    main()
