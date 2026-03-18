#!/usr/bin/env python3
"""
Distributed self-play server.

Runs on a remote machine (e.g. AMD mini PC). Listens for TCP connections
from the trainer, receives network weights + config, runs self-play games,
and sends training data back.

Usage:
    python scripts/selfplay_server.py --device cpu --port 9377
    python scripts/selfplay_server.py --device cuda --workers 4 --games 50
"""

import argparse
import socket
import sys
import os
import time
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from going.network.resnet import GoNetwork
from going.training.self_play import play_games_parallel, play_games_sequential
from going.training.distributed import (
    DEFAULT_PORT, send_msg, recv_msg, dict_to_config,
)


def log(msg):
    """Print with timestamp and flush."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def handle_selfplay_request(msg, network, device, args, request_num):
    """Process a self-play request: load weights, play games, return data."""
    config_dict = msg['config']
    config = dict_to_config(config_dict)

    # Override with server-side CLI args if provided
    if args.workers is not None:
        config.NUM_WORKERS = args.workers
    if args.games is not None:
        config.GAMES_PER_ITERATION = args.games

    # Load weights into network
    state_dict = msg['state_dict']
    network.load_state_dict(state_dict)
    network.to(device)
    network.eval()

    num_games = config.GAMES_PER_ITERATION

    log(f"Request #{request_num}: {num_games} games, "
        f"{config.NUM_WORKERS} workers, {config.MCTS_SIMS_SELFPLAY} sims")
    t0 = time.time()

    if device.type == "cpu" and config.NUM_WORKERS <= 1:
        results = play_games_sequential(network, config, device, num_games)
    else:
        if device.type == "cpu":
            network.share_memory()
        results = play_games_parallel(network, config, device, num_games)

    states, policies, values, ownerships, surprises, game_lengths = results
    elapsed = time.time() - t0

    avg_len = np.mean(game_lengths) if game_lengths else 0
    log(f"Request #{request_num} done: {len(states)} examples, "
        f"{num_games} games, avg length {avg_len:.0f}, {elapsed:.1f}s")

    # Convert to numpy arrays for efficient pickling
    return {
        'states': [np.asarray(s) for s in states],
        'policies': [np.asarray(p) for p in policies],
        'values': list(values),
        'ownerships': [np.asarray(o) for o in ownerships],
        'surprises': list(surprises),
        'game_lengths': list(game_lengths),
    }


def run_server(args):
    # Resolve device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "rocm" and torch.cuda.is_available():
        # ROCm uses CUDA API in PyTorch
        device = torch.device("cuda")
    elif args.device == "cpu":
        if args.threads:
            torch.set_num_threads(args.threads)
        device = torch.device("cpu")
    else:
        log(f"Warning: {args.device} not available, falling back to CPU")
        device = torch.device("cpu")

    log(f"Self-play server starting on port {args.port}, device={device}")

    # Pre-allocate network (will load weights on first request)
    network = None
    request_num = 0

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', args.port))
    server.listen(1)
    log(f"Listening on 0.0.0.0:{args.port}")

    try:
        while True:
            log("Waiting for connection...")
            conn, addr = server.accept()
            log(f"Connected: {addr[0]}:{addr[1]}")

            try:
                while True:
                    msg = recv_msg(conn)
                    if msg is None:
                        log("Connection closed by trainer.")
                        break

                    msg_type = msg.get('type', '')

                    if msg_type == 'shutdown':
                        log("Shutdown requested.")
                        conn.close()
                        return

                    if msg_type == 'selfplay':
                        request_num += 1

                        # Lazily create/recreate network if architecture changed
                        cfg = dict_to_config(msg['config'])
                        if network is None or _arch_changed(network, cfg):
                            network = GoNetwork(
                                num_filters=cfg.NUM_FILTERS,
                                num_res_blocks=cfg.NUM_RES_BLOCKS,
                                policy_size=cfg.POLICY_SIZE,
                                input_planes=cfg.INPUT_PLANES,
                                global_pool_freq=cfg.GLOBAL_POOL_FREQ,
                            )
                            log(f"Network created: {cfg.NUM_FILTERS}f, "
                                f"{cfg.NUM_RES_BLOCKS}b, {cfg.INPUT_PLANES}ip")

                        result = handle_selfplay_request(
                            msg, network, device, args, request_num)
                        send_msg(conn, result)
                        log(f"Results sent to trainer.")
                    else:
                        log(f"Unknown message type: {msg_type}")

            except (ConnectionResetError, BrokenPipeError) as e:
                log(f"Connection error: {e}")
            finally:
                conn.close()

    except KeyboardInterrupt:
        log("Server shutting down.")
    finally:
        server.close()


def _arch_changed(network, cfg):
    """Check if network architecture params differ from config."""
    return (network.num_filters != cfg.NUM_FILTERS or
            network.num_res_blocks != cfg.NUM_RES_BLOCKS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed self-play server')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                        help=f'TCP port (default: {DEFAULT_PORT})')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'rocm'],
                        help='Compute device (default: cpu)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Override NUM_WORKERS from trainer config')
    parser.add_argument('--games', type=int, default=None,
                        help='Override GAMES_PER_ITERATION from trainer config')
    parser.add_argument('--threads', type=int, default=None,
                        help='CPU threads (for --device cpu)')

    args = parser.parse_args()
    run_server(args)
