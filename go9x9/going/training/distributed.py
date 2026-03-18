"""
Distributed self-play protocol helpers.

Length-prefixed pickle over TCP, stdlib only.
"""

import pickle
import struct
import socket


DEFAULT_PORT = 9377

# Config keys to serialize for remote self-play
_CONFIG_KEYS = [
    'NUM_FILTERS', 'NUM_RES_BLOCKS', 'INPUT_PLANES', 'POLICY_SIZE',
    'GLOBAL_POOL_FREQ', 'OWNERSHIP_LOSS_WEIGHT',
    'MCTS_SIMS_SELFPLAY', 'MCTS_SIMS_FAST', 'MCTS_BATCH_SIZE',
    'MCTS_EARLY_TERM', 'P_FAST_MOVE',
    'C_PUCT', 'DIRICHLET_ALPHA', 'DIRICHLET_EPSILON',
    'TEMPERATURE_THRESHOLD', 'TEMPERATURE',
    'GAMES_PER_ITERATION', 'NUM_WORKERS',
    'MAX_GAME_LENGTH', 'KOMI',
]


def send_msg(sock: socket.socket, obj):
    """Send a length-prefixed pickled object."""
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack('>I', len(data))
    sock.sendall(header + data)


def recv_msg(sock: socket.socket):
    """Receive a length-prefixed pickled object."""
    header = _recv_exact(sock, 4)
    if header is None:
        return None
    length = struct.unpack('>I', header)[0]
    data = _recv_exact(sock, length)
    if data is None:
        return None
    return pickle.loads(data)


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    """Read exactly n bytes from socket."""
    parts = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(min(remaining, 65536))
        if not chunk:
            return None
        parts.append(chunk)
        remaining -= len(chunk)
    return b''.join(parts)


def config_to_dict(config) -> dict:
    """Extract relevant hyperparameters from Config for transport."""
    return {k: getattr(config, k) for k in _CONFIG_KEYS}


def dict_to_config(d: dict):
    """Create a simple namespace object from a config dict."""
    class RemoteConfig:
        pass
    cfg = RemoteConfig()
    for k, v in d.items():
        setattr(cfg, k, v)
    return cfg
