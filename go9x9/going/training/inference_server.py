"""
Inference server for MPS-accelerated self-play.

One process owns the network on MPS. Worker processes send batches of states
via a shared request queue and receive results back via per-worker queues.
Cross-worker batching is done by draining all pending requests before
each forward pass.
"""

import numpy as np
import torch
from queue import Empty


_POISON = None  # shutdown sentinel


def inference_server_worker(network, device_str, request_queue, result_queues):
    """
    Entry point for the inference server process.

    Receives (worker_id, req_id, states_np) tuples from request_queue,
    batches across workers, runs forward pass on `device_str`, and sends
    (req_id, policy_np, values_np, ownership_np) back to each worker's
    result queue.
    """
    device = torch.device(device_str)
    network = network.to(device)
    network.eval()

    while True:
        # Block until at least one request arrives
        try:
            first = request_queue.get(timeout=2.0)
        except Empty:
            continue

        if first is _POISON:
            break

        # Drain all other pending requests for cross-worker batching
        batch = [first]
        while True:
            try:
                req = request_queue.get_nowait()
                if req is _POISON:
                    request_queue.put(_POISON)  # re-queue for clean shutdown
                    break
                batch.append(req)
            except Empty:
                break

        # Each req: (worker_id, req_id, states_np) where states_np is (N, C, 9, 9)
        worker_ids = [r[0] for r in batch]
        req_ids    = [r[1] for r in batch]
        sizes      = [r[2].shape[0] for r in batch]

        states = np.concatenate([r[2] for r in batch], axis=0)
        state_tensor = torch.from_numpy(states).to(device)

        with torch.no_grad():
            policy_logits, values, ownership = network(state_tensor)

        policy_np    = policy_logits.cpu().numpy()
        values_np    = values.cpu().numpy()
        ownership_np = ownership.cpu().numpy()

        # Split results and send back to each worker
        offset = 0
        for worker_id, req_id, size in zip(worker_ids, req_ids, sizes):
            result_queues[worker_id].put((
                req_id,
                policy_np   [offset:offset + size],
                values_np   [offset:offset + size],
                ownership_np[offset:offset + size],
            ))
            offset += size


class RemoteNetwork:
    """
    Drop-in replacement for GoNetwork inside worker processes.

    Forwards tensor batches to the inference server and returns CPU tensors,
    so existing MCTS code (masking, softmax, .cpu().numpy()) works unchanged.
    """

    def __init__(self, worker_id: int, request_queue, result_queue):
        self.worker_id     = worker_id
        self.request_queue = request_queue
        self.result_queue  = result_queue
        self._req_id       = 0

    def __call__(self, state_tensor: torch.Tensor):
        req_id = self._req_id
        self._req_id += 1

        self.request_queue.put((self.worker_id, req_id, state_tensor.cpu().numpy()))

        # Result arrives in FIFO order for this worker
        result_req_id, policy_np, values_np, ownership_np = self.result_queue.get()
        assert result_req_id == req_id, "inference server result out of order"

        return (
            torch.from_numpy(policy_np),
            torch.from_numpy(values_np),
            torch.from_numpy(ownership_np),
        )

    # No-ops so SelfPlayGame / MCTS don't need to special-case RemoteNetwork
    def eval(self):
        return self

    def to(self, device):
        return self
