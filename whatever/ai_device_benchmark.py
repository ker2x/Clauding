#!/usr/bin/env python3
"""
AI Training Benchmark: Modular Arithmetic (Grokking)
=====================================================
Benchmarks a simple transformer on modular addition (a + b mod p).
Tests CPU (single/multi-thread), MPS, CUDA/ROCm with full & mixed precision.

No external data required - generates synthetic modular arithmetic dataset.
"""

import time
import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class BenchmarkConfig:
    # Modular arithmetic: (a + b) mod p
    prime: int = 97           # Vocabulary size = prime
    operation: str = "add"    # Operation type

    # Model architecture (small transformer)
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.0

    # Training
    batch_size: int = 512
    num_epochs: int = 200      # Enough to see learning, fast enough for benchmark
    learning_rate: float = 1e-3
    weight_decay: float = 0.5
    train_fraction: float = 0.5  # 50% train, 50% test

    # Benchmark
    warmup_batches: int = 5   # Warmup before timing


# ============================================================================
# Dataset Generation
# ============================================================================

def generate_modular_data(prime: int, operation: str = "add"):
    """
    Generate all pairs (a, b) with labels (a op b) mod prime.
    Returns: inputs tensor [N, 2], labels tensor [N]
    """
    a_vals = torch.arange(prime)
    b_vals = torch.arange(prime)

    # Create all pairs using meshgrid
    aa, bb = torch.meshgrid(a_vals, b_vals, indexing='ij')
    inputs = torch.stack([aa.flatten(), bb.flatten()], dim=1)  # [prime^2, 2]

    if operation == "add":
        labels = (inputs[:, 0] + inputs[:, 1]) % prime
    elif operation == "mul":
        labels = (inputs[:, 0] * inputs[:, 1]) % prime
    elif operation == "sub":
        labels = (inputs[:, 0] - inputs[:, 1]) % prime
    else:
        raise ValueError(f"Unknown operation: {operation}")

    return inputs, labels


def create_dataloaders(config: BenchmarkConfig, device: torch.device):
    """Create train/test dataloaders."""
    inputs, labels = generate_modular_data(config.prime, config.operation)

    # Shuffle and split
    n_samples = len(inputs)
    perm = torch.randperm(n_samples)
    inputs, labels = inputs[perm], labels[perm]

    n_train = int(n_samples * config.train_fraction)

    train_inputs, train_labels = inputs[:n_train], labels[:n_train]
    test_inputs, test_labels = inputs[n_train:], labels[n_train:]

    # Move to device
    train_dataset = TensorDataset(
        train_inputs.to(device),
        train_labels.to(device)
    )
    test_dataset = TensorDataset(
        test_inputs.to(device),
        test_labels.to(device)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    return train_loader, test_loader


# ============================================================================
# Model: Small Transformer for Modular Arithmetic
# ============================================================================

class ModularTransformer(nn.Module):
    """
    Transformer for modular arithmetic.
    Input: two tokens (a, b) as integers
    Output: logits over prime classes for (a op b) mod prime
    """

    def __init__(self, config: BenchmarkConfig):
        super().__init__()
        self.config = config

        # Token embedding (vocab = prime)
        self.token_embed = nn.Embedding(config.prime, config.embed_dim)

        # Positional embedding for 2 positions
        self.pos_embed = nn.Embedding(2, config.embed_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )

        # Output head
        self.ln_final = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.prime)

    def forward(self, x):
        """
        x: [batch, 2] integer tokens (a, b)
        returns: [batch, prime] logits
        """
        batch_size, seq_len = x.shape

        # Embeddings
        tok_emb = self.token_embed(x)  # [batch, 2, embed_dim]
        pos = torch.arange(seq_len, device=x.device)
        pos_emb = self.pos_embed(pos)  # [2, embed_dim]

        h = tok_emb + pos_emb  # [batch, 2, embed_dim]

        # Transformer
        h = self.transformer(h)  # [batch, 2, embed_dim]

        # Pool (mean over sequence)
        h = h.mean(dim=1)  # [batch, embed_dim]

        # Output
        h = self.ln_final(h)
        logits = self.head(h)  # [batch, prime]

        return logits


# ============================================================================
# Training Loop
# ============================================================================

@dataclass
class TrainResult:
    total_time: float
    samples_per_sec: float
    train_acc: float
    test_acc: float
    final_loss: float


def train_model(
    config: BenchmarkConfig,
    device: torch.device,
    use_mixed_precision: bool = False,
    verbose: bool = False
) -> TrainResult:
    """Train model and return benchmark results."""

    # Create data
    train_loader, test_loader = create_dataloaders(config, device)

    # Create model
    model = ModularTransformer(config).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Mixed precision scaler (for CUDA/MPS)
    use_amp = use_mixed_precision and device.type in ('cuda', 'mps')
    scaler = torch.amp.GradScaler(enabled=use_amp and device.type == 'cuda')
    amp_dtype = torch.float16 if device.type == 'cuda' else torch.float16

    # Warmup
    model.train()
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        if batch_idx >= config.warmup_batches:
            break
        with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)
        if use_amp and device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()

    # Synchronize before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()

    # Reset model for actual training
    model = ModularTransformer(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler(enabled=use_amp and device.type == 'cuda')

    # Training loop with timing
    total_samples = 0
    final_loss = 0.0

    start_time = time.perf_counter()

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0

        for inputs, labels in train_loader:
            with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                logits = model(inputs)
                loss = F.cross_entropy(logits, labels)

            if use_amp and device.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item() * inputs.size(0)
            epoch_samples += inputs.size(0)
            total_samples += inputs.size(0)

        final_loss = epoch_loss / epoch_samples

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{config.num_epochs}, Loss: {final_loss:.4f}")

    # Synchronize after training
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Evaluate accuracy
    model.eval()

    def compute_accuracy(loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                    logits = model(inputs)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0.0

    train_acc = compute_accuracy(train_loader)
    test_acc = compute_accuracy(test_loader)

    return TrainResult(
        total_time=total_time,
        samples_per_sec=total_samples / total_time,
        train_acc=train_acc,
        test_acc=test_acc,
        final_loss=final_loss
    )


# ============================================================================
# Device Detection
# ============================================================================

def get_available_devices():
    """Detect available compute devices."""
    devices = []

    # CPU single-threaded
    devices.append(("CPU (1 thread)", "cpu", 1))

    # CPU multi-threaded
    num_cores = os.cpu_count() or 4
    devices.append((f"CPU ({num_cores} threads)", "cpu", num_cores))

    # MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        devices.append(("MPS (Apple Silicon)", "mps", None))

    # CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        # Truncate long GPU names
        if len(gpu_name) > 25:
            gpu_name = gpu_name[:22] + "..."
        devices.append((f"CUDA ({gpu_name})", "cuda", None))

    # ROCm shows up as CUDA in PyTorch
    # (detected above if available)

    return devices


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_benchmark(config: BenchmarkConfig, verbose: bool = False):
    """Run benchmark on all available devices."""

    devices = get_available_devices()

    print("=" * 80)
    print("AI TRAINING BENCHMARK: Modular Arithmetic (Grokking)")
    print("=" * 80)
    print(f"Task: (a + b) mod {config.prime}")
    print(f"Dataset: {config.prime**2} total pairs, {config.train_fraction*100:.0f}% train")
    print(f"Model: Transformer ({config.num_layers}L, {config.num_heads}H, {config.embed_dim}D)")
    print(f"Training: {config.num_epochs} epochs, batch size {config.batch_size}")
    print(f"Detected devices: {len(devices)}")
    print("=" * 80)
    print()

    results = []

    for device_name, device_type, num_threads in devices:
        for precision in ["fp32", "mixed"]:
            use_mixed = (precision == "mixed")

            # Skip mixed precision on CPU (not beneficial)
            if device_type == "cpu" and use_mixed:
                continue

            # Set thread count for CPU
            if device_type == "cpu":
                torch.set_num_threads(num_threads)

            device = torch.device(device_type)
            config_name = f"{device_name} [{precision}]"

            print(f"Running: {config_name}...")

            try:
                result = train_model(config, device, use_mixed, verbose)
                results.append((config_name, result))
                print(f"  Done: {result.total_time:.2f}s, "
                      f"{result.samples_per_sec:.0f} samples/s, "
                      f"train={result.train_acc*100:.1f}%, test={result.test_acc*100:.1f}%")
            except Exception as e:
                print(f"  FAILED: {e}")
                results.append((config_name, None))

            print()

    # Print results table
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()

    # Table header
    header = f"{'Configuration':<35} {'Time (s)':>10} {'Samples/s':>12} {'Train %':>10} {'Test %':>10}"
    print(header)
    print("-" * len(header))

    # Find fastest for relative comparison
    valid_times = [r.total_time for _, r in results if r is not None]
    fastest = min(valid_times) if valid_times else 1.0

    for config_name, result in results:
        if result is None:
            print(f"{config_name:<35} {'FAILED':>10}")
        else:
            speedup = fastest / result.total_time
            speedup_str = f"({speedup:.2f}x)" if speedup < 0.99 or speedup > 1.01 else "(baseline)"
            print(f"{config_name:<35} {result.total_time:>10.2f} {result.samples_per_sec:>12.0f} "
                  f"{result.train_acc*100:>9.1f}% {result.test_acc*100:>9.1f}%")

    print()
    print("-" * len(header))

    # Speedup comparison
    print()
    print("SPEEDUP vs FASTEST:")
    for config_name, result in results:
        if result is not None:
            speedup = fastest / result.total_time
            bar_len = int(speedup * 20)
            bar = "█" * bar_len
            print(f"  {config_name:<33} {bar} {speedup:.2f}x")

    print()
    print("=" * 80)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Use deterministic seed for reproducibility
    torch.manual_seed(42)

    # Enable MPS fallback for unsupported ops
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    config = BenchmarkConfig()
    run_benchmark(config, verbose=False)
