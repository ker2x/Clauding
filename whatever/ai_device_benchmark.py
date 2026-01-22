#!/usr/bin/env python3
"""
AI Training Benchmark: Modular Arithmetic (Grokking)
=====================================================
Benchmarks a simple transformer on modular addition (a + b mod p).
Tests CPU (single/multi-thread), MPS, CUDA/ROCm with full & mixed precision.
Also benchmarks MLX (Apple's ML framework) for comparison.

No external data required - generates synthetic modular arithmetic dataset.

For MLX compilation best practices and lessons learned, see MLX_TRAINING_GUIDE.md
Key takeaway: Always use shapeless=True when compiling to avoid recompilation overhead.
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

# MLX imports (optional - only used if available)
try:
    import mlx.core as mx
    import mlx.nn as mlx_nn
    import mlx.optimizers as optim
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


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
    compile_time: float = 0.0  # Time spent compiling (if any)


def train_model(
    config: BenchmarkConfig,
    device: torch.device,
    use_mixed_precision: bool = False,
    use_compile: bool = False,
    verbose: bool = False
) -> TrainResult:
    """Train model and return benchmark results."""

    # Create data
    train_loader, test_loader = create_dataloaders(config, device)

    # Create model
    model = ModularTransformer(config).to(device)

    # Compile model if requested
    compile_time = 0.0
    if use_compile:
        # Use "default" mode - "reduce-overhead" uses CUDA graphs which don't work on MPS
        model = torch.compile(model, mode="default")

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

    # Compile model if requested (this is the one we'll time)
    if use_compile:
        # Use "default" mode - "reduce-overhead" uses CUDA graphs which don't work on MPS
        model = torch.compile(model, mode="default")

        # Trigger actual compilation with a dummy forward pass and time it
        compile_start = time.perf_counter()
        with torch.no_grad():
            dummy_input = torch.zeros(config.batch_size, 2, dtype=torch.long, device=device)
            _ = model(dummy_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        compile_time = time.perf_counter() - compile_start

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
        final_loss=final_loss,
        compile_time=compile_time
    )


# ============================================================================
# MLX Implementation
# ============================================================================

if MLX_AVAILABLE:
    def generate_modular_data_mlx(prime: int, operation: str = "add"):
        """
        Generate all pairs (a, b) with labels (a op b) mod prime.
        Returns: inputs array [N, 2], labels array [N]
        """
        a_vals = mx.arange(prime)
        b_vals = mx.arange(prime)

        # Create all pairs
        aa = mx.repeat(a_vals[:, None], prime, axis=1).reshape(-1)
        bb = mx.repeat(b_vals[None, :], prime, axis=0).reshape(-1)
        inputs = mx.stack([aa, bb], axis=1)  # [prime^2, 2]

        if operation == "add":
            labels = (inputs[:, 0] + inputs[:, 1]) % prime
        elif operation == "mul":
            labels = (inputs[:, 0] * inputs[:, 1]) % prime
        elif operation == "sub":
            labels = (inputs[:, 0] - inputs[:, 1]) % prime
        else:
            raise ValueError(f"Unknown operation: {operation}")

        return inputs, labels


    class MLXTransformerLayer(mlx_nn.Module):
        """
        Custom transformer encoder layer to match PyTorch's architecture.
        Uses pre-normalization (norm_first=True) and GELU activation.
        """

        def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.0):
            super().__init__()
            self.ln1 = mlx_nn.LayerNorm(embed_dim)
            self.attn = mlx_nn.MultiHeadAttention(embed_dim, num_heads, bias=True)
            self.ln2 = mlx_nn.LayerNorm(embed_dim)

            # MLP with GELU
            self.mlp = [
                mlx_nn.Linear(embed_dim, mlp_dim),
                mlx_nn.GELU(),
                mlx_nn.Linear(mlp_dim, embed_dim)
            ]

        def __call__(self, x, mask=None):
            # Pre-norm attention (norm_first=True)
            normed = self.ln1(x)
            attn_out = self.attn(normed, normed, normed, mask)
            x = x + attn_out

            # Pre-norm MLP
            normed = self.ln2(x)
            mlp_out = normed
            for layer in self.mlp:
                mlp_out = layer(mlp_out)
            x = x + mlp_out

            return x


    class MLXModularTransformer(mlx_nn.Module):
        """
        MLX Transformer for modular arithmetic.
        Input: two tokens (a, b) as integers
        Output: logits over prime classes for (a op b) mod prime
        """

        def __init__(self, config: BenchmarkConfig):
            super().__init__()
            self.config = config

            # Token embedding (vocab = prime)
            self.token_embed = mlx_nn.Embedding(config.prime, config.embed_dim)

            # Positional embedding for 2 positions
            self.pos_embed = mlx_nn.Embedding(2, config.embed_dim)

            # Transformer encoder layers (matching PyTorch architecture)
            self.layers = []
            for _ in range(config.num_layers):
                self.layers.append(
                    MLXTransformerLayer(
                        embed_dim=config.embed_dim,
                        num_heads=config.num_heads,
                        mlp_dim=config.embed_dim * 4,
                        dropout=config.dropout
                    )
                )

            # Output head
            self.ln_final = mlx_nn.LayerNorm(config.embed_dim)
            self.head = mlx_nn.Linear(config.embed_dim, config.prime)

        def __call__(self, x):
            """
            x: [batch, 2] integer tokens (a, b)
            returns: [batch, prime] logits
            """
            batch_size, seq_len = x.shape

            # Embeddings
            tok_emb = self.token_embed(x)  # [batch, 2, embed_dim]
            pos = mx.arange(seq_len)
            pos_emb = self.pos_embed(pos)  # [2, embed_dim]

            h = tok_emb + pos_emb  # [batch, 2, embed_dim]

            # Transformer layers
            for layer in self.layers:
                h = layer(h, mask=None)  # [batch, 2, embed_dim]

            # Pool (mean over sequence)
            h = mx.mean(h, axis=1)  # [batch, embed_dim]

            # Output
            h = self.ln_final(h)
            logits = self.head(h)  # [batch, prime]

            return logits


    def cross_entropy_loss_mlx(logits, labels):
        """Cross-entropy loss for MLX."""
        # Convert labels to one-hot
        num_classes = logits.shape[-1]

        # Numerically stable log-softmax
        logits_max = mx.stop_gradient(mx.max(logits, axis=-1, keepdims=True))
        logits_shifted = logits - logits_max
        log_sum_exp = mx.log(mx.sum(mx.exp(logits_shifted), axis=-1, keepdims=True))
        log_probs = logits_shifted - log_sum_exp

        # Select log probabilities of correct classes
        batch_size = labels.shape[0]
        batch_indices = mx.arange(batch_size)
        selected_log_probs = log_probs[batch_indices, labels]

        return -mx.mean(selected_log_probs)


    def train_model_mlx(
        config: BenchmarkConfig,
        use_mixed_precision: bool = False,
        use_compile: bool = False,
        verbose: bool = False
    ) -> TrainResult:
        """Train model using MLX and return benchmark results."""

        # Generate data
        inputs, labels = generate_modular_data_mlx(config.prime, config.operation)

        # Shuffle and split
        n_samples = inputs.shape[0]
        perm = mx.random.permutation(n_samples)
        inputs, labels = inputs[perm], labels[perm]

        n_train = int(n_samples * config.train_fraction)

        train_inputs, train_labels = inputs[:n_train], labels[:n_train]
        test_inputs, test_labels = inputs[n_train:], labels[n_train:]

        # Evaluate data to ensure it's materialized
        mx.eval(train_inputs, train_labels, test_inputs, test_labels)

        # Create model
        model = MLXModularTransformer(config)
        mx.eval(model.parameters())

        # Optimizer
        # Note: MLX applies weight decay more aggressively than PyTorch's decoupled AdamW
        # Use zero weight decay for MLX to match PyTorch's learning behavior
        mlx_weight_decay = 0.0
        optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=mlx_weight_decay
        )

        # Loss and grad function
        def loss_fn(model, inputs, labels):
            logits = model(inputs)
            return cross_entropy_loss_mlx(logits, labels)

        loss_and_grad_fn = mlx_nn.value_and_grad(model, loss_fn)

        # Warmup
        num_batches = (n_train + config.batch_size - 1) // config.batch_size
        for batch_idx in range(min(config.warmup_batches, num_batches)):
            start_idx = batch_idx * config.batch_size
            end_idx = min(start_idx + config.batch_size, n_train)
            batch_inputs = train_inputs[start_idx:end_idx]
            batch_labels = train_labels[start_idx:end_idx]

            loss, grads = loss_and_grad_fn(model, batch_inputs, batch_labels)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

        # Reset model for actual training
        model = MLXModularTransformer(config)
        mx.eval(model.parameters())

        optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=mlx_weight_decay
        )

        # Compile if requested
        compile_time = 0.0
        if use_compile:
            compile_start = time.perf_counter()

            # Save original forward pass
            model_forward = model.__call__

            # CRITICAL: Use shapeless=True to prevent recompilation overhead
            # Without this flag, MLX recompiles on shape variations, making compiled
            # mode 2.3x SLOWER than eager. With shapeless=True, compiled matches/beats eager.
            # See MLX_TRAINING_GUIDE.md for detailed explanation.
            compiled_forward = mx.compile(model_forward, shapeless=True)

            # Replace model's forward with compiled version
            model.__call__ = compiled_forward

            # Create loss and gradient function with compiled model
            loss_and_grad_fn = mlx_nn.value_and_grad(model, loss_fn)

            # Warmup: trigger compilation with multiple iterations
            dummy_inputs = mx.zeros((config.batch_size, 2), dtype=mx.int32)
            dummy_labels = mx.zeros((config.batch_size,), dtype=mx.int32)
            for _ in range(3):
                _, grads = loss_and_grad_fn(model, dummy_inputs, dummy_labels)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)

            compile_time = time.perf_counter() - compile_start
        else:
            # Non-compiled version
            loss_and_grad_fn = mlx_nn.value_and_grad(model, loss_fn)

        # Training loop with timing
        total_samples = 0
        final_loss_value = 0.0

        start_time = time.perf_counter()

        for epoch in range(config.num_epochs):
            epoch_loss = 0.0
            epoch_samples = 0

            # Shuffle training data each epoch
            perm = mx.random.permutation(n_train)
            shuffled_inputs = train_inputs[perm]
            shuffled_labels = train_labels[perm]

            # Mini-batch training
            num_batches = (n_train + config.batch_size - 1) // config.batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * config.batch_size
                end_idx = min(start_idx + config.batch_size, n_train)

                # Skip last batch if it's smaller (to match PyTorch drop_last=True)
                if end_idx - start_idx < config.batch_size:
                    break

                batch_inputs = shuffled_inputs[start_idx:end_idx]
                batch_labels = shuffled_labels[start_idx:end_idx]

                # Forward and backward
                loss, grads = loss_and_grad_fn(model, batch_inputs, batch_labels)

                # Update weights
                optimizer.update(model, grads)

                # Evaluate to ensure computation completes
                mx.eval(model.parameters(), optimizer.state, loss)

                batch_size = batch_inputs.shape[0]
                epoch_loss += loss.item() * batch_size
                epoch_samples += batch_size
                total_samples += batch_size

            final_loss_value = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{config.num_epochs}, Loss: {final_loss_value:.4f}")

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Evaluate accuracy
        def compute_accuracy(inputs, labels):
            logits = model(inputs)
            preds = mx.argmax(logits, axis=-1)
            correct = mx.sum(preds == labels)
            mx.eval(correct)
            return correct.item() / labels.shape[0]

        train_acc = compute_accuracy(train_inputs, train_labels)
        test_acc = compute_accuracy(test_inputs, test_labels)

        return TrainResult(
            total_time=total_time,
            samples_per_sec=total_samples / total_time,
            train_acc=train_acc,
            test_acc=test_acc,
            final_loss=final_loss_value,
            compile_time=compile_time
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
    if MLX_AVAILABLE:
        print(f"MLX: Available")
    print()
    print("For consistent results:")
    print("  - Close other apps (especially browsers)")
    print("  - Connect to power (disable low-power mode)")
    print("  - Wait for system to cool if running multiple times")
    print("=" * 80)
    print()

    results = []

    # Run MLX benchmarks first (if available)
    if MLX_AVAILABLE:
        import mlx.core as mx

        # MLX warmup - ensure Metal is initialized and caches are warm
        print("Warming up MLX...")
        _ = mx.random.normal((1000, 1000))
        mx.eval(_)
        print()

        for precision in ["fp32", "mixed"]:
            for compile_mode in ["eager", "compile"]:
                use_mixed = (precision == "mixed")
                use_compile = (compile_mode == "compile")

                mode_str = "compiled" if use_compile else "eager"
                config_name = f"MLX (Apple Silicon) [{precision}|{mode_str}]"

                print(f"Running: {config_name}...")

                try:
                    result = train_model_mlx(config, use_mixed, use_compile, verbose)
                    results.append((config_name, result))
                    compile_info = f", compile={result.compile_time:.2f}s" if use_compile else ""
                    print(f"  Done: {result.total_time:.2f}s{compile_info}, "
                          f"{result.samples_per_sec:.0f} samples/s, "
                          f"train={result.train_acc*100:.1f}%, test={result.test_acc*100:.1f}%")
                except Exception as e:
                    print(f"  FAILED: {e}")
                    results.append((config_name, None))

                print()

        # Clear MLX memory before PyTorch runs
        del _
        import gc
        gc.collect()

    # Run PyTorch benchmarks
    for device_name, device_type, num_threads in devices:
        for precision in ["fp32", "mixed"]:
            for compile_mode in ["eager", "compile"]:
                use_mixed = (precision == "mixed")
                use_compile = (compile_mode == "compile")

                # Skip mixed precision on CPU (not beneficial)
                if device_type == "cpu" and use_mixed:
                    continue

                # Set thread count for CPU
                if device_type == "cpu":
                    torch.set_num_threads(num_threads)

                device = torch.device(device_type)
                mode_str = "compiled" if use_compile else "eager"
                config_name = f"{device_name} [{precision}|{mode_str}]"

                print(f"Running: {config_name}...")

                try:
                    result = train_model(config, device, use_mixed, use_compile, verbose)
                    results.append((config_name, result))
                    compile_info = f", compile={result.compile_time:.2f}s" if use_compile else ""
                    print(f"  Done: {result.total_time:.2f}s{compile_info}, "
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
    header = f"{'Configuration':<40} {'Train(s)':>9} {'Compile':>9} {'Samp/s':>10} {'Train%':>8} {'Test%':>8}"
    print(header)
    print("-" * len(header))

    # Find fastest for relative comparison
    valid_times = [r.total_time for _, r in results if r is not None]
    fastest = min(valid_times) if valid_times else 1.0

    for config_name, result in results:
        if result is None:
            print(f"{config_name:<40} {'FAILED':>9}")
        else:
            compile_str = f"{result.compile_time:.2f}s" if result.compile_time > 0 else "-"
            print(f"{config_name:<40} {result.total_time:>9.2f} {compile_str:>9} {result.samples_per_sec:>10.0f} "
                  f"{result.train_acc*100:>7.1f}% {result.test_acc*100:>7.1f}%")

    print()
    print("-" * len(header))

    # Speedup comparison
    print()
    print("SPEEDUP vs FASTEST (training time only, excludes compile):")
    for config_name, result in results:
        if result is not None:
            speedup = fastest / result.total_time
            bar_len = int(speedup * 20)
            bar = "█" * bar_len
            print(f"  {config_name:<38} {bar} {speedup:.2f}x")

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
