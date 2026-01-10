"""
CPU-specific fixes for intermittent conv2d slowdown.

On CPU (not GPU), the slowdown is likely caused by:
1. CPU frequency scaling (thermal throttling or power management)
2. Background processes competing for CPU
3. PyTorch thread contention
4. Memory allocation/deallocation overhead
5. Cache misses or memory bandwidth bottlenecks
"""

import torch
import os
import psutil


def fix_1_set_thread_count():
    """
    Set PyTorch thread count to match physical CPU cores.

    Too many threads can cause contention. Too few wastes parallelism.
    Best practice: num_physical_cores or num_physical_cores - 1
    """
    physical_cores = psutil.cpu_count(logical=False)

    # Set to physical cores (not hyperthreaded logical cores)
    torch.set_num_threads(physical_cores)

    # Also set for underlying libraries
    os.environ['OMP_NUM_THREADS'] = str(physical_cores)
    os.environ['MKL_NUM_THREADS'] = str(physical_cores)
    os.environ['NUMEXPR_NUM_THREADS'] = str(physical_cores)

    print(f"✓ PyTorch threads set to {physical_cores} (physical cores)")
    print(f"  Total logical cores: {psutil.cpu_count(logical=True)}")


def fix_2_disable_profiler():
    """
    Disable PyTorch profiler and autograd anomaly detection.
    These can cause intermittent overhead.
    """
    torch.autograd.set_detect_anomaly(False)
    print("✓ Autograd anomaly detection disabled")


def fix_3_preallocate_tensors():
    """
    Preallocate common tensor sizes to reduce allocation overhead.

    Call this after creating the agent, passing the batch size.
    """
    def preallocate(agent, batch_size, state_shape):
        device = agent.device

        # Preallocate common tensor shapes
        dummy_states = torch.randn(batch_size, *state_shape).to(device)
        dummy_actions = torch.randn(batch_size, agent.action_dim).to(device)

        # Do a few dummy forward passes to warm up allocator
        with torch.no_grad():
            for _ in range(3):
                _ = agent.critic_1(dummy_states, dummy_actions)
                _ = agent.critic_2(dummy_states, dummy_actions)
                _ = agent.actor(dummy_states)

        del dummy_states, dummy_actions

        print("✓ Tensor allocator warmed up")

    return preallocate


def fix_4_set_cpu_affinity():
    """
    Pin Python process to specific CPU cores to reduce context switching.

    This prevents the OS from moving the process between cores, which
    can cause cache misses and variable performance.
    """
    try:
        process = psutil.Process()
        physical_cores = psutil.cpu_count(logical=False)

        # Use first N physical cores (avoid hyperthreading)
        cores_to_use = list(range(physical_cores))
        process.cpu_affinity(cores_to_use)

        print(f"✓ Process pinned to cores: {cores_to_use}")
    except (AttributeError, OSError) as e:
        print(f"✗ Could not set CPU affinity: {e}")


def fix_5_increase_process_priority():
    """
    Increase process priority to reduce impact of background tasks.

    WARNING: Use carefully. May affect system responsiveness.
    """
    try:
        process = psutil.Process()
        if os.name == 'posix':  # Linux/macOS
            # Nice values: -20 (highest) to 19 (lowest)
            # Default is 0, we use -5 for slightly higher priority
            os.nice(-5)
            print(f"✓ Process priority increased (nice value: -5)")
        elif os.name == 'nt':  # Windows
            process.nice(psutil.HIGH_PRIORITY_CLASS)
            print(f"✓ Process priority set to HIGH")
    except (PermissionError, OSError) as e:
        print(f"✗ Could not increase priority (may need sudo/admin): {e}")


def fix_6_check_thermal_throttling():
    """
    Check CPU temperature and frequency to detect thermal throttling.

    If temperature is high or frequency is reduced, thermal throttling
    may be causing the intermittent slowdown.
    """
    try:
        # Check CPU frequency
        freq = psutil.cpu_freq()
        if freq:
            print(f"  CPU Frequency:")
            print(f"    Current: {freq.current:.0f} MHz")
            print(f"    Min:     {freq.min:.0f} MHz")
            print(f"    Max:     {freq.max:.0f} MHz")

            if freq.current < freq.max * 0.8:
                print(f"  ⚠️  WARNING: CPU running at reduced frequency!")
                print(f"     This suggests thermal throttling or power saving.")

        # Check temperature (requires psutil sensors, not always available)
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    for entry in entries:
                        if entry.current > 80:
                            print(f"  ⚠️  WARNING: {name} at {entry.current}°C (hot!)")
        except AttributeError:
            print(f"  (Temperature monitoring not available on this system)")

    except Exception as e:
        print(f"  Could not check thermal status: {e}")


def fix_7_disable_turbo_boost():
    """
    Disable CPU turbo boost for consistent performance.

    Turbo boost can cause frequency fluctuations leading to variable
    performance. Disabling it trades peak performance for consistency.

    NOTE: This requires system-level permissions and is OS-specific.
    """
    print("  Manual action required to disable turbo boost:")
    print("  macOS: sudo nvram boot-args='serverperfmode=1'")
    print("  Linux: echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo")
    print("  Windows: Power Options > Processor power management > Max state = 99%")


def diagnose_cpu():
    """Print comprehensive CPU diagnostics."""
    print("\n" + "="*70)
    print("CPU DIAGNOSTICS")
    print("="*70)

    # CPU info
    print(f"\nCPU Info:")
    print(f"  Physical cores: {psutil.cpu_count(logical=False)}")
    print(f"  Logical cores:  {psutil.cpu_count(logical=True)}")
    print(f"  PyTorch threads: {torch.get_num_threads()}")

    # Frequency and thermal
    fix_6_check_thermal_throttling()

    # Memory
    mem = psutil.virtual_memory()
    print(f"\nMemory:")
    print(f"  Total:     {mem.total / 1024**3:.1f} GB")
    print(f"  Available: {mem.available / 1024**3:.1f} GB")
    print(f"  Used:      {mem.percent:.1f}%")

    # CPU usage by other processes
    print(f"\nTop CPU consumers:")
    processes = []
    for proc in psutil.process_iter(['name', 'cpu_percent']):
        try:
            processes.append((proc.info['name'], proc.info['cpu_percent']))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    top_processes = sorted(processes, key=lambda x: x[1] or 0, reverse=True)[:5]
    for name, cpu_pct in top_processes:
        if cpu_pct and cpu_pct > 0:
            print(f"    {name}: {cpu_pct:.1f}%")

    print("="*70 + "\n")


def apply_all_fixes(agent, batch_size, state_shape):
    """
    Apply all recommended CPU fixes.

    Call this after creating the agent in train.py:
        from fix_cpu_slowdown import apply_all_fixes
        apply_all_fixes(agent, args.batch_size, state_shape)
    """
    print("\n" + "="*70)
    print("APPLYING CPU PERFORMANCE FIXES")
    print("="*70 + "\n")

    # Diagnose first
    diagnose_cpu()

    print("Applying fixes...\n")

    # Fix 1: Set thread count (MOST IMPORTANT)
    fix_1_set_thread_count()

    # Fix 2: Disable profiler overhead
    fix_2_disable_profiler()

    # Fix 3: Preallocate tensors
    preallocate = fix_3_preallocate_tensors()
    preallocate(agent, batch_size, state_shape)

    # Fix 4: CPU affinity (optional, can help)
    # fix_4_set_cpu_affinity()

    # Fix 5: Process priority (optional, use with caution)
    # fix_5_increase_process_priority()

    print("\n" + "="*70)
    print("Fixes applied! Monitor training for improvements.")
    print("="*70 + "\n")


if __name__ == "__main__":
    print(__doc__)
    print("\nTo use these fixes, add to train.py after creating agent:")
    print("\n  from fix_cpu_slowdown import apply_all_fixes")
    print("  apply_all_fixes(agent, args.batch_size, state_shape)")
    print("\nOr diagnose current CPU state:")
    print("\n  from fix_cpu_slowdown import diagnose_cpu")
    print("  diagnose_cpu()")
