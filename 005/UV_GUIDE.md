# Using `uv` - Fast Python Package Management

`uv` is a blazingly fast Python package installer and resolver written in Rust by Astral (creators of Ruff). It's 10-100x faster than `pip`.

## Installation Performance

When installing requirements.txt (45 packages, including PyTorch):
```
✓ Resolved 45 packages in 35ms (pip takes several seconds)
✓ Installed 45 packages in 5.67s
✓ Total time: 3m 44s (mostly downloading 3.5GB of CUDA libraries)
```

## Quick Start

### Create Virtual Environment
```bash
# Instead of: python -m venv .venv
uv venv .venv

# Activate it normally
source .venv/bin/activate
```

### Install Packages
```bash
# Instead of: pip install package
uv pip install package

# Instead of: pip install -r requirements.txt
uv pip install -r requirements.txt

# Install specific version
uv pip install numpy==2.3.4
```

### Other Commands
```bash
# List installed packages
uv pip list

# Uninstall package
uv pip uninstall package

# Freeze requirements
uv pip freeze > requirements.txt
```

## Key Advantages

1. **Speed**: 10-100x faster package resolution
2. **Disk Usage**: Shares cached packages across projects
3. **Reliability**: Better dependency resolution
4. **Drop-in**: Compatible with pip/requirements.txt

## Why It's Faster

- Written in Rust (not Python)
- Parallel downloads
- Smart caching
- Optimized dependency resolver

## When to Use `uv` vs `pip`

**Use `uv` for:**
- Installing dependencies (much faster)
- Creating virtual environments
- Resolving complex dependency trees

**Use `pip` for:**
- Compatibility with older systems
- When `uv` is not available

## Example Workflow

```bash
cd /home/user/Clauding/005

# Create environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Run your code
python play_human_gui.py
```

## Notes

- `uv` is a drop-in replacement - your existing requirements.txt files work unchanged
- Virtual environments created by `uv` are standard Python venvs
- Package resolution is typically **35ms** vs several seconds with pip
- Caches packages in `~/.cache/uv` for reuse across projects

## Learn More

- GitHub: https://github.com/astral-sh/uv
- Docs: https://docs.astral.sh/uv/
