#!/bin/bash
cd "$(dirname "$0")"
rm -rf build/ *.so *.egg-info
pip install . --no-build-isolation --force-reinstall --no-deps
