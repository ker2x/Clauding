#!/bin/bash

# Vintage Oscilloscope Build and Launch Script
# This script builds the project and launches the app

echo "🔨 Building Vintage Oscilloscope..."

# Navigate to the project directory
cd "$(dirname "$0")"

# Build the project
if cmake --build build -j $(sysctl -n hw.ncpu); then
    echo "✅ Build successful!"

    # Code sign for development
    echo "🔐 Code signing for development..."
    codesign --sign - Oscilloscope.app

    # Update symlink if needed
    if [ ! -L "Oscilloscope.app" ] || [ ! -e "Oscilloscope.app" ]; then
        rm -f Oscilloscope.app
        ln -s build/Oscilloscope.app Oscilloscope.app
        echo "🔗 Created app symlink"
    fi

    echo "🚀 Launching Vintage Oscilloscope..."
    open Oscilloscope.app

    echo ""
    echo "🎯 If the app doesn't open, try running it directly:"
    echo "   cd $(pwd)"
    echo "   ./Oscilloscope.app/Contents/MacOS/Oscilloscope"
    echo ""
    echo "🎛️  Controls:"
    echo "   • A: Toggle AGC (Automatic Gain Control)"
    echo "   • ↑/↓: Adjust manual gain"
    echo "   • Cmd+E: Export screenshot"
    echo "   • Cmd+Q: Quit"
else
    echo "❌ Build failed!"
    exit 1
fi