#!/bin/bash

if [ ! -d "WindTunnel.app" ]; then
    echo "App not built yet. Building..."
    ./build.sh
fi

if [ -d "WindTunnel.app" ]; then
    echo "🌪️  Launching Virtual Wind Tunnel..."
    open WindTunnel.app
else
    echo "✗ Build failed or app not found"
    exit 1
fi
