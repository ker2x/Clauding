#!/bin/bash
# Build and run Speech to Text application
set -e

cd "$(dirname "$0")"

./build.sh
echo "Launching..."
open SpeechToText.app
