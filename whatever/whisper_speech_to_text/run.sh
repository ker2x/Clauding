#!/bin/bash
set -e
cd "$(dirname "$0")"
./build.sh
echo "Launching..."
open WhisperSpeechToText.app
