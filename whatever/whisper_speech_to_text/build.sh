#!/bin/bash
# Build script for Whisper Speech to Text (WhisperKit)

set -e

APP_NAME="WhisperSpeechToText"
BUNDLE_NAME="${APP_NAME}.app"

echo "Building ${APP_NAME} with WhisperKit..."
echo "(First build will download dependencies - this may take a while)"
echo ""

# Build with Swift Package Manager
swift build -c release

# Create app bundle
rm -rf "${BUNDLE_NAME}"
mkdir -p "${BUNDLE_NAME}/Contents/MacOS"
mkdir -p "${BUNDLE_NAME}/Contents/Resources"

# Copy Info.plist
cp Info.plist "${BUNDLE_NAME}/Contents/"

# Copy executable
cp .build/release/WhisperSpeechToText "${BUNDLE_NAME}/Contents/MacOS/"

echo ""
echo "Build complete: ${BUNDLE_NAME}"
echo ""
echo "Run with: open ${BUNDLE_NAME}"
echo ""
echo "NOTE: First run will download the Whisper model (~142MB for 'base')."
echo "Models are cached in ~/Library/Caches/com.argmaxinc.whisperkit/"
