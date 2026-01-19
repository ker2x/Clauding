#!/bin/bash
# Build script for Speech to Text application
# Creates a proper macOS app bundle with required privacy permissions

set -e

APP_NAME="SpeechToText"
BUNDLE_NAME="${APP_NAME}.app"

echo "Building ${APP_NAME}..."

# Clean previous build
rm -rf "${BUNDLE_NAME}"

# Create app bundle structure
mkdir -p "${BUNDLE_NAME}/Contents/MacOS"
mkdir -p "${BUNDLE_NAME}/Contents/Resources"

# Copy Info.plist
cp Info.plist "${BUNDLE_NAME}/Contents/"

# Compile Swift source
# Frameworks:
# - AppKit: GUI components
# - Speech: SFSpeechRecognizer for speech-to-text
# - AVFoundation: AVAudioEngine for microphone input
swiftc main.swift \
    -o "${BUNDLE_NAME}/Contents/MacOS/${APP_NAME}" \
    -framework AppKit \
    -framework Speech \
    -framework AVFoundation \
    -O \
    -whole-module-optimization

echo "Build complete: ${BUNDLE_NAME}"
echo ""
echo "Run with: open ${BUNDLE_NAME}"
echo "Or: ./${BUNDLE_NAME}/Contents/MacOS/${APP_NAME}"
