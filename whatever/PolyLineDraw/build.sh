#!/bin/bash
cd "$(dirname "$0")"
rm -rf build
mkdir -p build/PolyLineDraw.app/Contents/MacOS
mkdir -p build/PolyLineDraw.app/Contents/Resources

# Compile binary
swiftc *.swift -o build/PolyLineDraw.app/Contents/MacOS/PolyLineDraw -sdk $(xcrun --show-sdk-path --sdk macosx) -target arm64-apple-macosx26.0

# Copy Info.plist
cp Info.plist build/PolyLineDraw.app/Contents/Info.plist

echo "Build complete: build/PolyLineDraw.app"
