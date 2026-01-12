#!/bin/bash

echo "🌪️  Building Virtual Wind Tunnel..."

# Create app bundle structure
APP_NAME="WindTunnel.app"
CONTENTS="$APP_NAME/Contents"
MACOS="$CONTENTS/MacOS"
RESOURCES="$CONTENTS/Resources"

echo "Creating app bundle structure..."
mkdir -p "$MACOS"
mkdir -p "$RESOURCES"

# Compile Metal shaders to metallib
echo "Compiling Metal shaders..."
xcrun -sdk macosx metal -c Shaders.metal -o Shaders.air 2>&1
if [ $? -ne 0 ]; then
    echo "⚠️  Metal compiler not found. Trying alternative method..."
    # Metal shaders will be compiled at runtime from source
    cp Shaders.metal "$RESOURCES/default.metal"
else
    xcrun -sdk macosx metallib Shaders.air -o "$RESOURCES/default.metallib"
    rm -f Shaders.air
    echo "✓ Metal shaders compiled"
fi

# Copy config
cp config.json "$RESOURCES/"

# Create Info.plist
cat > "$CONTENTS/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>WindTunnel</string>
    <key>CFBundleIdentifier</key>
    <string>com.cfd.windtunnel</string>
    <key>CFBundleName</key>
    <string>WindTunnel</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

# Compile Swift application
echo "Compiling Swift application..."
swiftc WindTunnel.swift \
    -parse-as-library \
    -o "$MACOS/WindTunnel" \
    -framework MetalKit \
    -framework SwiftUI \
    -framework AppKit \
    -framework Metal \
    -target arm64-apple-macos13.0

if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    echo ""
    echo "To run the wind tunnel:"
    echo "  open $APP_NAME"
    echo ""
    echo "Or directly:"
    echo "  ./$MACOS/WindTunnel"
else
    echo "✗ Build failed"
    exit 1
fi
