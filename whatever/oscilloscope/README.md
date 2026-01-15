# Vintage Oscilloscope - Quick Reference

## Build & Run

### Option 1: CMake (Recommended)
```bash
cd /Users/ker/PycharmProjects/Clauding/whatever/oscilloscope
cmake -S . -B build
cmake --build build -j $(sysctl -n hw.ncpu)
open build/Oscilloscope.app
```

### Option 2: Shell Script (Quick)
```bash
cd /Users/ker/PycharmProjects/Clauding/whatever/oscilloscope
./compile.sh
open build/Oscilloscope.app
```

## Features
- ✅ Real-time microphone visualization
- ✅ Vintage green CRT phosphor glow
- ✅ Screen curvature, scanlines, persistence
- ✅ 60 FPS Metal rendering
- ✅ 60 FPS Metal rendering
- ✅ Automatic Gain Control (AGC) or Manual Gain
- ✅ Cmd+Q to quit

## Controls
- **A**: Toggle Automatic Gain Control (AGC) on/off
- **Up Arrow**: Increase Gain (switches to Manual)
- **Down Arrow**: Decrease Gain (switches to Manual)


## Key Files
- `AudioCaptureManager.mm` - AVAudioEngine audio capture
- `Shaders.metal` - CRT shader effects
- `OscilloscopeView.mm` - Metal rendering
- `main.mm` - App and menu setup

## Troubleshooting
**No waveform?**
- Check microphone permission in System Settings > Privacy & Security
- Make sure you're speaking/playing audio

**Build fails?**
- Ensure Xcode command line tools installed: `xcode-select --install`

## Technical Notes
- Switched from AudioUnit to AVAudioEngine for reliability

- Ring buffer size: 4096 samples
- Render buffer: 512 samples at 60Hz
