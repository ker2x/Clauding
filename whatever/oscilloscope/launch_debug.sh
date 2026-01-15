#!/bin/bash
# Launch oscilloscope and capture NSLog output
killall Oscilloscope 2>/dev/null
sleep 0.5
./build/Oscilloscope.app/Contents/MacOS/Oscilloscope 2>&1 | grep -E "(AudioCaptureManager|Detected device|Buffer size set|audio capture started|Enable|Disable)" | head -10
