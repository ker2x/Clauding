#import "AudioCaptureManager.h"
#import "AGC.h"
#import <AVFoundation/AVFoundation.h>
#import <AudioToolbox/AudioToolbox.h>
#import <stdatomic.h>

#define SAMPLE_RATE 48000.0
#define BUFFER_SIZE 256 // ~5.8ms latency at 44.1kHz

// Forward declare the C callbacks
static OSStatus inputCallback(void *inRefCon,
                              AudioUnitRenderActionFlags *ioActionFlags,
                              const AudioTimeStamp *inTimeStamp,
                              UInt32 inBusNumber, UInt32 inNumberFrames,
                              AudioBufferList *ioData);

static OSStatus
deviceChangeListener(AudioObjectID inObjectID, UInt32 inNumberAddresses,
                     const AudioObjectPropertyAddress *inAddresses,
                     void *inClientData);

@implementation AudioCaptureManager {
  AudioComponentInstance _audioUnit;
  LockFreeRingBuffer _ringBuffer;
  BOOL _isRunning;
  BOOL _permissionGranted;

  // AGC state (uses AGC.h module)
  AGCState _agcState;

  // Device capabilities
  float _actualSampleRate;
}

- (instancetype)init {

  self = [super init];
  if (self) {
    _isRunning = NO;
    _permissionGranted = NO;

    // Initialize AGC state using AGC module
    _agcEnabled = YES;
    agc_init(&_agcState);
    _gain = agc_get_gain(&_agcState);  // Sync public property

    // Initialize lock-free ring buffer (static allocation, cannot fail)
    lfringbuffer_init(&_ringBuffer);

    // SECURITY: Verify microphone permission before setting up audio capture
    // This is a defensive check to prevent bypass if this class is instantiated
    // without going through the proper permission flow in the app delegate.
    AVAuthorizationStatus status =
        [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeAudio];
    if (status != AVAuthorizationStatusAuthorized) {
      NSLog(@"⚠️ AudioCaptureManager: Microphone permission not granted "
            @"(status=%ld). Audio capture disabled.",
            (long)status);
      return self;
    }

    _permissionGranted = YES;
    NSLog(@"✓ AudioCaptureManager: Microphone permission verified");
    [self setupAudioUnit];
  }
  return self;
}

- (BOOL)permissionGranted {
  return _permissionGranted;
}

// ARC automatically calls [super dealloc] - suppress false warning
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wobjc-missing-super-calls"
- (void)dealloc {
  [self stop];

  // Remove device change listener
  AudioObjectPropertyAddress propertyAddress = {
      kAudioHardwarePropertyDefaultInputDevice, kAudioObjectPropertyScopeGlobal,
      kAudioObjectPropertyElementMain};
  AudioObjectRemovePropertyListener(kAudioObjectSystemObject, &propertyAddress,
                                    deviceChangeListener,
                                    (__bridge void *)self);

  if (_audioUnit) {
    AudioComponentInstanceDispose(_audioUnit);
  }
}
#pragma clang diagnostic pop

- (void)setupAudioUnit {
  // Find HAL output component
  AudioComponentDescription desc;
  desc.componentType = kAudioUnitType_Output;
  desc.componentSubType = kAudioUnitSubType_HALOutput;
  desc.componentManufacturer = kAudioUnitManufacturer_Apple;
  desc.componentFlags = 0;
  desc.componentFlagsMask = 0;

  AudioComponent component = AudioComponentFindNext(NULL, &desc);
  if (!component) {
    NSLog(@"Failed to find HAL output component");
    return;
  }

  OSStatus status = AudioComponentInstanceNew(component, &_audioUnit);
  if (status != noErr) {
    NSLog(@"Failed to create audio unit: %d", (int)status);
    return;
  }

  // CRITICAL: Enable input, disable output
  UInt32 enableIO = 1;
  status = AudioUnitSetProperty(_audioUnit, kAudioOutputUnitProperty_EnableIO,
                                kAudioUnitScope_Input,
                                1, // Input bus
                                &enableIO, sizeof(enableIO));
  NSLog(@"Enable input status: %d", (int)status);

  enableIO = 0;
  status = AudioUnitSetProperty(_audioUnit, kAudioOutputUnitProperty_EnableIO,
                                kAudioUnitScope_Output,
                                0, // Output bus
                                &enableIO, sizeof(enableIO));
  NSLog(@"Disable output status: %d", (int)status);

  // CRITICAL: Disable voice processing to get raw audio signal
  // This prevents Bluetooth headsets from applying automatic denoise/AGC
  UInt32 disableVoiceProcessing = 0;
  status = AudioUnitSetProperty(
      _audioUnit, kAUVoiceIOProperty_VoiceProcessingEnableAGC,
      kAudioUnitScope_Global, 0, &disableVoiceProcessing,
      sizeof(disableVoiceProcessing));
  if (status == noErr) {
    NSLog(@"✓ Disabled voice processing AGC");
  }

  // Also disable echo cancellation and other processing
  status =
      AudioUnitSetProperty(_audioUnit, kAUVoiceIOProperty_BypassVoiceProcessing,
                           kAudioUnitScope_Global, 0, &enableIO,
                           sizeof(enableIO)); // enableIO is already 0
  if (status == noErr) {
    NSLog(@"✓ Bypassed voice processing");
  }

  // Set default input device
  AudioObjectPropertyAddress propertyAddress = {
      kAudioHardwarePropertyDefaultInputDevice, kAudioObjectPropertyScopeGlobal,
      kAudioObjectPropertyElementMain};

  AudioDeviceID defaultInputDevice;
  UInt32 deviceSize = sizeof(defaultInputDevice);
  status =
      AudioObjectGetPropertyData(kAudioObjectSystemObject, &propertyAddress, 0,
                                 NULL, &deviceSize, &defaultInputDevice);

  if (status == noErr) {
    NSLog(@"Setting input device: %u", defaultInputDevice);
    status =
        AudioUnitSetProperty(_audioUnit, kAudioOutputUnitProperty_CurrentDevice,
                             kAudioUnitScope_Global, 0, &defaultInputDevice,
                             sizeof(defaultInputDevice));

    if (status != noErr) {
      NSLog(@"Failed to set device: %d", (int)status);
    }

    // Query the device's native format to get actual sample rate
    AudioStreamBasicDescription deviceFormat;
    UInt32 formatSize = sizeof(deviceFormat);
    status = AudioUnitGetProperty(_audioUnit, kAudioUnitProperty_StreamFormat,
                                  kAudioUnitScope_Input, 1, &deviceFormat,
                                  &formatSize);
    if (status == noErr) {
      _actualSampleRate = deviceFormat.mSampleRate;
      NSLog(@"✓ Detected device format: %.1f Hz, %u channel(s)",
            _actualSampleRate, (unsigned)deviceFormat.mChannelsPerFrame);

      // Validate sample rate is reasonable
      if (_actualSampleRate < 8000.0 || _actualSampleRate > 192000.0) {
        NSLog(@"⚠️ Unusual sample rate detected, falling back to 48000 Hz");
        _actualSampleRate = 48000.0;
      }
    } else {
      NSLog(@"⚠️ Failed to query device format, using default 48000 Hz");
      _actualSampleRate = 48000.0;
    }
  } else {
    // No default device, use fallback
    _actualSampleRate = 48000.0;
  }

  // CRITICAL: Set buffer size to 256 samples for low latency
  UInt32 bufferSize = BUFFER_SIZE;
  status = AudioUnitSetProperty(_audioUnit, kAudioDevicePropertyBufferFrameSize,
                                kAudioUnitScope_Global, 0, &bufferSize,
                                sizeof(bufferSize));
  if (status != noErr) {
    NSLog(@"Failed to set buffer size: %d", (int)status);
  } else {
    NSLog(@"✓ Buffer size set to %u samples (~%.1fms at %.1f Hz)", bufferSize,
          (float)bufferSize / _actualSampleRate * 1000.0f, _actualSampleRate);
  }

  // Set audio format using device's actual sample rate (mono, 32-bit float)
  AudioStreamBasicDescription audioFormat;
  audioFormat.mSampleRate = _actualSampleRate; // Use detected device rate
  audioFormat.mFormatID = kAudioFormatLinearPCM;
  audioFormat.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
  audioFormat.mFramesPerPacket = 1;
  audioFormat.mChannelsPerFrame = 1; // Mono
  audioFormat.mBitsPerChannel = 32;
  audioFormat.mBytesPerPacket = sizeof(float);
  audioFormat.mBytesPerFrame = sizeof(float);

  status = AudioUnitSetProperty(_audioUnit, kAudioUnitProperty_StreamFormat,
                                kAudioUnitScope_Output,
                                1, // Input bus output
                                &audioFormat, sizeof(audioFormat));
  if (status != noErr) {
    NSLog(@"Failed to set stream format: %d", (int)status);
  }

  // Set up input callback
  AURenderCallbackStruct callbackStruct;
  callbackStruct.inputProc = inputCallback;
  callbackStruct.inputProcRefCon = (__bridge void *)self;

  status = AudioUnitSetProperty(
      _audioUnit, kAudioOutputUnitProperty_SetInputCallback,
      kAudioUnitScope_Global, 1, &callbackStruct, sizeof(callbackStruct));
  if (status != noErr) {
    NSLog(@"Failed to set input callback: %d", (int)status);
  }

  // Initialize audio unit
  status = AudioUnitInitialize(_audioUnit);
  if (status != noErr) {
    NSLog(@"Failed to initialize audio unit: %d", (int)status);
  } else {
    NSLog(@"✓ AudioUnit HAL configured successfully");

    // Register for device change notifications
    AudioObjectPropertyAddress propertyAddress = {
        kAudioHardwarePropertyDefaultInputDevice,
        kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
    AudioObjectAddPropertyListener(kAudioObjectSystemObject, &propertyAddress,
                                   deviceChangeListener, (__bridge void *)self);
    NSLog(@"✓ Registered for device change notifications");
  }
}

- (void)handleDeviceChange {
  if (!_audioUnit)
    return;

  NSLog(@"🔄 Audio device changed, reinitializing AudioUnit...");

  BOOL wasRunning = _isRunning;

  // Stop if running
  if (wasRunning) {
    AudioOutputUnitStop(_audioUnit);
    _isRunning = NO;
    NSLog(@"  Stopped AudioUnit");
  }

  // Uninitialize the AudioUnit to force reconfiguration
  OSStatus status = AudioUnitUninitialize(_audioUnit);
  if (status != noErr) {
    NSLog(@"⚠️ Failed to uninitialize AudioUnit: %d", (int)status);
  } else {
    NSLog(@"  Uninitialized AudioUnit");
  }

  // Small delay to let macOS settle the device switch
  dispatch_after(
      dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.15 * NSEC_PER_SEC)),
      dispatch_get_main_queue(), ^{
        // Get the new default input device
        AudioObjectPropertyAddress devicePropertyAddress = {
            kAudioHardwarePropertyDefaultInputDevice,
            kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};

        AudioDeviceID newDevice;
        UInt32 deviceSize = sizeof(newDevice);
        OSStatus status = AudioObjectGetPropertyData(
            kAudioObjectSystemObject, &devicePropertyAddress, 0, NULL,
            &deviceSize, &newDevice);

        if (status == noErr) {
          NSLog(@"  Setting new device: %u", newDevice);

          // Set the new device on the AudioUnit
          status = AudioUnitSetProperty(
              _audioUnit, kAudioOutputUnitProperty_CurrentDevice,
              kAudioUnitScope_Global, 0, &newDevice, sizeof(newDevice));

          if (status != noErr) {
            NSLog(@"⚠️ Failed to set new device: %d", (int)status);
          }

          // Query the new device's format directly
          AudioObjectPropertyAddress formatPropertyAddress = {
              kAudioDevicePropertyStreamFormat, kAudioDevicePropertyScopeInput,
              kAudioObjectPropertyElementMain};

          AudioStreamBasicDescription deviceFormat;
          UInt32 formatSize = sizeof(deviceFormat);
          status =
              AudioObjectGetPropertyData(newDevice, &formatPropertyAddress, 0,
                                         NULL, &formatSize, &deviceFormat);

          if (status == noErr) {
            float oldRate = _actualSampleRate;
            _actualSampleRate = deviceFormat.mSampleRate;
            NSLog(@"✓ Device format: %.1f Hz, %u ch (was %.1f Hz)",
                  _actualSampleRate, (unsigned)deviceFormat.mChannelsPerFrame,
                  oldRate);

            // Validate sample rate
            if (_actualSampleRate < 8000.0 || _actualSampleRate > 192000.0) {
              NSLog(@"⚠️ Unusual sample rate, falling back to 48000 Hz");
              _actualSampleRate = 48000.0;
            }

            // Update the AudioUnit's stream format to match
            AudioStreamBasicDescription audioFormat;
            audioFormat.mSampleRate = _actualSampleRate;
            audioFormat.mFormatID = kAudioFormatLinearPCM;
            audioFormat.mFormatFlags =
                kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
            audioFormat.mFramesPerPacket = 1;
            audioFormat.mChannelsPerFrame = 1; // Mono
            audioFormat.mBitsPerChannel = 32;
            audioFormat.mBytesPerPacket = sizeof(float);
            audioFormat.mBytesPerFrame = sizeof(float);

            status = AudioUnitSetProperty(
                _audioUnit, kAudioUnitProperty_StreamFormat,
                kAudioUnitScope_Output, 1, &audioFormat, sizeof(audioFormat));
            if (status != noErr) {
              NSLog(@"⚠️ Failed to update stream format: %d", (int)status);
            }
          }
        }

        // Re-initialize the AudioUnit
        status = AudioUnitInitialize(_audioUnit);
        if (status != noErr) {
          NSLog(@"⚠️ Failed to reinitialize AudioUnit: %d", (int)status);
        } else {
          NSLog(@"  Re-initialized AudioUnit");
        }

        // Restart if it was running
        if (wasRunning) {
          status = AudioOutputUnitStart(_audioUnit);
          if (status == noErr) {
            _isRunning = YES;
            NSLog(@"✓ AudioUnit restarted with new device at %.1f Hz",
                  _actualSampleRate);
          } else {
            NSLog(@"⚠️ Failed to restart AudioUnit: %d", (int)status);
          }
        }
      });
}

- (void)start {
  if (_isRunning)
    return;

  // SECURITY: Refuse to start if permission was not granted
  if (!_permissionGranted) {
    NSLog(@"⚠️ AudioCaptureManager: Cannot start - microphone permission not "
          @"granted");
    return;
  }

  if (!_audioUnit) {
    NSLog(@"⚠️ AudioCaptureManager: Cannot start - audio unit not initialized");
    return;
  }

  OSStatus status = AudioOutputUnitStart(_audioUnit);
  if (status == noErr) {
    _isRunning = YES;
    NSLog(
        @"✓ Low-latency audio capture started (%u samples, ~%.1fms at %.1f Hz)",
        BUFFER_SIZE, (float)BUFFER_SIZE / _actualSampleRate * 1000.0f,
        _actualSampleRate);
  } else {
    NSLog(@"✗ Failed to start audio unit: %d", (int)status);
  }
}

- (void)stop {
  if (!_isRunning)
    return;

  AudioOutputUnitStop(_audioUnit);
  _isRunning = NO;
  NSLog(@"Audio capture stopped");
}

- (BOOL)isRunning {
  return _isRunning;
}

- (AudioComponentInstance)audioUnit {
  return _audioUnit;
}

- (LockFreeRingBuffer *)ringBuffer {
  return &_ringBuffer;
}

- (NSUInteger)getLatestSamples:(float *)outBuffer
                    maxSamples:(NSUInteger)maxSamples {
  // Read from lock-free ring buffer
  NSUInteger samplesRead =
      lfringbuffer_read(&_ringBuffer, outBuffer, (uint32_t)maxSamples);

  if (samplesRead == 0)
    return 0;

  // Sync manual gain if AGC is disabled
  if (!_agcEnabled) {
    agc_set_manual_gain(&_agcState, _gain);
  }

  // Process samples with AGC module (handles peak detection, gain smoothing, soft clipping)
  agc_process(&_agcState, outBuffer, (uint32_t)samplesRead, _agcEnabled ? 1 : 0);

  // Sync public gain property for UI display and smooth transition to manual mode
  _gain = agc_get_gain(&_agcState);

  return samplesRead;
}

@end

// ZERO-LOCK AUDIO CALLBACK
// CRITICAL: This runs on high-priority audio thread
// NO locks, NO @synchronized, NO Obj-C message sends (except ring buffer)
// NO allocations, NO logging (in production)
static OSStatus inputCallback(void *inRefCon,
                              AudioUnitRenderActionFlags *ioActionFlags,
                              const AudioTimeStamp *inTimeStamp,
                              UInt32 inBusNumber, UInt32 inNumberFrames,
                              AudioBufferList *ioData) {
  AudioCaptureManager *manager = (__bridge AudioCaptureManager *)inRefCon;

  // Track frame count (exposed via ivar via atomic accessor if we added it, but
  // let's stick to basics) Note: We access ivar directly here which is
  // technically breaking encapsulation but we need speed and we're inside the
  // implementation block context effectively if we were a method, but we're a C
  // function. Actually we can't access ivars from C function easily without
  // public accessors or KVC (too slow). Let's just trust updateCallbackStatus
  // for now.

  // Use max buffer size to avoid VLA. Increased to 4096 to prevent overflow if
  // OS requests more frames (safety margin).
  const UInt32 MAX_FRAMES = 4096;
  float tempBuffer[MAX_FRAMES];

  // CRITICAL FIX: Clamp number of frames to buffer size
  // The OS acts as master and can request more frames than expected.
  // We must ensure we don't write past the end of our stack buffer.
  if (inNumberFrames > MAX_FRAMES) {
    inNumberFrames = MAX_FRAMES;
  }

  // Set up buffer list for rendering
  AudioBufferList bufferList;
  bufferList.mNumberBuffers = 1;
  bufferList.mBuffers[0].mNumberChannels = 1;
  bufferList.mBuffers[0].mDataByteSize = inNumberFrames * sizeof(float);
  bufferList.mBuffers[0].mData = tempBuffer;

  // Render audio from INPUT BUS (bus 1)
  OSStatus status =
      AudioUnitRender([manager audioUnit], ioActionFlags, inTimeStamp,
                      1, // Input bus
                      inNumberFrames, &bufferList);

  static int callbackCount = 0;
  if (++callbackCount % 200 == 1) {
    NSLog(@"DEBUG callback: status=%d, frames=%u", (int)status,
          (unsigned)inNumberFrames);
  }

  if (status == noErr) {
    // Write to lock-free ring buffer (uses only atomics, real-time safe)
    lfringbuffer_write([manager ringBuffer], tempBuffer, inNumberFrames);
  }

  return noErr; // Always return success to keep audio thread alive
}

// Device Change Listener
// Called when the default input device changes
static OSStatus
deviceChangeListener(AudioObjectID inObjectID, UInt32 inNumberAddresses,
                     const AudioObjectPropertyAddress *inAddresses,
                     void *inClientData) {
  AudioCaptureManager *manager = (__bridge AudioCaptureManager *)inClientData;

  // Dispatch to main thread to handle the change
  dispatch_async(dispatch_get_main_queue(), ^{
    [manager handleDeviceChange];
  });

  return noErr;
}
