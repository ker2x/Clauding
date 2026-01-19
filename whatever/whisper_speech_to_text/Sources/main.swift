// Whisper Speech-to-Text Application
// Uses WhisperKit (Apple's official Swift package) for local transcription
//
// Frameworks/Libraries:
// - WhisperKit: Whisper model inference on Apple Silicon
// - AVFoundation: Audio capture from microphone
// - AppKit: Native macOS GUI
//
// How it works:
// 1. WhisperKit downloads and loads Whisper model (cached locally)
// 2. AVAudioEngine captures microphone audio
// 3. Audio is buffered and sent to WhisperKit for transcription
// 4. Results displayed in real-time

import Cocoa
import AVFoundation
import WhisperKit

// MARK: - Whisper Transcription Manager

@MainActor
class WhisperTranscriptionManager: ObservableObject {
    private var whisperKit: WhisperKit?
    private var audioEngine: AVAudioEngine?
    private var audioBuffer: [Float] = []
    private let bufferLock = NSLock()

    // Audio settings - Whisper expects 16kHz mono
    private let sampleRate: Double = 16000
    var transcriptionInterval: TimeInterval = 3.0  // Process every N seconds (configurable)
    private var lastTranscriptionTime: Date = Date()
    private var isTranscribing: Bool = false

    var isModelLoaded: Bool { whisperKit != nil }
    var isListening: Bool = false
    var selectedLanguage: String = "fr"  // Default to French

    // Callbacks
    var onStatusUpdate: ((String) -> Void)?
    var onTranscription: ((String) -> Void)?
    var onError: ((String) -> Void)?

    /// Available Whisper models (smallest to largest)
    /// WhisperKit handles the full model path internally
    static let availableModels = [
        "tiny",      // ~75MB, fastest, least accurate
        "base",      // ~142MB, good balance
        "small",     // ~466MB, better accuracy
        "medium",    // ~1.5GB, high accuracy
        "large-v3"   // ~3GB, best accuracy
    ]

    /// Display names for UI
    static let modelDisplayNames = [
        "tiny (~75MB)",
        "base (~142MB)",
        "small (~466MB)",
        "medium (~1.5GB)",
        "large-v3 (~3GB)"
    ]

    /// Available languages with their Whisper codes
    static let availableLanguages: [(code: String, name: String)] = [
        ("en", "English"),
        ("fr", "French"),
        ("de", "German"),
        ("ja", "Japanese")
    ]

    /// Load Whisper model
    func loadModel(name: String = "base") async {
        onStatusUpdate?("Downloading model '\(name)'...")

        do {
            // WhisperKit downloads and caches models automatically
            // Models cached in ~/.cache/huggingface/hub/
            whisperKit = try await WhisperKit(
                model: name,
                computeOptions: .init(
                    melCompute: .cpuAndGPU,
                    audioEncoderCompute: .cpuAndNeuralEngine,
                    textDecoderCompute: .cpuAndNeuralEngine
                ),
                verbose: true,
                logLevel: .debug
            )

            onStatusUpdate?("Model '\(name)' ready")
        } catch {
            onError?("Failed to load model: \(error.localizedDescription)")
            onStatusUpdate?("Model load failed")
        }
    }

    /// Start listening to microphone
    func startListening() throws {
        guard whisperKit != nil else {
            throw NSError(domain: "Whisper", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Model not loaded"])
        }

        audioEngine = AVAudioEngine()
        guard let audioEngine = audioEngine else { return }

        let inputNode = audioEngine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        // Install tap to capture audio
        inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { [weak self] buffer, _ in
            self?.processAudioBuffer(buffer, inputFormat: inputFormat)
        }

        audioEngine.prepare()
        try audioEngine.start()
        isListening = true

        onStatusUpdate?("Listening...")
    }

    /// Stop listening
    func stopListening() {
        isListening = false
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine = nil

        // Process any remaining audio
        bufferLock.lock()
        let remainingSamples = audioBuffer
        audioBuffer.removeAll()
        bufferLock.unlock()

        if !remainingSamples.isEmpty {
            Task {
                await transcribe(samples: remainingSamples)
            }
        }

        onStatusUpdate?("Stopped")
    }

    /// Process incoming audio buffer
    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer, inputFormat: AVAudioFormat) {
        guard let floatData = buffer.floatChannelData else { return }

        let frameCount = Int(buffer.frameLength)
        var samples = [Float](repeating: 0, count: frameCount)

        // Copy first channel
        for i in 0..<frameCount {
            samples[i] = floatData[0][i]
        }

        // Resample to 16kHz if needed
        if inputFormat.sampleRate != sampleRate {
            samples = resample(samples, from: inputFormat.sampleRate, to: sampleRate)
        }

        bufferLock.lock()
        audioBuffer.append(contentsOf: samples)

        // Check if it's time to transcribe (every 3 seconds)
        let timeSinceLastTranscription = Date().timeIntervalSince(lastTranscriptionTime)
        let hasEnoughAudio = audioBuffer.count >= Int(sampleRate * 1.0)  // At least 1 second

        if timeSinceLastTranscription >= transcriptionInterval && hasEnoughAudio && !isTranscribing {
            let samplesToProcess = audioBuffer
            audioBuffer.removeAll()
            lastTranscriptionTime = Date()
            isTranscribing = true
            bufferLock.unlock()

            Task { @MainActor in
                await self.transcribe(samples: samplesToProcess)
                self.isTranscribing = false
                // Check if new audio arrived while transcribing
                self.checkForPendingTranscription()
            }
            return
        }

        bufferLock.unlock()
    }

    /// Simple linear interpolation resampling
    private func resample(_ samples: [Float], from sourceRate: Double, to targetRate: Double) -> [Float] {
        let ratio = targetRate / sourceRate
        let newCount = Int(Double(samples.count) * ratio)
        var resampled = [Float](repeating: 0, count: newCount)

        for i in 0..<newCount {
            let srcIdx = Double(i) / ratio
            let idx0 = Int(srcIdx)
            let idx1 = min(idx0 + 1, samples.count - 1)
            let frac = Float(srcIdx - Double(idx0))
            resampled[i] = samples[idx0] * (1 - frac) + samples[idx1] * frac
        }

        return resampled
    }

    /// Apply Automatic Gain Control to normalize audio levels
    private func applyAGC(_ samples: [Float]) -> [Float] {
        guard !samples.isEmpty else { return samples }

        // Calculate RMS level
        let rms = sqrt(samples.reduce(0) { $0 + $1 * $1 } / Float(samples.count))
        guard rms > 0.0001 else { return samples }  // Avoid division by near-zero

        // Target RMS level (Whisper works well with normalized audio around 0.1-0.2)
        let targetRMS: Float = 0.15

        // Calculate gain (with limits to prevent extreme amplification)
        let gain = min(max(targetRMS / rms, 0.5), 10.0)

        // Apply gain with soft clipping to prevent distortion
        return samples.map { sample in
            let amplified = sample * gain
            // Soft clip using tanh for natural limiting
            return tanh(amplified)
        }
    }

    /// Transcribe audio samples
    private func transcribe(samples: [Float]) async {
        guard let whisperKit = whisperKit else { return }

        // Check if audio has enough energy (avoid transcribing silence)
        let rms = sqrt(samples.reduce(0) { $0 + $1 * $1 } / Float(max(samples.count, 1)))
        guard rms > 0.003 else {
            // Too quiet, skip transcription to avoid hallucinations
            return
        }

        // Apply AGC to normalize audio levels
        let normalizedSamples = applyAGC(samples)

        do {
            // Configure decoding with fixed language to prevent switching
            let decodingOptions = DecodingOptions(
                language: selectedLanguage,  // Force language
                skipSpecialTokens: false,    // Keep tags like [MUSIC], [APPLAUSE]
                withoutTimestamps: true,     // We don't need timestamps
                suppressBlank: true          // Suppress blank outputs
            )

            let results = try await whisperKit.transcribe(
                audioArray: normalizedSamples,
                decodeOptions: decodingOptions
            )

            if let text = results.first?.text, !text.isEmpty {
                var cleanedText = text.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)

                // Only remove BLANK_AUDIO tokens, keep other tags like [MUSIC], [APPLAUSE]
                let blankTokens = ["[BLANK_AUDIO]", "(BLANK_AUDIO)"]
                for token in blankTokens {
                    cleanedText = cleanedText.replacingOccurrences(of: token, with: "")
                }
                cleanedText = cleanedText.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)

                // Skip if only whitespace or very short (likely noise)
                if cleanedText.count > 1 {
                    onTranscription?(cleanedText)
                }
            }
        } catch {
            onError?("Transcription error: \(error.localizedDescription)")
        }
    }

    /// Check if pending audio needs transcription after current one completes
    private func checkForPendingTranscription() {
        guard !isTranscribing else { return }

        bufferLock.lock()
        let hasEnoughAudio = audioBuffer.count >= Int(sampleRate * 1.0)
        if hasEnoughAudio {
            let samplesToProcess = audioBuffer
            audioBuffer.removeAll()
            lastTranscriptionTime = Date()
            isTranscribing = true
            bufferLock.unlock()

            Task { @MainActor in
                await self.transcribe(samples: samplesToProcess)
                self.isTranscribing = false
                // Recursively check for more pending audio
                self.checkForPendingTranscription()
            }
        } else {
            bufferLock.unlock()
        }
    }
}

// MARK: - Main Window Controller

class MainWindowController: NSWindowController {
    private var whisperManager: WhisperTranscriptionManager!

    private var textView: NSTextView!
    private var statusLabel: NSTextField!
    private var modelPopup: NSPopUpButton!
    private var languagePopup: NSPopUpButton!
    private var intervalSlider: NSSlider!
    private var intervalLabel: NSTextField!
    private var fontSizeSlider: NSSlider!
    private var fontSizeLabel: NSTextField!
    private var loadModelButton: NSButton!
    private var toggleButton: NSButton!
    private var copyButton: NSButton!
    private var saveButton: NSButton!
    private var clearButton: NSButton!

    // Status bar elements
    private var statusBar: NSView!
    private var recordingIndicator: NSView!
    private var statusBarLabel: NSTextField!

    private var fullTranscript = ""
    private var currentFontSize: CGFloat = 14
    private var currentModelName: String = ""

    convenience init() {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 800, height: 600),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "Whisper Speech to Text"
        window.center()
        window.minSize = NSSize(width: 500, height: 400)

        self.init(window: window)

        whisperManager = WhisperTranscriptionManager()
        setupUI()
        setupCallbacks()
    }

    private func setupUI() {
        guard let window = window else { return }

        let contentView = NSView(frame: window.contentView!.bounds)
        contentView.autoresizingMask = [.width, .height]
        window.contentView = contentView

        // Status label
        statusLabel = NSTextField(labelWithString: "Select a model and click Load")
        statusLabel.font = NSFont.systemFont(ofSize: 14, weight: .medium)
        statusLabel.textColor = .secondaryLabelColor
        statusLabel.translatesAutoresizingMaskIntoConstraints = false
        contentView.addSubview(statusLabel)

        // Model selector
        modelPopup = NSPopUpButton(frame: .zero, pullsDown: false)
        modelPopup.translatesAutoresizingMaskIntoConstraints = false
        for displayName in WhisperTranscriptionManager.modelDisplayNames {
            modelPopup.addItem(withTitle: displayName)
        }
        modelPopup.selectItem(at: 2) // Default to "small"
        contentView.addSubview(modelPopup)

        // Language selector
        languagePopup = NSPopUpButton(frame: .zero, pullsDown: false)
        languagePopup.translatesAutoresizingMaskIntoConstraints = false
        languagePopup.target = self
        languagePopup.action = #selector(languageChanged)
        for (index, lang) in WhisperTranscriptionManager.availableLanguages.enumerated() {
            languagePopup.addItem(withTitle: lang.name)
            // Default to French (index 1)
            if lang.code == "fr" {
                languagePopup.selectItem(at: index)
            }
        }
        contentView.addSubview(languagePopup)

        // Interval slider (1-10 seconds)
        intervalLabel = NSTextField(labelWithString: "Interval: 3s")
        intervalLabel.font = NSFont.systemFont(ofSize: 12)
        intervalLabel.translatesAutoresizingMaskIntoConstraints = false
        contentView.addSubview(intervalLabel)

        intervalSlider = NSSlider(value: 3.0, minValue: 1.0, maxValue: 10.0, target: self, action: #selector(intervalChanged))
        intervalSlider.translatesAutoresizingMaskIntoConstraints = false
        intervalSlider.numberOfTickMarks = 10
        intervalSlider.allowsTickMarkValuesOnly = true
        contentView.addSubview(intervalSlider)

        // Font size slider (10-32 pt)
        fontSizeLabel = NSTextField(labelWithString: "Font: 14pt")
        fontSizeLabel.font = NSFont.systemFont(ofSize: 12)
        fontSizeLabel.translatesAutoresizingMaskIntoConstraints = false
        contentView.addSubview(fontSizeLabel)

        fontSizeSlider = NSSlider(value: 14.0, minValue: 10.0, maxValue: 32.0, target: self, action: #selector(fontSizeChanged))
        fontSizeSlider.translatesAutoresizingMaskIntoConstraints = false
        fontSizeSlider.numberOfTickMarks = 12
        fontSizeSlider.allowsTickMarkValuesOnly = true
        contentView.addSubview(fontSizeSlider)

        // Load model button
        loadModelButton = NSButton(title: "Load Model", target: self, action: #selector(loadModel))
        loadModelButton.bezelStyle = .rounded
        loadModelButton.translatesAutoresizingMaskIntoConstraints = false
        contentView.addSubview(loadModelButton)

        // Toggle button
        toggleButton = NSButton(title: "Start Listening", target: self, action: #selector(toggleListening))
        toggleButton.bezelStyle = .rounded
        toggleButton.translatesAutoresizingMaskIntoConstraints = false
        toggleButton.isEnabled = false
        contentView.addSubview(toggleButton)

        // Copy button
        copyButton = NSButton(title: "Copy", target: self, action: #selector(copyToClipboard))
        copyButton.bezelStyle = .rounded
        copyButton.translatesAutoresizingMaskIntoConstraints = false
        contentView.addSubview(copyButton)

        // Save button
        saveButton = NSButton(title: "Save...", target: self, action: #selector(saveTranscript))
        saveButton.bezelStyle = .rounded
        saveButton.translatesAutoresizingMaskIntoConstraints = false
        contentView.addSubview(saveButton)

        // Clear button
        clearButton = NSButton(title: "Clear", target: self, action: #selector(clearTranscript))
        clearButton.bezelStyle = .rounded
        clearButton.translatesAutoresizingMaskIntoConstraints = false
        contentView.addSubview(clearButton)

        // Scroll view for text
        let scrollView = NSScrollView(frame: .zero)
        scrollView.translatesAutoresizingMaskIntoConstraints = false
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = false
        scrollView.autohidesScrollers = true
        scrollView.borderType = .bezelBorder
        contentView.addSubview(scrollView)

        // Text view
        textView = NSTextView(frame: .zero)
        textView.isEditable = true
        textView.isSelectable = true
        textView.font = NSFont.monospacedSystemFont(ofSize: 14, weight: .regular)
        textView.textColor = .labelColor
        textView.backgroundColor = .textBackgroundColor
        textView.autoresizingMask = [.width]
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.textContainer?.widthTracksTextView = true
        textView.textContainer?.containerSize = NSSize(width: CGFloat.greatestFiniteMagnitude,
                                                        height: CGFloat.greatestFiniteMagnitude)
        scrollView.documentView = textView

        // Status bar at bottom
        statusBar = NSView(frame: .zero)
        statusBar.translatesAutoresizingMaskIntoConstraints = false
        statusBar.wantsLayer = true
        statusBar.layer?.backgroundColor = NSColor.windowBackgroundColor.cgColor
        contentView.addSubview(statusBar)

        // Recording indicator (red dot)
        recordingIndicator = NSView(frame: .zero)
        recordingIndicator.translatesAutoresizingMaskIntoConstraints = false
        recordingIndicator.wantsLayer = true
        recordingIndicator.layer?.cornerRadius = 5
        recordingIndicator.layer?.backgroundColor = NSColor.systemGray.cgColor
        statusBar.addSubview(recordingIndicator)

        // Status bar label
        statusBarLabel = NSTextField(labelWithString: "No model loaded")
        statusBarLabel.font = NSFont.systemFont(ofSize: 11)
        statusBarLabel.textColor = .secondaryLabelColor
        statusBarLabel.translatesAutoresizingMaskIntoConstraints = false
        statusBar.addSubview(statusBarLabel)

        // Layout:
        // Row 1: Status
        // Row 2: Model + Language + Load Model + Start/Stop
        // Row 3: Interval + Font Size + Copy + Save + Clear
        // Row 4: Text view
        NSLayoutConstraint.activate([
            // Row 1: Status
            statusLabel.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 12),
            statusLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 16),
            statusLabel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -16),

            // Row 2: Model selection and main controls
            modelPopup.topAnchor.constraint(equalTo: statusLabel.bottomAnchor, constant: 10),
            modelPopup.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 16),
            modelPopup.widthAnchor.constraint(equalToConstant: 160),

            languagePopup.centerYAnchor.constraint(equalTo: modelPopup.centerYAnchor),
            languagePopup.leadingAnchor.constraint(equalTo: modelPopup.trailingAnchor, constant: 8),
            languagePopup.widthAnchor.constraint(equalToConstant: 90),

            loadModelButton.centerYAnchor.constraint(equalTo: modelPopup.centerYAnchor),
            loadModelButton.leadingAnchor.constraint(equalTo: languagePopup.trailingAnchor, constant: 12),

            toggleButton.centerYAnchor.constraint(equalTo: modelPopup.centerYAnchor),
            toggleButton.leadingAnchor.constraint(equalTo: loadModelButton.trailingAnchor, constant: 8),

            // Row 3: Sliders and action buttons
            intervalLabel.topAnchor.constraint(equalTo: modelPopup.bottomAnchor, constant: 10),
            intervalLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 16),
            intervalLabel.widthAnchor.constraint(equalToConstant: 65),

            intervalSlider.centerYAnchor.constraint(equalTo: intervalLabel.centerYAnchor),
            intervalSlider.leadingAnchor.constraint(equalTo: intervalLabel.trailingAnchor, constant: 4),
            intervalSlider.widthAnchor.constraint(equalToConstant: 80),

            fontSizeLabel.centerYAnchor.constraint(equalTo: intervalLabel.centerYAnchor),
            fontSizeLabel.leadingAnchor.constraint(equalTo: intervalSlider.trailingAnchor, constant: 16),
            fontSizeLabel.widthAnchor.constraint(equalToConstant: 65),

            fontSizeSlider.centerYAnchor.constraint(equalTo: intervalLabel.centerYAnchor),
            fontSizeSlider.leadingAnchor.constraint(equalTo: fontSizeLabel.trailingAnchor, constant: 4),
            fontSizeSlider.widthAnchor.constraint(equalToConstant: 80),

            copyButton.centerYAnchor.constraint(equalTo: intervalLabel.centerYAnchor),
            copyButton.trailingAnchor.constraint(equalTo: saveButton.leadingAnchor, constant: -8),

            saveButton.centerYAnchor.constraint(equalTo: intervalLabel.centerYAnchor),
            saveButton.trailingAnchor.constraint(equalTo: clearButton.leadingAnchor, constant: -8),

            clearButton.centerYAnchor.constraint(equalTo: intervalLabel.centerYAnchor),
            clearButton.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -16),

            // Row 4: Text view
            scrollView.topAnchor.constraint(equalTo: intervalLabel.bottomAnchor, constant: 12),
            scrollView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 16),
            scrollView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -16),
            scrollView.bottomAnchor.constraint(equalTo: statusBar.topAnchor, constant: -8),

            // Status bar at bottom
            statusBar.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
            statusBar.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),
            statusBar.bottomAnchor.constraint(equalTo: contentView.bottomAnchor),
            statusBar.heightAnchor.constraint(equalToConstant: 24),

            // Recording indicator
            recordingIndicator.leadingAnchor.constraint(equalTo: statusBar.leadingAnchor, constant: 12),
            recordingIndicator.centerYAnchor.constraint(equalTo: statusBar.centerYAnchor),
            recordingIndicator.widthAnchor.constraint(equalToConstant: 10),
            recordingIndicator.heightAnchor.constraint(equalToConstant: 10),

            // Status bar label
            statusBarLabel.leadingAnchor.constraint(equalTo: recordingIndicator.trailingAnchor, constant: 8),
            statusBarLabel.centerYAnchor.constraint(equalTo: statusBar.centerYAnchor),
            statusBarLabel.trailingAnchor.constraint(lessThanOrEqualTo: statusBar.trailingAnchor, constant: -12)
        ])
    }

    private func setupCallbacks() {
        whisperManager.onStatusUpdate = { [weak self] status in
            DispatchQueue.main.async {
                guard let self = self else { return }
                self.statusLabel.stringValue = status
                if self.whisperManager.isModelLoaded {
                    self.toggleButton.isEnabled = true
                    self.statusLabel.textColor = .systemGreen
                    // Only re-enable controls if NOT currently listening
                    if !self.whisperManager.isListening {
                        self.modelPopup.isEnabled = true
                        self.languagePopup.isEnabled = true
                        self.loadModelButton.isEnabled = true
                        self.intervalSlider.isEnabled = true
                    }
                }
                self.updateStatusBar()
            }
        }

        whisperManager.onTranscription = { [weak self] text in
            DispatchQueue.main.async {
                self?.appendTranscription(text)
            }
        }

        whisperManager.onError = { [weak self] error in
            DispatchQueue.main.async {
                self?.textView.string += "\n[Error: \(error)]\n"
                self?.textView.scrollToEndOfDocument(nil)
            }
        }
    }

    private func appendTranscription(_ text: String) {
        if !fullTranscript.isEmpty {
            fullTranscript += " "
        }
        fullTranscript += text
        textView.string = fullTranscript
        textView.scrollToEndOfDocument(nil)
    }

    private func updateStatusBar() {
        if whisperManager.isListening {
            recordingIndicator.layer?.backgroundColor = NSColor.systemRed.cgColor
            statusBarLabel.stringValue = "● Recording — Model: \(currentModelName)"
            statusBarLabel.textColor = .labelColor
        } else if whisperManager.isModelLoaded {
            recordingIndicator.layer?.backgroundColor = NSColor.systemGreen.cgColor
            statusBarLabel.stringValue = "Ready — Model: \(currentModelName)"
            statusBarLabel.textColor = .secondaryLabelColor
        } else if !currentModelName.isEmpty {
            recordingIndicator.layer?.backgroundColor = NSColor.systemOrange.cgColor
            statusBarLabel.stringValue = "Loading: \(currentModelName)..."
            statusBarLabel.textColor = .secondaryLabelColor
        } else {
            recordingIndicator.layer?.backgroundColor = NSColor.systemGray.cgColor
            statusBarLabel.stringValue = "No model loaded"
            statusBarLabel.textColor = .tertiaryLabelColor
        }
    }

    @objc private func loadModel() {
        let selectedIndex = modelPopup.indexOfSelectedItem
        guard selectedIndex >= 0 && selectedIndex < WhisperTranscriptionManager.availableModels.count else { return }

        currentModelName = WhisperTranscriptionManager.availableModels[selectedIndex]
        loadModelButton.isEnabled = false
        modelPopup.isEnabled = false
        toggleButton.isEnabled = false
        statusLabel.stringValue = "Loading model..."
        statusLabel.textColor = .systemOrange
        updateStatusBar()

        Task {
            await whisperManager.loadModel(name: currentModelName)
        }
    }

    @objc private func languageChanged() {
        let selectedIndex = languagePopup.indexOfSelectedItem
        guard selectedIndex >= 0 && selectedIndex < WhisperTranscriptionManager.availableLanguages.count else { return }

        let langCode = WhisperTranscriptionManager.availableLanguages[selectedIndex].code
        whisperManager.selectedLanguage = langCode
    }

    @objc private func intervalChanged() {
        let value = Int(intervalSlider.doubleValue)
        intervalLabel.stringValue = "Interval: \(value)s"
        whisperManager.transcriptionInterval = Double(value)
    }

    @objc private func fontSizeChanged() {
        let value = Int(fontSizeSlider.doubleValue)
        currentFontSize = CGFloat(value)
        fontSizeLabel.stringValue = "Font: \(value)pt"
        textView.font = NSFont.monospacedSystemFont(ofSize: currentFontSize, weight: .regular)
    }

    @objc private func toggleListening() {
        if whisperManager.isListening {
            whisperManager.stopListening()
            toggleButton.title = "Start Listening"
            statusLabel.textColor = .systemGreen
            // Re-enable controls when stopped
            modelPopup.isEnabled = true
            languagePopup.isEnabled = true
            loadModelButton.isEnabled = true
            intervalSlider.isEnabled = true
            updateStatusBar()
        } else {
            do {
                try whisperManager.startListening()
                toggleButton.title = "Stop Listening"
                statusLabel.textColor = .systemGreen
                // Disable controls while listening
                modelPopup.isEnabled = false
                languagePopup.isEnabled = false
                loadModelButton.isEnabled = false
                intervalSlider.isEnabled = false
                updateStatusBar()
            } catch {
                statusLabel.stringValue = "Error: \(error.localizedDescription)"
                statusLabel.textColor = .systemRed
            }
        }
    }

    @objc private func copyToClipboard() {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(textView.string, forType: .string)

        let originalTitle = copyButton.title
        copyButton.title = "Copied!"
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in
            self?.copyButton.title = originalTitle
        }
    }

    @objc private func saveTranscript() {
        let savePanel = NSSavePanel()
        savePanel.allowedContentTypes = [.plainText]
        savePanel.nameFieldStringValue = "whisper_transcript.txt"

        savePanel.beginSheetModal(for: window!) { [weak self] response in
            guard response == .OK, let url = savePanel.url else { return }

            do {
                try self?.textView.string.write(to: url, atomically: true, encoding: .utf8)
            } catch {
                let alert = NSAlert()
                alert.messageText = "Failed to save"
                alert.informativeText = error.localizedDescription
                alert.runModal()
            }
        }
    }

    @objc private func clearTranscript() {
        fullTranscript = ""
        textView.string = ""
    }
}

// MARK: - Application Delegate

class AppDelegate: NSObject, NSApplicationDelegate {
    var windowController: MainWindowController!

    func applicationDidFinishLaunching(_ notification: Notification) {
        windowController = MainWindowController()
        windowController.showWindow(nil)
        windowController.window?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }

    @objc func clearModelCache() {
        // WhisperKit uses Hugging Face Hub cache
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        let hfCacheURL = homeDir.appendingPathComponent(".cache/huggingface/hub")

        // Find all whisperkit model folders
        var whisperKitFolders: [URL] = []
        if let contents = try? FileManager.default.contentsOfDirectory(at: hfCacheURL, includingPropertiesForKeys: nil) {
            whisperKitFolders = contents.filter { $0.lastPathComponent.contains("whisperkit") || $0.lastPathComponent.contains("whisper") }
        }

        // Also check app-specific cache
        let appCacheURL = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
            .appendingPathComponent("com.local.whisperspeechtotext")

        let alert = NSAlert()
        alert.messageText = "Clear Model Cache?"

        var infoText = "This will delete all downloaded Whisper models.\n\nLocations checked:"
        infoText += "\n• \(hfCacheURL.path)"
        if !whisperKitFolders.isEmpty {
            infoText += "\n  Found: \(whisperKitFolders.map { $0.lastPathComponent }.joined(separator: ", "))"
        }
        infoText += "\n• \(appCacheURL.path)"

        alert.informativeText = infoText
        alert.alertStyle = .warning
        alert.addButton(withTitle: "Clear Cache")
        alert.addButton(withTitle: "Cancel")

        if alert.runModal() == .alertFirstButtonReturn {
            var cleared = false
            var errors: [String] = []

            // Clear WhisperKit folders from HF cache
            for folder in whisperKitFolders {
                do {
                    try FileManager.default.removeItem(at: folder)
                    cleared = true
                } catch {
                    errors.append(folder.lastPathComponent + ": " + error.localizedDescription)
                }
            }

            // Clear app cache
            if FileManager.default.fileExists(atPath: appCacheURL.path) {
                do {
                    try FileManager.default.removeItem(at: appCacheURL)
                    cleared = true
                } catch {
                    errors.append("App cache: " + error.localizedDescription)
                }
            }

            let resultAlert = NSAlert()
            if cleared && errors.isEmpty {
                resultAlert.messageText = "Cache Cleared"
                resultAlert.informativeText = "Model cache has been deleted."
                resultAlert.alertStyle = .informational
            } else if cleared && !errors.isEmpty {
                resultAlert.messageText = "Partially Cleared"
                resultAlert.informativeText = "Some items cleared, but errors occurred:\n" + errors.joined(separator: "\n")
                resultAlert.alertStyle = .warning
            } else if !errors.isEmpty {
                resultAlert.messageText = "Error"
                resultAlert.informativeText = "Failed to clear cache:\n" + errors.joined(separator: "\n")
                resultAlert.alertStyle = .critical
            } else {
                resultAlert.messageText = "No Cache Found"
                resultAlert.informativeText = "No WhisperKit model cache was found."
                resultAlert.alertStyle = .informational
            }
            resultAlert.runModal()
        }
    }
}

// MARK: - Main Entry Point

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)

// Menu bar
let mainMenu = NSMenu()

let appMenuItem = NSMenuItem()
mainMenu.addItem(appMenuItem)
let appMenu = NSMenu()
appMenuItem.submenu = appMenu
appMenu.addItem(withTitle: "About Whisper Speech to Text",
                action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)),
                keyEquivalent: "")
appMenu.addItem(NSMenuItem.separator())
appMenu.addItem(withTitle: "Clear Model Cache...",
                action: #selector(AppDelegate.clearModelCache),
                keyEquivalent: "")
appMenu.addItem(NSMenuItem.separator())
appMenu.addItem(withTitle: "Quit", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")

let editMenuItem = NSMenuItem()
mainMenu.addItem(editMenuItem)
let editMenu = NSMenu(title: "Edit")
editMenuItem.submenu = editMenu
editMenu.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
editMenu.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
editMenu.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
editMenu.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")

app.mainMenu = mainMenu
app.run()
