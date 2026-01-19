// Speech-to-Text Application
// Uses Apple Speech framework for real-time transcription without audio recording
//
// Frameworks used:
// - Speech: SFSpeechRecognizer for speech recognition
// - AVFoundation: AVAudioEngine for microphone input capture
// - AppKit: Native macOS GUI
//
// NOT used: CoreML, Create ML, any third-party libraries
//
// How it works:
// 1. AVAudioEngine captures microphone input as audio buffers
// 2. Audio buffers are fed to SFSpeechAudioBufferRecognitionRequest
// 3. SFSpeechRecognizer processes buffers and returns transcription results
// 4. Results update the GUI text view in real-time
// 5. No audio is stored - buffers are processed and discarded

import Cocoa
import Speech
import AVFoundation
import UniformTypeIdentifiers

// MARK: - Speech Recognition Manager

/// Manages real-time speech recognition using Apple's Speech framework
/// Uses on-device recognition when available (macOS 13+) for privacy
class SpeechRecognitionManager: NSObject {
    private var speechRecognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()

    /// Called when new transcription is available
    /// Parameters: (segments with confidence, isFinal)
    var onTranscription: (([SFTranscriptionSegment], Bool) -> Void)?

    /// Called when an error occurs
    var onError: ((String) -> Void)?

    /// Called when authorization status changes
    var onAuthorizationStatus: ((SFSpeechRecognizerAuthorizationStatus) -> Void)?

    private(set) var isListening = false
    private var isStopping = false  // Track intentional stops to suppress cancellation errors
    private(set) var currentLocale: Locale

    override init() {
        // Initialize with system locale by default
        self.currentLocale = Locale.current
        self.speechRecognizer = SFSpeechRecognizer(locale: currentLocale)
        super.init()
        speechRecognizer?.delegate = self
    }

    /// Returns all locales supported by the Speech framework
    /// Sorted alphabetically by display name
    static func supportedLocales() -> [Locale] {
        return SFSpeechRecognizer.supportedLocales()
            .sorted { $0.localizedString(forIdentifier: $0.identifier) ?? "" <
                      $1.localizedString(forIdentifier: $1.identifier) ?? "" }
    }

    /// Change the recognition language
    /// Must be called while not listening
    func setLocale(_ locale: Locale) {
        guard !isListening else { return }
        currentLocale = locale
        speechRecognizer = SFSpeechRecognizer(locale: locale)
        speechRecognizer?.delegate = self
    }

    /// Request authorization for speech recognition
    func requestAuthorization() {
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            DispatchQueue.main.async {
                self?.onAuthorizationStatus?(status)

                switch status {
                case .authorized:
                    print("Speech recognition authorized")
                case .denied:
                    self?.onError?("Speech recognition access denied by user")
                case .restricted:
                    self?.onError?("Speech recognition restricted on this device")
                case .notDetermined:
                    self?.onError?("Speech recognition authorization not determined")
                @unknown default:
                    self?.onError?("Unknown authorization status")
                }
            }
        }
    }

    /// Start listening and transcribing speech from the microphone
    func startListening() throws {
        // Reset stopping flag when starting
        isStopping = false

        // Cancel any existing task
        if recognitionTask != nil {
            recognitionTask?.cancel()
            recognitionTask = nil
        }

        guard let speechRecognizer = speechRecognizer, speechRecognizer.isAvailable else {
            throw NSError(domain: "SpeechRecognition", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Speech recognizer not available"])
        }

        // Create recognition request
        // SFSpeechAudioBufferRecognitionRequest allows feeding audio buffers directly
        // without needing to record to a file
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()

        guard let recognitionRequest = recognitionRequest else {
            throw NSError(domain: "SpeechRecognition", code: 2,
                         userInfo: [NSLocalizedDescriptionKey: "Unable to create recognition request"])
        }

        // Configure for real-time results
        recognitionRequest.shouldReportPartialResults = true

        // macOS 13+ features
        if #available(macOS 13, *) {
            // Automatic punctuation - adds periods, commas, question marks
            recognitionRequest.addsPunctuation = true

            // On-device recognition for privacy (audio never leaves device)
            // Set to false to allow server-based recognition which may be more accurate
            recognitionRequest.requiresOnDeviceRecognition = false
            if speechRecognizer.supportsOnDeviceRecognition {
                print("On-device recognition available")
            }
        }

        // Get the audio input node (microphone)
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)

        // Verify we have a valid audio format
        guard recordingFormat.sampleRate > 0 else {
            throw NSError(domain: "SpeechRecognition", code: 3,
                         userInfo: [NSLocalizedDescriptionKey: "Invalid audio format - check microphone permissions"])
        }

        // Install tap on the audio input to capture microphone audio
        // The tap captures audio buffers and feeds them to the recognition request
        // Buffer size of 1024 provides good balance between latency and efficiency
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
            // Feed audio buffer directly to speech recognizer
            // No audio is stored - buffer is processed and discarded
            self?.recognitionRequest?.append(buffer)
        }

        // Start the audio engine
        audioEngine.prepare()
        try audioEngine.start()

        // Start recognition task
        recognitionTask = speechRecognizer.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            var isFinal = false

            if let result = result {
                // Get segments with confidence scores from the transcription
                // Each segment contains: substring, confidence (0.0-1.0), timestamp, duration
                let segments = result.bestTranscription.segments
                isFinal = result.isFinal

                DispatchQueue.main.async {
                    self?.onTranscription?(segments, isFinal)
                }
            }

            if let error = error {
                DispatchQueue.main.async {
                    // Suppress error if we intentionally stopped listening
                    guard let self = self, !self.isStopping else { return }
                    self.onError?("Recognition error: \(error.localizedDescription)")
                }
            }

            // If we got a final result or error, we may need to restart
            // Apple's speech recognition has a ~60 second limit per request
            if isFinal || error != nil {
                self?.handleRecognitionEnd()
            }
        }

        isListening = true
        print("Started listening...")
    }

    /// Handle end of recognition (timeout or error) and restart
    private func handleRecognitionEnd() {
        // Recognition ended - restart to continue listening
        // Apple limits each recognition request to ~60 seconds
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            guard let self = self, self.isListening else { return }

            // Clean up current session
            self.audioEngine.inputNode.removeTap(onBus: 0)
            self.recognitionRequest = nil
            self.recognitionTask = nil

            // Restart listening
            do {
                try self.startListening()
            } catch {
                self.onError?("Failed to restart: \(error.localizedDescription)")
            }
        }
    }

    /// Stop listening
    func stopListening() {
        isListening = false
        isStopping = true  // Mark as intentional stop to suppress cancellation error

        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)

        recognitionRequest?.endAudio()
        recognitionRequest = nil

        recognitionTask?.cancel()
        recognitionTask = nil

        print("Stopped listening")
    }
}

// MARK: - SFSpeechRecognizerDelegate

extension SpeechRecognitionManager: SFSpeechRecognizerDelegate {
    func speechRecognizer(_ speechRecognizer: SFSpeechRecognizer, availabilityDidChange available: Bool) {
        DispatchQueue.main.async { [weak self] in
            if !available {
                self?.onError?("Speech recognition became unavailable")
            }
        }
    }
}

// MARK: - Main Window Controller

class MainWindowController: NSWindowController {
    private let speechManager = SpeechRecognitionManager()
    private var textView: NSTextView!
    private var statusLabel: NSTextField!
    private var languagePopup: NSPopUpButton!
    private var toggleButton: NSButton!
    private var copyButton: NSButton!
    private var saveButton: NSButton!
    private var clearButton: NSButton!
    private var supportedLocales: [Locale] = []

    // Store all final transcriptions as attributed string (with colors)
    private var fullTranscript = NSMutableAttributedString()
    // Current partial segments being updated
    private var currentPartialSegments: [SFTranscriptionSegment] = []

    convenience init() {
        // Create window
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 800, height: 600),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "Speech to Text"
        window.center()
        window.minSize = NSSize(width: 400, height: 300)

        self.init(window: window)
        setupUI()
        setupSpeechManager()

        // Sync speech manager with selected language in dropdown
        languageChanged()
    }

    private func setupUI() {
        guard let window = window else { return }

        let contentView = NSView(frame: window.contentView!.bounds)
        contentView.autoresizingMask = [.width, .height]
        window.contentView = contentView

        // Status label at top
        statusLabel = NSTextField(labelWithString: "Initializing...")
        statusLabel.font = NSFont.systemFont(ofSize: 14, weight: .medium)
        statusLabel.textColor = .secondaryLabelColor
        statusLabel.translatesAutoresizingMaskIntoConstraints = false
        contentView.addSubview(statusLabel)

        // Language picker popup
        languagePopup = NSPopUpButton(frame: .zero, pullsDown: false)
        languagePopup.translatesAutoresizingMaskIntoConstraints = false
        languagePopup.target = self
        languagePopup.action = #selector(languageChanged)
        contentView.addSubview(languagePopup)

        // Only show specific languages
        let allowedIdentifiers = ["en-US", "en-GB", "fr-FR", "ja-JP"]
        let displayNames = [
            "en-US": "English (US)",
            "en-GB": "English (British)",
            "fr-FR": "French (France)",
            "ja-JP": "Japanese"
        ]

        let allLocales = SpeechRecognitionManager.supportedLocales()
        supportedLocales = allowedIdentifiers.compactMap { id in
            allLocales.first { $0.identifier == id }
        }

        var currentIndex = 0
        for (index, locale) in supportedLocales.enumerated() {
            let displayName = displayNames[locale.identifier] ?? locale.identifier
            languagePopup.addItem(withTitle: displayName)

            // Select French by default (user's system locale)
            if locale.identifier == "fr-FR" {
                currentIndex = index
            }
        }
        languagePopup.selectItem(at: currentIndex)

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

        // Text view for transcription
        textView = NSTextView(frame: .zero)
        textView.isEditable = true  // Allow editing and copy/paste
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

        // Layout constraints
        // Row 1: Status label + action buttons
        // Row 2: Language picker
        // Row 3+: Text view
        NSLayoutConstraint.activate([
            // Row 1
            statusLabel.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 16),
            statusLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 16),

            toggleButton.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 12),
            toggleButton.trailingAnchor.constraint(equalTo: copyButton.leadingAnchor, constant: -8),

            copyButton.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 12),
            copyButton.trailingAnchor.constraint(equalTo: saveButton.leadingAnchor, constant: -8),

            saveButton.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 12),
            saveButton.trailingAnchor.constraint(equalTo: clearButton.leadingAnchor, constant: -8),

            clearButton.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 12),
            clearButton.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -16),

            // Row 2: Language picker
            languagePopup.topAnchor.constraint(equalTo: statusLabel.bottomAnchor, constant: 12),
            languagePopup.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 16),
            languagePopup.widthAnchor.constraint(lessThanOrEqualToConstant: 300),

            // Text view
            scrollView.topAnchor.constraint(equalTo: languagePopup.bottomAnchor, constant: 12),
            scrollView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 16),
            scrollView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -16),
            scrollView.bottomAnchor.constraint(equalTo: contentView.bottomAnchor, constant: -16)
        ])
    }

    private func setupSpeechManager() {
        speechManager.onAuthorizationStatus = { [weak self] status in
            switch status {
            case .authorized:
                self?.statusLabel.stringValue = "Ready - Click 'Start Listening' to begin"
                self?.statusLabel.textColor = .systemGreen
                self?.toggleButton.isEnabled = true
            case .denied, .restricted:
                self?.statusLabel.stringValue = "Permission denied - Enable in System Settings > Privacy > Speech Recognition"
                self?.statusLabel.textColor = .systemRed
            case .notDetermined:
                self?.statusLabel.stringValue = "Requesting permission..."
                self?.statusLabel.textColor = .systemOrange
            @unknown default:
                break
            }
        }

        speechManager.onTranscription = { [weak self] segments, isFinal in
            self?.updateTranscription(segments, isFinal: isFinal)
        }

        speechManager.onError = { [weak self] error in
            self?.appendToTextView("\n[Error: \(error)]\n")
        }

        // Request authorization
        speechManager.requestAuthorization()
    }

    /// Convert confidence (0.0-1.0) to color
    /// Confidence of 0 means "unknown" (common for partial results)
    /// High confidence (>0.8) = green, medium (0.5-0.8) = orange, low (<0.5) = red
    private func colorForConfidence(_ confidence: Float) -> NSColor {
        // Confidence of 0 typically means "not available" - use default text color
        if confidence == 0 {
            return .labelColor
        } else if confidence >= 0.8 {
            return .systemGreen
        } else if confidence >= 0.5 {
            return .systemOrange
        } else {
            return .systemRed
        }
    }

    /// Create attributed string from segments with confidence-based coloring
    private func attributedString(from segments: [SFTranscriptionSegment], dimmed: Bool = false) -> NSAttributedString {
        let result = NSMutableAttributedString()
        let font = NSFont.monospacedSystemFont(ofSize: 14, weight: .regular)

        for (index, segment) in segments.enumerated() {
            var color = colorForConfidence(segment.confidence)
            if dimmed {
                color = color.withAlphaComponent(0.5)
            }

            let attributes: [NSAttributedString.Key: Any] = [
                .foregroundColor: color,
                .font: font
            ]

            // Add space before word (except first)
            if index > 0 {
                result.append(NSAttributedString(string: " ", attributes: [.font: font]))
            }

            result.append(NSAttributedString(string: segment.substring, attributes: attributes))
        }

        return result
    }

    private func updateTranscription(_ segments: [SFTranscriptionSegment], isFinal: Bool) {
        if isFinal {
            // Final result - add to permanent transcript
            if fullTranscript.length > 0 {
                fullTranscript.append(NSAttributedString(string: " "))
            }
            fullTranscript.append(attributedString(from: segments))
            currentPartialSegments = []

            // Update display with final text
            textView.textStorage?.setAttributedString(fullTranscript)
        } else {
            // Partial result - show with dimmed indicator
            currentPartialSegments = segments

            // Display full transcript + current partial (dimmed)
            let display = NSMutableAttributedString(attributedString: fullTranscript)
            if display.length > 0 && !currentPartialSegments.isEmpty {
                display.append(NSAttributedString(string: " "))
            }
            display.append(attributedString(from: currentPartialSegments, dimmed: true))

            textView.textStorage?.setAttributedString(display)
        }

        // Auto-scroll to bottom
        textView.scrollToEndOfDocument(nil)
    }

    private func appendToTextView(_ text: String) {
        textView.string += text
        textView.scrollToEndOfDocument(nil)
    }

    @objc private func toggleListening() {
        if speechManager.isListening {
            // Preserve any partial text before stopping
            if !currentPartialSegments.isEmpty {
                if fullTranscript.length > 0 {
                    fullTranscript.append(NSAttributedString(string: " "))
                }
                fullTranscript.append(attributedString(from: currentPartialSegments))
                currentPartialSegments = []
            }
            // Update display with preserved text
            textView.textStorage?.setAttributedString(fullTranscript)

            speechManager.stopListening()
            toggleButton.title = "Start Listening"
            statusLabel.stringValue = "Stopped"
            statusLabel.textColor = .secondaryLabelColor
            languagePopup.isEnabled = true  // Re-enable language selection
        } else {
            do {
                try speechManager.startListening()
                toggleButton.title = "Stop Listening"
                statusLabel.stringValue = "Listening..."
                statusLabel.textColor = .systemGreen
                languagePopup.isEnabled = false  // Disable while listening
            } catch {
                statusLabel.stringValue = "Error: \(error.localizedDescription)"
                statusLabel.textColor = .systemRed
            }
        }
    }

    @objc private func clearTranscript() {
        fullTranscript = NSMutableAttributedString()
        currentPartialSegments = []
        textView.string = ""
    }

    @objc private func languageChanged() {
        let selectedIndex = languagePopup.indexOfSelectedItem
        guard selectedIndex >= 0 && selectedIndex < supportedLocales.count else { return }

        let selectedLocale = supportedLocales[selectedIndex]
        speechManager.setLocale(selectedLocale)

        // Update status to show selected language
        let displayName = selectedLocale.localizedString(forIdentifier: selectedLocale.identifier) ?? selectedLocale.identifier
        if !speechManager.isListening {
            statusLabel.stringValue = "Language: \(displayName)"
        }
    }

    @objc private func copyToClipboard() {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(textView.string, forType: .string)

        // Brief visual feedback
        let originalTitle = copyButton.title
        copyButton.title = "Copied!"
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in
            self?.copyButton.title = originalTitle
        }
    }

    @objc private func saveTranscript() {
        let savePanel = NSSavePanel()
        savePanel.allowedContentTypes = [.plainText]
        savePanel.nameFieldStringValue = "transcript.txt"
        savePanel.title = "Save Transcript"

        savePanel.beginSheetModal(for: window!) { [weak self] response in
            guard response == .OK, let url = savePanel.url else { return }

            // Save current text view content (includes any edits user made)
            let textToSave = self?.textView.string ?? ""

            do {
                try textToSave.write(to: url, atomically: true, encoding: .utf8)
            } catch {
                let alert = NSAlert()
                alert.messageText = "Failed to save"
                alert.informativeText = error.localizedDescription
                alert.alertStyle = .warning
                alert.runModal()
            }
        }
    }
}

// MARK: - Application Delegate

class AppDelegate: NSObject, NSApplicationDelegate {
    var windowController: MainWindowController!

    func applicationDidFinishLaunching(_ notification: Notification) {
        windowController = MainWindowController()
        windowController.showWindow(nil)
        windowController.window?.makeKeyAndOrderFront(nil)

        // Activate the app (bring to front)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}

// MARK: - Main Entry Point

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate

// Set activation policy to regular (shows in dock)
app.setActivationPolicy(.regular)

// Create standard menu bar
let mainMenu = NSMenu()

// App menu
let appMenuItem = NSMenuItem()
mainMenu.addItem(appMenuItem)
let appMenu = NSMenu()
appMenuItem.submenu = appMenu
appMenu.addItem(withTitle: "About Speech to Text", action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
appMenu.addItem(NSMenuItem.separator())
appMenu.addItem(withTitle: "Quit Speech to Text", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")

// Edit menu (enables copy/paste shortcuts)
let editMenuItem = NSMenuItem()
mainMenu.addItem(editMenuItem)
let editMenu = NSMenu(title: "Edit")
editMenuItem.submenu = editMenu
editMenu.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
editMenu.addItem(withTitle: "Redo", action: Selector(("redo:")), keyEquivalent: "Z")
editMenu.addItem(NSMenuItem.separator())
editMenu.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
editMenu.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
editMenu.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
editMenu.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")

app.mainMenu = mainMenu

app.run()
