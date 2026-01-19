import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject var appState: AppState
    @State private var isImporting = false
    @State private var isExporting = false
    @State private var exportDocument = SVGDocument()
    @State private var exportDefaultFilename: String? = "drawing"
    
    var body: some View {
        ZStack {
            CanvasView()
                .ignoresSafeArea()
        }
        .toolbar {
            if appState.showUI {
                LiquidGlassToolbar(
                    selectedColor: $appState.selectedColor,
                    onNew: { appState.clear() },
                    onSave: { performSave() },
                    onSaveAs: { performSaveAs() },
                    onOpen: { performOpen() }
                )
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: .openFile)) { _ in
            performOpen()
        }
        .onReceive(NotificationCenter.default.publisher(for: .saveFile)) { _ in
            performSave()
        }
        .onReceive(NotificationCenter.default.publisher(for: .saveAsFile)) { _ in
            performSaveAs()
        }
        .fileImporter(isPresented: $isImporting, allowedContentTypes: [.svg], allowsMultipleSelection: false) { result in
            switch result {
            case .success(let urls):
                if let url = urls.first {
                    appState.loadFromSVG(url: url)
                }
            case .failure:
                break
            }
        }
        .fileExporter(
            isPresented: $isExporting,
            document: exportDocument,
            contentType: .svg,
            defaultFilename: exportDefaultFilename
        ) { result in
            switch result {
            case .success(let url):
                // Update current file URL after a successful Save As
                appState.currentFileURL = url
            case .failure:
                break
            }
        }
    }
    
    @MainActor
    private func performSave() {
        if let url = appState.currentFileURL {
            appState.saveToSVG(url: url)
        } else {
            performSaveAs()
        }
    }

    @MainActor
    private func performOpen() {
        openSVG()
    }

    @MainActor
    private func performSaveAs() {
        do {
            // Set the filename BEFORE generating data to avoid picking up temp file names
            if let currentURL = appState.currentFileURL {
                exportDefaultFilename = currentURL.deletingPathExtension().lastPathComponent
            } else {
                exportDefaultFilename = "drawing"
            }
            
            let data = try generateSVGData()
            exportDocument = SVGDocument(data: data)
            isExporting = true
        } catch {
            // If export data generation fails, fall back silently (or handle error UI if desired)
        }
    }

    @MainActor
    private func generateSVGData() throws -> Data {
        // Preserve the current file URL to prevent it from being overwritten by temp file
        let originalURL = appState.currentFileURL
        
        let tempDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        let tempURL = tempDir.appendingPathComponent(UUID().uuidString).appendingPathExtension("svg")
        appState.saveToSVG(url: tempURL)
        let data = try Data(contentsOf: tempURL)
        try? FileManager.default.removeItem(at: tempURL)
        
        // Restore the original URL
        appState.currentFileURL = originalURL
        
        return data
    }

    @MainActor
    private func openSVG() {
        isImporting = true
    }
}
private struct LiquidGlassToolbar: ToolbarContent {
    @Binding var selectedColor: Color
    var onNew: () -> Void
    var onSave: () -> Void
    var onSaveAs: () -> Void
    var onOpen: () -> Void

    @ToolbarContentBuilder
    var body: some ToolbarContent {
        
        ToolbarItem() {
            ColorPicker("", selection: $selectedColor)
                .labelsHidden()
                .help("System Color Picker")
        }

        ToolbarItem() {
            Button(action: onNew) {
                Label("New", systemImage: "plus")
            }
            .help("New")
            .keyboardShortcut("n", modifiers: .command)
        }

        ToolbarItem() {
            Button(action: onSave) {
                Label("Save", systemImage: "square.and.arrow.down")
            }
            .help("Save")
            .keyboardShortcut("s", modifiers: .command)
        }

        ToolbarItem() {
            Button(action: onSaveAs) {
                Label("Save As", systemImage: "square.and.arrow.down.on.square")
            }
            .help("Save As")
            .keyboardShortcut("s", modifiers: [.command, .shift])
        }

        ToolbarItem() {
            Button(action: onOpen) {
                Label("Open", systemImage: "folder")
            }
            .help("Open")
            .keyboardShortcut("o", modifiers: .command)
        }
        
        ToolbarSpacer()
        
        ToolbarItem() {
            Text("Space: Finish | Click: Add Point")
                .padding(.horizontal, 12)
        }
    }
}

private struct SVGDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.svg] }
    var data: Data
    init(data: Data = Data()) {
        self.data = data
    }

    init(configuration: ReadConfiguration) throws {
        self.data = configuration.file.regularFileContents ?? Data()
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: data)
    }
}

