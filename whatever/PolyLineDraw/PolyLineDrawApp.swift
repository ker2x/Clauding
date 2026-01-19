import SwiftUI

@main
struct PolyLineDrawApp: App {
    @StateObject var appState = AppState()
    
    init() {
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
        //        .frame(minWidth: 800, minHeight: 600)
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1200, height: 800)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("New") {
                    appState.clear()
                }
                .keyboardShortcut("n", modifiers: .command)
            }
            
            CommandGroup(after: .newItem) {
                Button("Open...") {
                    NotificationCenter.default.post(name: .openFile, object: nil)
                }
                .keyboardShortcut("o", modifiers: .command)
                
                Divider()
                
                Button("Save") {
                    NotificationCenter.default.post(name: .saveFile, object: nil)
                }
                .keyboardShortcut("s", modifiers: .command)
                
                Button("Save As...") {
                    NotificationCenter.default.post(name: .saveAsFile, object: nil)
                }
                .keyboardShortcut("s", modifiers: [.command, .shift])
            }
            
//            CommandMenu("View") {
//                Toggle("Show UI Overlay", isOn: $appState.showUI)
//                    .keyboardShortcut("u", modifiers: .command)
//            }
        }
    }
}

extension Notification.Name {
    static let openFile = Notification.Name("openFile")
    static let saveFile = Notification.Name("saveFile")
    static let saveAsFile = Notification.Name("saveAsFile")
}
