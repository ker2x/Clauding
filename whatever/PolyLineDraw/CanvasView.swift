import SwiftUI
import AppKit

struct CanvasView: NSViewRepresentable {
    @EnvironmentObject var appState: AppState

    func makeNSView(context: Context) -> InteractiveCanvas {
        let view = InteractiveCanvas()
        view.appState = appState
        return view
    }

    func updateNSView(_ nsView: InteractiveCanvas, context: Context) {
        nsView.appState = appState
        nsView.setNeedsDisplay(nsView.bounds)
    }
}

class InteractiveCanvas: NSView {
    var appState: AppState?
    var trackingArea: NSTrackingArea?
    var currentMousePosition: CGPoint?

    override var acceptsFirstResponder: Bool { true }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let trackingArea = trackingArea {
            removeTrackingArea(trackingArea)
        }
        let options: NSTrackingArea.Options = [.mouseMoved, .activeInKeyWindow, .activeAlways, .inVisibleRect]
        trackingArea = NSTrackingArea(rect: bounds, options: options, owner: self, userInfo: nil)
        addTrackingArea(trackingArea!)
    }

    override func resetCursorRects() {
        addCursorRect(bounds, cursor: .crosshair)
    }

    override func mouseDown(with event: NSEvent) {
        guard let appState = appState else { return }
        
        let point = convert(event.locationInWindow, from: nil)
        // Convert screen point to world point
        // screen = world * zoom + offset
        // world = (screen - offset) / zoom
        let worldPoint = CGPoint(
            x: (point.x - appState.panOffset.x) / appState.zoom,
            y: (point.y - appState.panOffset.y) / appState.zoom
        )

        if appState.currentPolyline == nil {
            appState.startNewPolyline(at: worldPoint)
        } else {
            appState.addPointToCurrent(worldPoint)
        }
        setNeedsDisplay(bounds)
    }

    override func mouseMoved(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        currentMousePosition = point
        setNeedsDisplay(bounds)
    }
    
    override func mouseDragged(with event: NSEvent) {
        // Left drag (if we wanted drawing-on-drag, but strictly click-click for now)
    }
    
    override func scrollWheel(with event: NSEvent) {
        guard let appState = appState else { return }
        // Natural tracking: scrollingDelta follows finger movement
        appState.panOffset.x += event.scrollingDeltaX
        appState.panOffset.y -= event.scrollingDeltaY // Inverted Y for natural feel usually
        setNeedsDisplay(bounds)
    }
    
    override func rightMouseDragged(with event: NSEvent) {
        pan(with: event)
    }
    
    override func otherMouseDragged(with event: NSEvent) {
        if event.buttonNumber == 2 { // Middle mouse usually
            pan(with: event)
        }
    }
    
    private func pan(with event: NSEvent) {
        guard let appState = appState else { return }
        appState.panOffset.x += event.deltaX
        appState.panOffset.y -= event.deltaY // Y is often flipped in events vs coords
        setNeedsDisplay(bounds)
    }

    override func keyDown(with event: NSEvent) {
        if event.keyCode == 49 { // Spacebar
            appState?.finishCurrentPolyline()
            setNeedsDisplay(bounds)
        } else {
            super.keyDown(with: event)
        }
    }

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        guard let appState = appState else { return }

        guard let context = NSGraphicsContext.current?.cgContext else { return }
        
        // Fill background with soft off-white (essential for drawing app)
        context.setFillColor(NSColor(red: 0.976, green: 0.976, blue: 0.976, alpha: 1.0).cgColor)
        context.fill(bounds)
        
        // Apply transform for Infinite Canvas
        context.saveGState()
        context.translateBy(x: appState.panOffset.x, y: appState.panOffset.y)
        context.scaleBy(x: appState.zoom, y: appState.zoom)

        // Draw helper
        func drawPolyline(_ polyline: Polyline) {
            guard !polyline.points.isEmpty else { return }
            
            let color = NSColor(srgbRed: polyline.color.red, green: polyline.color.green, blue: polyline.color.blue, alpha: polyline.color.opacity)
            context.setStrokeColor(color.cgColor)
            
            context.setLineWidth(polyline.lineWidth)
            context.setLineCap(.round)
            context.setLineJoin(.round)
            
            context.beginPath()
            let p0 = polyline.points[0]
            context.move(to: CGPoint(x: p0.x, y: p0.y))
            
            for i in 1..<polyline.points.count {
                let p = polyline.points[i]
                context.addLine(to: CGPoint(x: p.x, y: p.y))
            }
            context.strokePath()
        }

        // Draw existing polylines
        for poly in appState.polylines {
            drawPolyline(poly)
        }

        // Draw current active polyline
        if let current = appState.currentPolyline {
            drawPolyline(current)
            
            // Draw rubber band line to cursor
            if let mousePos = currentMousePosition, let lastPoint = current.points.last {
                // Convert mousePos (screen) to world
                let worldMouse = CGPoint(
                    x: (mousePos.x - appState.panOffset.x) / appState.zoom,
                    y: (mousePos.y - appState.panOffset.y) / appState.zoom
                )
                
                let color = NSColor(srgbRed: current.color.red, green: current.color.green, blue: current.color.blue, alpha: current.color.opacity * 0.5)
                context.setStrokeColor(color.cgColor)
                
                context.setLineWidth(1.0)
                context.setLineDash(phase: 0, lengths: [5, 5])
                
                context.beginPath()
                context.move(to: CGPoint(x: lastPoint.x, y: lastPoint.y))
                context.addLine(to: worldMouse)
                context.strokePath()
            }
        }
        
        context.restoreGState()
    }
}

extension Color {
    func withOpacity(_ opacity: Double) -> Color {
        return self.opacity(opacity)
    }
}
