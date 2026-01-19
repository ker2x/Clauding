import SwiftUI
import Combine

struct DrawingPoint: Identifiable, Codable {
    var id = UUID()
    var x: CGFloat
    var y: CGFloat
}

struct Polyline: Identifiable, Codable {
    var id = UUID()
    var points: [DrawingPoint]
    var color: ColorData
    var lineWidth: CGFloat = 2.0
}

struct ColorData: Codable {
    var red: Double
    var green: Double
    var blue: Double
    var opacity: Double
    
    var swiftUIColor: Color {
        Color(.sRGB, red: red, green: green, blue: blue, opacity: opacity)
    }
    
    init(red: Double, green: Double, blue: Double, opacity: Double) {
        self.red = red
        self.green = green
        self.blue = blue
        self.opacity = opacity
    }

    init(color: Color) {
        // Convert to NSColor in sRGB color space to ensure valid components
        let nsColor = NSColor(color).usingColorSpace(.sRGB) ?? NSColor(color)
        
        if let components = nsColor.cgColor.components {
            if components.count >= 3 {
                self.red = Double(components[0])
                self.green = Double(components[1])
                self.blue = Double(components[2])
                self.opacity = components.count > 3 ? Double(components[3]) : 1.0
            } else if components.count >= 1 {
                // Grayscale
                self.red = Double(components[0])
                self.green = Double(components[0])
                self.blue = Double(components[0])
                self.opacity = components.count > 1 ? Double(components[1]) : 1.0
            } else {
                self.red = 0; self.green = 0; self.blue = 0; self.opacity = 1
            }
        } else {
            self.red = 0; self.green = 0; self.blue = 0; self.opacity = 1
        }
    }
}

class AppState: ObservableObject {
    @Published var polylines: [Polyline] = []
    @Published var currentPolyline: Polyline?
    @Published var selectedColor: Color = .black
    
    // Canvas State
    @Published var zoom: CGFloat = 1.0
    @Published var panOffset: CGPoint = .zero
    
    // File tracking
    @Published var currentFileURL: URL?
    
    // UI State
    @Published var showUI: Bool = true
    
    func startNewPolyline(at point: CGPoint) {
        currentPolyline = Polyline(points: [DrawingPoint(x: point.x, y: point.y)], color: ColorData(color: selectedColor))
    }
    
    func addPointToCurrent(_ point: CGPoint) {
        guard currentPolyline != nil else { return }
        currentPolyline?.points.append(DrawingPoint(x: point.x, y: point.y))
    }
    
    func updateCurrentPoint(_ point: CGPoint) {
        // For live preview of the line segment being drawn (mouse move)
        // We might want a separate 'previewPoint' or just reuse the last point approach?
        // Strategy: 'currentPolyline' contains CONFIRMED points. 
        // We'll handle the "rubber banding" line in the view state or a temp point.
    }
    
    func finishCurrentPolyline() {
        if let poly = currentPolyline {
            if poly.points.count > 1 {
                polylines.append(poly)
            }
            currentPolyline = nil
        }
    }
    
    // MARK: - Persistence (SVG)
    
    func saveToSVG(url: URL) {
        var svg = "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n"
        svg += "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">\n"
        
        for poly in polylines {
            let pointsString = poly.points.map { "\($0.x),\($0.y)" }.joined(separator: " ")
            let c = poly.color
            let colorString = "rgb(\(Int(c.red * 255)),\(Int(c.green * 255)),\(Int(c.blue * 255)))"
            svg += "  <polyline points=\"\(pointsString)\" stroke=\"\(colorString)\" stroke-width=\"\(poly.lineWidth)\" stroke-opacity=\"\(c.opacity)\" fill=\"none\" />\n"
        }
        
        svg += "</svg>"
        
        do {
            try svg.write(to: url, atomically: true, encoding: .utf8)
            currentFileURL = url  // Track the saved file
        } catch {
            print("Failed to save SVG: \(error)")
        }
    }
    
    func loadFromSVG(url: URL) {
        guard let parser = XMLParser(contentsOf: url) else { return }
        let delegate = SVGParserDelegate()
        parser.delegate = delegate
        if parser.parse() {
            self.polylines = delegate.polylines
            // Reset view
            self.panOffset = .zero
            self.zoom = 1.0
            currentFileURL = url  // Track the opened file
        }
    }
    
    func clear() {
        polylines = []
        currentPolyline = nil
        panOffset = .zero
        zoom = 1.0
        currentFileURL = nil
    }
}

class SVGParserDelegate: NSObject, XMLParserDelegate {
    var polylines: [Polyline] = []
    
    func parser(_ parser: XMLParser, didStartElement elementName: String, namespaceURI: String?, qualifiedName qName: String?, attributes attributeDict: [String : String] = [:]) {
        if elementName == "polyline" {
            guard let pointsStr = attributeDict["points"] else { return }
            
            // Parse Points
            let coords = pointsStr.components(separatedBy: CharacterSet(charactersIn: " ,")).filter { !$0.isEmpty }.compactMap { Double($0) }
            var points: [DrawingPoint] = []
            var i = 0
            while i < coords.count - 1 {
                points.append(DrawingPoint(x: coords[i], y: coords[i+1]))
                i += 2
            }
            
            // Parse Color
            var colorData = ColorData(red: 0, green: 0, blue: 0, opacity: 1)
            
            if let stroke = attributeDict["stroke"] {
                if stroke.hasPrefix("rgb") {
                    let components = stroke.replacingOccurrences(of: "rgb(", with: "").replacingOccurrences(of: ")", with: "").components(separatedBy: ",")
                    if components.count >= 3 {
                        colorData.red = (Double(components[0].trimmingCharacters(in: .whitespaces)) ?? 0) / 255.0
                        colorData.green = (Double(components[1].trimmingCharacters(in: .whitespaces)) ?? 0) / 255.0
                        colorData.blue = (Double(components[2].trimmingCharacters(in: .whitespaces)) ?? 0) / 255.0
                    }
                }
            }
            
            if let opacityStr = attributeDict["stroke-opacity"] {
                colorData.opacity = Double(opacityStr) ?? 1.0
            }
            
            let width = Double(attributeDict["stroke-width"] ?? "2") ?? 2.0
            
            polylines.append(Polyline(points: points, color: colorData, lineWidth: width))
        }
    }
}
