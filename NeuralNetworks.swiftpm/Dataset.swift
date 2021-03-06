import Foundation
import CoreGraphics

public enum DataSizeType: Int, Codable {
    case oneD = 1
    case twoD
    case threeD
}

public struct DataSize: Codable {
    var type: DataSizeType
    var width: Int
    var height: Int?
    var depth: Int?
    
    public init(width: Int) {
        type = .oneD
        self.width = width
    }
    
    public init(width: Int, height: Int) {
        type = .twoD
        self.width = width
        self.height = height
    }
    
    public init(width: Int, height: Int, depth: Int) {
        type = .threeD
        self.width = width
        self.height = height
        self.depth = depth
    }
    
}

public struct DataPiece: Codable, Equatable {
    public static func == (lhs: DataPiece, rhs: DataPiece) -> Bool {
        return lhs.body == rhs.body
    }
    
    public var size: DataSize
    public var body: [Float]
    
    func get(x: Int) -> Float {
        return body[x]
    }
    
    func get(x: Int, y: Int) -> Float {
        return body[x+y*size.width]
    }
    
    func get(x: Int, y: Int, z: Int) -> Float {
        return body[z+(x+y*size.width)*size.depth!]
    }
    
    public init(size: DataSize, body: [Float]) {
        var flatSize = size.width
        if let height = size.height {
            flatSize *= height
        }
        if let depth = size.depth {
            flatSize *= depth
        }
        if flatSize != body.count {
            fatalError("DataPiece body does not conform to DataSize.")
        }
        self.size = size
        self.body = body
    }
    
    public init(label: Int, count: Int) {
        self.size = .init(width: count)
        self.body = Array(repeating: .zero, count: count)
        self.body[label] = 1
    }
}

public struct DataItem: Codable {
    var input: DataPiece
    var output: DataPiece
    var point: CGPoint?
    
    public init(input: DataPiece, output: DataPiece) {
        self.input = input
        self.output = output
    }
    
    public init(input: [Float], inputSize: DataSize, output: [Float], outputSize: DataSize) {
        self.input = DataPiece(size: inputSize, body: input)
        self.output = DataPiece(size: outputSize, body: output)
    }
    
    public init(flatInput: [Float], flatOutput: [Float]) {
        self.input = DataPiece(size: .init(width: flatInput.count), body: flatInput)
        self.output = DataPiece(size: .init(width: flatOutput.count), body: flatOutput)
    }
    
    public init(point: CGPoint, input: [Float], output: Float) {
        self.init(flatInput: input, flatOutput: [output])
        self.point = point
    }
    
    public init(decimal: Int, output: Bool) {
        var binary = [Int]()
        var currentDecimal = decimal
        while decimal > 0 {
            binary.append(currentDecimal % 2)
            currentDecimal /= 2
        }
        self.input = DataPiece(size: .init(width: binary.count), body: binary.reversed().map { Float($0) })
        self.output = DataPiece(size: .init(width: 1), body: [output ? 1 : 0])
    }
}

public struct Dataset: Codable {
    public var items: [DataItem]
    
    public init(canvasRect: CGRect, firstGenerator: () -> CGPoint, secondGenerator: () -> CGPoint, count: Int, inputs: [InputType]) {
        items = (0..<count/2).map { _ in
            var point = firstGenerator()
            while !canvasRect.contains(point) {
                point = firstGenerator()
            }
            return .init(point: point, input: inputs.map { $0.inputForPoint(point) }, output: 1)
        } + (0..<count/2).map { _ in
            var point = secondGenerator()
            while !canvasRect.contains(point) {
                point = secondGenerator()
            }
            return .init(point: point, input: inputs.map { $0.inputForPoint(point) }, output: 0)
        }
    }
    
    public init(items: [DataItem]) {
        self.items = items
    }
}
