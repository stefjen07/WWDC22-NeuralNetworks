import Foundation
import CoreGraphics
import SpriteKit

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

    public init(input: DataPiece, output: DataPiece) {
        self.input = input
        self.output = output
    }

    public init(input: [Float], inputSize: DataSize, output: [Float], outputSize: DataSize) {
        self.input = DataPiece(size: inputSize, body: input)
        self.output = DataPiece(size: outputSize, body: output)
    }

    public init(input: [Float], output: [Float]) {
        self.input = DataPiece(size: .init(width: input.count), body: input)
        self.output = DataPiece(size: .init(width: output.count), body: output)
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

    public func save(to url: URL) {
        let encoder = JSONEncoder()
        guard let encoded = try? encoder.encode(self) else {
            print("Unable to encode model.")
            return
        }
        do {
            try encoded.write(to: url)
        } catch {
            print("Unable to write model to disk.")
        }
    }

    public init(from url: URL) {
        let decoder = JSONDecoder()
        guard let data = try? Data(contentsOf: url) else {
            fatalError("Unable to get data from Dataset file.")
        }
        guard let decoded = try? decoder.decode(Dataset.self, from: data) else {
            fatalError("Unable to decode data from Dataset file.")
        }
        self.items = decoded.items
    }

    public init(items: [DataItem]) {
        self.items = items
    }
}

final public class NeuralNetwork: Codable {
    public var layers: [Layer] = []
    public var learningRate = Float(0.05)
    public var epochs = 30
    public var batchSize = 16
    var dropoutEnabled = true
    public var trainScene = SKScene(size: .init(width: 400, height: 800))
    public var delay = 100
    var inputNeurons: [Neuron] = []
    var testInput: DataPiece?

    private enum CodingKeys: String, CodingKey {
        case layers
        case learningRate
        case epochs
        case batchSize
    }

    private func generateCheckPoints() -> [CGPoint] {
        return (0..<canvasSize).flatMap { x in
            (0..<canvasSize).map { y in
                return CGPoint(x: x - canvasSize / 2, y: y - canvasSize / 2)
            }
        }
    }

    func generateOutputMaps(input: DataPiece) {
        generateCheckPoints().forEach { point in
            let input = DataPiece(size: .init(width: 2), body: [Float(point.x), Float(point.y)])
            forward(
                networkInput: input,
                savePoint: .init(x: point.x + CGFloat(canvasSize) / 2, y: point.y + CGFloat(canvasSize) / 2)
            )
        }
    }

    func showOutputMaps() {
        for i in 0..<layers.count {
            layers[i].showOutputMaps()
        }
    }

    public func printSummary() {
        for rawLayer in layers {
            switch rawLayer {
            case let layer as Dense:
                print("Dense layer: \(layer.neurons.count) neurons")
            case let layer as Dropout:
                print("Dropout layer: \(layer.neurons.count) neurons, \(layer.probability) probability")
            default:
                break
            }
        }
    }

    public func encode(to encoder: Encoder) throws {
        let wrappers = layers.map { LayerWrapper($0) }
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(wrappers, forKey: .layers)
        try container.encode(learningRate, forKey: .learningRate)
        try container.encode(epochs, forKey: .epochs)
        try container.encode(batchSize, forKey: .batchSize)
    }

    static func fromFile(fileName: String) -> NeuralNetwork? {
        let decoder = JSONDecoder()
        let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(fileName)
        guard let data = try? Data(contentsOf: url) else {
            print("Unable to read model from file.")
            return nil
        }
        guard let decoded = try? decoder.decode(NeuralNetwork.self, from: data) else {
            print("Unable to decode model.")
            return nil
        }
        return decoded
    }

    public init() {

    }

    public required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let wrappers = try container.decode([LayerWrapper].self, forKey: .layers)
        self.layers = wrappers.map { $0.layer }
        self.learningRate = try container.decode(Float.self, forKey: .learningRate)
        self.epochs = try container.decode(Int.self, forKey: .epochs)
        self.batchSize = try container.decode(Int.self, forKey: .batchSize)
    }

    public func saveModel(fileName: String) {
        let encoder = JSONEncoder()
        guard let encoded = try? encoder.encode(self) else {
            print("Unable to encode model.")
            return
        }
        let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(fileName)
        do {
            try encoded.write(to: url)
        } catch {
            print("Unable to write model to disk.")
        }
    }

    public func train(set: Dataset) -> Float {
        var error = Float.zero
        dropoutEnabled = true
        for epoch in 0..<epochs {
            var shuffledSet = set.items.shuffled()
            error = Float.zero
            while !shuffledSet.isEmpty {
                let batch = shuffledSet.prefix(self.batchSize)
                for item in batch {
                    let predictions = self.forward(networkInput: item.input)
                    for i in 0..<item.output.body.count {
                        error += pow(item.output.body[i]-predictions.body[i], 2)/2
                    }
                    self.backward(expected: item.output)
                    self.deltaWeights(row: item.input)
                }
                for layer in self.layers {
                    layer.updateWeights()
                }
                shuffledSet.removeFirst(min(self.batchSize, shuffledSet.count))
                if let testInput = testInput {
                    generateOutputMaps(input: testInput)
                    showOutputMaps()
                }
                usleep(useconds_t(delay * 1000))
            }
            print("Epoch \(epoch+1), error \(error).")
        }
        print(layers.first as? Dense)
        return error
    }

    public func predict(input: DataPiece) -> DataPiece {
        dropoutEnabled = false
        return forward(networkInput: input)
    }

    public func predictMax(input: DataPiece) -> Int {
        let output = predict(input: input)
        var maxi = 0
        for i in 1..<output.body.count where output.body[i] > output.body[maxi] {
            maxi = i
        }
        return maxi
    }

    private func deltaWeights(row: DataPiece) {
        var input = row
        for i in 0..<layers.count {
            input = layers[i].deltaWeights(input: input, learningRate: learningRate)
        }
    }

    private func forward(networkInput: DataPiece, savePoint: CGPoint? = nil) -> DataPiece {
        var input = networkInput
        for i in 0..<layers.count {
            input = layers[i].forward(input: input, dropoutEnabled: dropoutEnabled, savePoint: savePoint)
        }
        return input
    }

    private func backward(expected: DataPiece) {
        var input = expected
        var previous: Layer?
        for i in (0..<layers.count).reversed() {
            input = layers[i].backward(input: input, previous: previous)
            if !(layers[i] is Dropout) {
                previous = layers[i]
            }
        }
    }
}

public func classifierOutput(classes: Int, correct: Int) -> DataPiece {
    if correct>=classes {
        fatalError("Correct class must be less than classes number.")
    }
    var output = Array(repeating: Float.zero, count: classes)
    output[correct] = 1.0
    return DataPiece(size: .init(width: classes), body: output)
}
