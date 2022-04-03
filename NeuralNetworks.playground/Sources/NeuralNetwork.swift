import Foundation
import CoreGraphics
import SpriteKit

public enum InputType: Int, Codable {
    case x
    case y
    case x2
    case y2
    case sinx
    case siny
    
    func inputForPoint(_ point: CGPoint) -> Float {
        switch self {
        case .x:
            return Float(point.x)
        case .y:
            return Float(point.y)
        case .x2:
            return Float(pow(point.x, 2))
        case .y2:
            return Float(pow(point.y, 2))
        case .sinx:
            return Float(sin(point.x))
        case .siny:
            return Float(sin(point.y))
        }
    }
}

final public class NeuralNetwork: Codable {
    public var layers: [Layer]
    public var lossFunction: LossFunction
    public var learningRate: Float
    public var epochs: Int
    public var batchSize: Int
    public var trainScene = SKScene(size: .init(width: 400, height: 800))
    public var delay: Int
    var inputNeurons: [Neuron] = []
    var inputs: [InputType]

    private enum CodingKeys: String, CodingKey {
        case inputs
        case layers
        case inputNeurons
        case lossFunction
        case learningRate
        case epochs
        case batchSize
        case delay
    }
    
    var pointsToCheck: [CGPoint] {
        return (0..<Int(canvasRect.height)).flatMap { y in
            (0..<Int(canvasRect.width)).map { x in
                return CGPoint(x: CGFloat(x) + canvasRect.minX, y: CGFloat(y) + canvasRect.minY)
            }
        }
    }
    
    func inputForPoint(_ point: CGPoint, neuronIndex: Int) -> Float {
        return inputs[neuronIndex].inputForPoint(point)
    }
    
    func normalMap(neuronIndex: Int) -> [CGFloat] {
        let inputs = pointsToCheck.map {
            return CGFloat(inputForPoint($0, neuronIndex: neuronIndex))
        }
        return inputs
    }
    
    func generateInputMaps() {
        (0..<inputNeurons.count).forEach { neuronIndex in
            inputNeurons[neuronIndex].texture.modifyPixelData { ptr, length in
                let pixelPtr = ptr?.assumingMemoryBound(to: RGBA.self)
                let pixelCount = Int(length / MemoryLayout<RGBA>.stride)
                let pixelBuffer = UnsafeMutableBufferPointer(start: pixelPtr, count: pixelCount)
                for (index, value) in self.normalMap(neuronIndex: neuronIndex).enumerated() {
                    let value = value * 255
                    pixelBuffer[index] = RGBA(
                        red: UInt8(max(0, min(255, value))),
                        green: 0,
                        blue: UInt8(max(0, min(255, 255 - value))),
                        alpha: 255
                    )
                }
            }
        }
    }

    func generateOutputMaps() {
        pointsToCheck.forEach { point in
            let input = DataPiece(size: .init(width: inputs.count), body: inputs.map { $0.inputForPoint(point) })
            forward(
                networkInput: input,
                savePoint: .init(x: point.x - canvasRect.minX, y: point.y - canvasRect.minY)
            )
        }
    }

    func showOutputMaps() {
        DispatchQueue.concurrentPerform(iterations: layers.count) { i in
            self.layers[i].showOutputMaps()
        }
    }

    public func printSummary() {
        for rawLayer in layers {
            switch rawLayer {
            case let layer as FullyConnectedLayer:
                print("FullyConnectedLayer layer: \(layer.neurons.count) neurons")
            default:
                break
            }
        }
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
    
    public init(inputs: [InputType], layers: [Layer], lossFunction: LossFunction, learningRate: Float, epochs: Int, batchSize: Int, delay: Int) {
        self.inputs = inputs
        self.inputNeurons = (0..<inputs.count).map { _ in
            Neuron(weights: [], weightsDelta: [], bias: 0, biasDelta: 0)
        }
        self.layers = layers
        self.lossFunction = lossFunction
        self.learningRate = learningRate
        self.epochs = epochs
        self.batchSize = batchSize
        self.delay = delay
    }

    public func encode(to encoder: Encoder) throws {
        let wrappers = layers.map { LayerWrapper($0) }
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(wrappers, forKey: .layers)
        try container.encode(inputNeurons, forKey: .inputNeurons)
        try container.encode(learningRate, forKey: .learningRate)
        try container.encode(epochs, forKey: .epochs)
        try container.encode(batchSize, forKey: .batchSize)
        try container.encode(delay, forKey: .delay)
        try container.encode(lossFunction, forKey: .lossFunction)
    }
    
    public required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.inputs = try container.decode([InputType].self, forKey: .inputs)
        let wrappers = try container.decode([LayerWrapper].self, forKey: .layers)
        self.layers = wrappers.map { $0.layer }
        self.inputNeurons = try container.decode([Neuron].self, forKey: .inputNeurons)
        self.learningRate = try container.decode(Float.self, forKey: .learningRate)
        self.epochs = try container.decode(Int.self, forKey: .epochs)
        self.batchSize = try container.decode(Int.self, forKey: .batchSize)
        self.delay = try container.decode(Int.self, forKey: .delay)
        self.lossFunction = try container.decode(LossFunction.self, forKey: .lossFunction)
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
        generateOutputMaps()
        showOutputMaps()
        var error: Float = 0
        for epoch in 0..<epochs {
            var shuffledSet = set.items.shuffled()
            error = 0
            var outputSize = 0
            var guessed: Float = 0
            while !shuffledSet.isEmpty {
                let batch = shuffledSet.prefix(self.batchSize)
                for item in batch {
                    let predictions = self.forward(networkInput: item.input)
                    for i in 0..<item.output.body.count {
                        guessed += (round(predictions.body[i]) == round(item.output.body[i])) ? 1 : 0
                        error += lossFunction.loss(prediction: predictions.body[i], expectation: item.output.body[i])
                        outputSize += 1
                    }
                    self.backward(expected: item.output)
                    self.deltaWeights(row: item.input)
                }
                for layer in self.layers {
                    layer.updateWeights(batchSize: batchSize, learningRate: learningRate)
                }
                shuffledSet.removeFirst(min(self.batchSize, shuffledSet.count))
                usleep(useconds_t(delay * 1000))
            }
            generateOutputMaps()
            showOutputMaps()
            error = lossFunction.cost(sum: error, outputSize: outputSize)
            print("Epoch \(epoch+1), error \(error), accuracy \(guessed / Float(outputSize))")
        }
        return error
    }

    public func predict(input: DataPiece) -> DataPiece {
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
            input = layers[i].forward(input: input, savePoint: savePoint)
        }
        return input
    }

    private func backward(expected: DataPiece) {
        var input: DataPiece? = expected
        var previous: Layer? = nil
        for i in (0..<layers.count).reversed() {
            input = layers[i].backward(input: input, previous: previous)
            previous = layers[i]
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
