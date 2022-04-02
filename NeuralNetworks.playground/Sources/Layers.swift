import Foundation
import SpriteKit

let canvasRect = CGRect(x: -10, y: -10, width: 20, height: 20)

struct RGBA {
    var red: UInt8
    var green: UInt8
    var blue: UInt8
    var alpha: UInt8

    init(red: UInt8, green: UInt8, blue: UInt8, alpha: UInt8) {
        let alphaScale = Float(alpha) / Float(UInt8.max)
        self.red = red.scaled(by: alphaScale)
        self.blue = blue.scaled(by: alphaScale)
        self.green = green.scaled(by: alphaScale)
        self.alpha = alpha
    }
}

public struct LayerWrapper: Codable {
    let layer: Layer

    private enum CodingKeys: String, CodingKey {
        case base
        case payload
    }

    private enum Base: Int, Codable {
        case FullyConnectedLayer = 0
    }

    init(_ layer: Layer) {
        self.layer = layer
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch layer {
        case let payload as FullyConnectedLayer:
            try container.encode(Base.FullyConnectedLayer, forKey: .base)
            try container.encode(payload, forKey: .payload)
        default:
            fatalError()
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let base = try container.decode(Base.self, forKey: .base)

        switch base {
        case .FullyConnectedLayer:
            self.layer = try container.decode(FullyConnectedLayer.self, forKey: .payload)
        }
    }

}

extension UInt8 {
    func scaled(by value: Float) -> UInt8 {
        var scale = UInt(round(Float(self) * value))
        scale = Swift.min(scale, UInt(UInt8.max))
        return UInt8(scale)
    }
}

public class Layer: Codable {
    var maxWeight: Float = 0, minWeight: Float = 0, lastMaxWeight: Float = 0, lastMinWeight: Float = 0

    var neurons: [Neuron] = []
    var outputMap: [[[Float]]]
    fileprivate var function: ActivationFunction

    private enum CodingKeys: String, CodingKey {
        case neurons
        case function
        case output
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(function.rawValue, forKey: .function)
        try container.encode(neurons, forKey: .neurons)
    }

    public required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        function = try container.decode(ActivationFunction.self, forKey: .function)
        neurons = try container.decode([Neuron].self, forKey: .neurons)
        outputMap = .init(
            repeating: .init(
                repeating: .init(
                    repeating: 0,
                    count: neurons.count
                ),
                count: Int(canvasRect.width)
            ),
            count: Int(canvasRect.height)
        )
    }

    func modifyTexture(_ texture: SKMutableTexture, neuronIndex: Int) {
        DispatchQueue.main.async {
            texture.modifyPixelData { ptr, length in
                let pixelPtr = ptr?.assumingMemoryBound(to: RGBA.self)
                let pixelCount = Int(length / MemoryLayout<RGBA>.stride)
                let pixelBuffer = UnsafeMutableBufferPointer(start: pixelPtr, count: pixelCount)
                for (index, value) in self.outputMap.reduce([], +).enumerated() {
                    let value = tanh(value[neuronIndex]) * 255
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

    func showOutputMaps() {
        neurons.indices.forEach { neuronIndex in
            modifyTexture(self.neurons[neuronIndex].texture, neuronIndex: neuronIndex)
        }
    }

    func forward(input: DataPiece, dropoutEnabled: Bool, savePoint: CGPoint? = nil) -> DataPiece {
        return input
    }

    func backward(input: DataPiece, previous: Layer?) -> DataPiece {
        return input
    }

    func deltaWeights(input: DataPiece, learningRate: Float) -> DataPiece {
        return input
    }

    func updateWeights(learningRate: Float) {
        return
    }

    fileprivate init(function: ActivationFunction, neuronsCount: Int) {
        self.function = function
        outputMap = .init(
            repeating: .init(
                repeating: .init(
                    repeating: 0,
                    count: neuronsCount
                ),
                count: Int(canvasRect.width)
            ),
            count: Int(canvasRect.height)
        )
    }
}

public class FullyConnectedLayer: Layer {
    public init(inputSize: Int, neuronsCount: Int, function: ActivationFunction) {
        super.init(function: function, neuronsCount: neuronsCount)
        self.neurons = (0..<neuronsCount).map { _ in
            return Neuron(
                weights: (0..<inputSize).map { _ in
                    Float.random(in: -10.0 ... 10.0)
                },
                weightsDelta: .init(repeating: Float.zero, count: inputSize),
                bias: 0.0,
                biasDelta: 0.0
            )
        }
    }

    public required init(from decoder: Decoder) throws {
        try super.init(from: decoder)
    }

    override func forward(input: DataPiece, dropoutEnabled: Bool, savePoint: CGPoint? = nil) -> DataPiece {
        input.body.withUnsafeBufferPointer { inputPtr in
            neurons.withUnsafeBufferPointer { neuronsPtr in
                DispatchQueue.concurrentPerform(iterations: neuronsPtr.count, execute: { i in
                    var output = neuronsPtr[i].bias
                    neuronsPtr[i].weights.withUnsafeBufferPointer { weightsPtr in
                        DispatchQueue.concurrentPerform(iterations: neuronsPtr[i].weights.count, execute: { i in
                            output += weightsPtr[i] * inputPtr[i]
                        })
                    }
                    neuronsPtr[i].output = function.transfer(input: output)
                })
            }
        }
        let output = neurons.map { $0.output }
        if let savePoint = savePoint {
            outputMap[Int(savePoint.x)][Int(savePoint.y)] = output
        }
        return DataPiece(size: .init(width: output.count), body: output)
    }

    override func backward(input: DataPiece, previous: Layer?) -> DataPiece {
        var errors = Array(repeating: Float.zero, count: neurons.count)
        if let previous = previous {
            for j in 0..<neurons.count {
                for neuron in previous.neurons {
                    errors[j] += neuron.weights[j] * neuron.biasDelta
                }
            }
        } else {
            for j in 0..<neurons.count {
                errors[j] = neurons[j].output - input.body[j]
            }
        }
        for j in 0..<neurons.count {
            neurons[j].biasDelta = errors[j] * function.derivative(output: neurons[j].output)
        }
        return DataPiece(size: .init(width: 0), body: [])
    }

    override func deltaWeights(input: DataPiece, learningRate: Float) -> DataPiece {
        neurons.withUnsafeMutableBufferPointer { neuronsPtr in
            input.body.withUnsafeBufferPointer { inputPtr in
                DispatchQueue.concurrentPerform(iterations: neuronsPtr.count, execute: { i in
                    neuronsPtr[i].weightsDelta.withUnsafeMutableBufferPointer { deltaPtr in
                        DispatchQueue.concurrentPerform(iterations: deltaPtr.count, execute: { j in
                            deltaPtr[j] -= learningRate * neuronsPtr[i].biasDelta * inputPtr[j]
                        })
                    }
                })
            }
        }
        let output = neurons.map { $0.output }
        return DataPiece(size: .init(width: output.count), body: output)
    }

    override func updateWeights(learningRate: Float) {
        let neuronsCount = neurons.count
        neurons.withUnsafeMutableBufferPointer { neuronsPtr in
            DispatchQueue.concurrentPerform(iterations: neuronsCount, execute: { i in
                neuronsPtr[i].weights.withUnsafeMutableBufferPointer { weightsPtr in
                    neuronsPtr[i].weightsDelta.withUnsafeMutableBufferPointer { deltaPtr in
                        let weightsCount = weightsPtr.count
                        DispatchQueue.concurrentPerform(iterations: weightsCount, execute: { j in
                            weightsPtr[j] += deltaPtr[j]
                            deltaPtr[j] = 0
                        })
                        neuronsPtr[i].bias -= learningRate * neuronsPtr[i].biasDelta
                    }
                }
            })
        }
        lastMinWeight = minWeight
        lastMaxWeight = maxWeight
        minWeight = 0
        maxWeight = 0.5
        for neuron in neurons {
            for i in 0..<min(neuron.synapses.count, neuron.weights.count) {
                minWeight = min(minWeight, neuron.weights[i])
                maxWeight = max(maxWeight, neuron.weights[i])
                DispatchQueue.main.async {
                    neuron.synapses[i].strokeColor = weightToColor(
                        CGFloat((neuron.weights[i] - self.lastMinWeight) / self.lastMaxWeight)
                    )
                }
            }
        }
    }
}
