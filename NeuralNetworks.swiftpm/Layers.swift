import Foundation
import SpriteKit

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

public class Layer: Codable {
    var neurons: [Neuron]
    var canvasSize: CGSize = .zero {
        didSet {
            neurons.forEach { neuron in
                neuron.canvasSize = canvasSize
            }
        }
    }
    fileprivate var function: ActivationFunction
    
    func modifyTexture(_ texture: SKMutableTexture, neuronIndex: Int) {
        DispatchQueue.main.async {
            texture.modifyPixelData { ptr, length in
                let pixelPtr = ptr?.assumingMemoryBound(to: RGBA.self)
                let pixelCount = Int(length / MemoryLayout<RGBA>.stride)
                let pixelBuffer = UnsafeMutableBufferPointer(start: pixelPtr, count: pixelCount)
                for (index, value) in self.neurons[neuronIndex].outputMap.reduce([], +).enumerated() {
                    let color = valueToColor(value)
                    if index < pixelCount {
                        pixelBuffer[index] = RGBA(color: color)
                    }
                }
            }
        }
    }
    
    func updateSynapses() {
        for neuron in neurons {
            for i in 0..<min(neuron.synapses.count, neuron.weights.count) {
                DispatchQueue.main.sync {
                    neuron.synapses[i].strokeColor = valueToColor(neuron.weights[i])
                }
            }
        }
    }
    
    func showOutputMaps() {
        neurons.indices.forEach { neuronIndex in
            modifyTexture(self.neurons[neuronIndex].texture, neuronIndex: neuronIndex)
        }
    }
    
    func forward(input: DataPiece, savePoint: CGPoint? = nil) -> DataPiece {
        return input
    }
    
    func backward(input: DataPiece?, previous: Layer?) -> DataPiece? {
        return input
    }
    
    func deltaWeights(input: DataPiece, learningRate: Float) -> DataPiece {
        return input
    }
    
    func updateWeights(batchSize: Int, learningRate: Float) {
        return
    }
    
    fileprivate init(function: ActivationFunction = .plain, neuronsCount: Int) {
        self.function = function
        self.neurons = Array(repeating: .init(weights: [], weightsDelta: [], bias: 0, biasDelta: 0), count: neuronsCount)
    }
}

public class FullyConnectedLayer: Layer {
    public init(inputSize: Int, neuronsCount: Int, function: ActivationFunction) {
        super.init(function: function, neuronsCount: neuronsCount)
        self.neurons = (0..<neuronsCount).map { _ in
            return Neuron(
                weights: (0..<inputSize).map { _ in
                    Float.random(in: -1.0 ... 1.0)
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
    
    override func forward(input: DataPiece, savePoint: CGPoint? = nil) -> DataPiece {
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
        
        if let savePoint = savePoint {
            neurons.forEach { neuron in
                neuron.outputMap[Int(savePoint.y)][Int(savePoint.x)] = neuron.output
            }
        }
        
        let output = neurons.map { $0.output }
        return DataPiece(size: .init(width: output.count), body: output)
    }
    
    override func backward(input: DataPiece?, previous: Layer?) -> DataPiece? {
        var errors = Array(repeating: Float.zero, count: neurons.count)
        if let previous = previous {
            for j in 0..<neurons.count {
                for neuron in previous.neurons {
                    errors[j] += neuron.weights[j] * neuron.biasDelta
                }
            }
        } else if let input = input {
            for j in 0..<neurons.count {
                errors[j] = neurons[j].output - input.body[j]
            }
        }
        for j in 0..<neurons.count {
            neurons[j].biasDelta = errors[j] * function.derivative(output: neurons[j].output)
        }
        return nil
    }
    
    override func deltaWeights(input: DataPiece, learningRate: Float) -> DataPiece {
        neurons.withUnsafeMutableBufferPointer { neuronsPtr in
            input.body.withUnsafeBufferPointer { inputPtr in
                DispatchQueue.concurrentPerform(iterations: neuronsPtr.count, execute: { i in
                    neuronsPtr[i].weightsDelta.withUnsafeMutableBufferPointer { deltaPtr in
                        DispatchQueue.concurrentPerform(iterations: deltaPtr.count, execute: { j in
                            deltaPtr[j] -= learningRate * neuronsPtr[i].biasDelta * inputPtr[j]
                        })
                        neuronsPtr[i].totalBiasDelta -= learningRate * neuronsPtr[i].biasDelta
                    }
                })
            }
        }
        let output = neurons.map { $0.output }
        return DataPiece(size: .init(width: output.count), body: output)
    }
    
    override func updateWeights(batchSize: Int, learningRate: Float) {
        let neuronsCount = neurons.count
        neurons.withUnsafeMutableBufferPointer { neuronsPtr in
            DispatchQueue.concurrentPerform(iterations: neuronsCount, execute: { i in
                neuronsPtr[i].weights.withUnsafeMutableBufferPointer { weightsPtr in
                    neuronsPtr[i].weightsDelta.withUnsafeMutableBufferPointer { deltaPtr in
                        let weightsCount = weightsPtr.count
                        DispatchQueue.concurrentPerform(iterations: weightsCount, execute: { j in
                            weightsPtr[j] += deltaPtr[j] / Float(batchSize)
                            deltaPtr[j] = 0
                        })
                        neuronsPtr[i].bias += neuronsPtr[i].totalBiasDelta / Float(batchSize)
                    }
                }
            })
        }
    }
}
