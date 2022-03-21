import Foundation
import SpriteKit

let canvasSize = 50

public struct LayerWrapper: Codable {
    let layer: Layer
    
    private enum CodingKeys: String, CodingKey {
        case base
        case payload
    }

    private enum Base: Int, Codable {
        case dense = 0
        case dropout
    }
    
    init(_ layer: Layer) {
        self.layer = layer
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch layer {
        case let payload as Dense:
            try container.encode(Base.dense, forKey: .base)
            try container.encode(payload, forKey: .payload)
        case let payload as Dropout:
            try container.encode(Base.dropout, forKey: .base)
            try container.encode(payload, forKey: .payload)
        default:
            fatalError()
        }
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let base = try container.decode(Base.self, forKey: .base)
        
        switch base {
        case .dense:
            self.layer = try container.decode(Dense.self, forKey: .payload)
        case .dropout:
            self.layer = try container.decode(Dropout.self, forKey: .payload)
        }
    }

}

public class Layer: Codable {
    var maxWeight: Float = 0, minWeight: Float = 0, lastMaxWeight: Float = 0, lastMinWeight: Float = 0
    
    var neurons: [Neuron] = []
    var outputMap: [[[Float]]]
    fileprivate var function: ActivationFunction
    fileprivate var output: DataPiece?
    
    private enum CodingKeys: String, CodingKey {
        case neurons
        case function
        case output
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(function.rawValue, forKey: .function)
        try container.encode(neurons, forKey: .neurons)
        try container.encode(output, forKey: .output)
    }
    
    public required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let activationRaw = try container.decode(Int.self, forKey: .function)
        function = getActivationFunction(rawValue: activationRaw)
        neurons = try container.decode([Neuron].self, forKey: .neurons)
        outputMap = .init(
            repeating: .init(
                repeating: .init(
                    repeating: 0,
                    count: neurons.count
                ),
                count: canvasSize
            ),
            count: canvasSize
        )
        output = try container.decode(DataPiece.self, forKey: .output)
    }
    
    struct RGBA32: Equatable {
        var color: UInt32

        init(red: UInt8, green: UInt8, blue: UInt8, alpha: UInt8) {
            color = (UInt32(red) << 24) | (UInt32(green) << 16) | (UInt32(blue) << 8) | (UInt32(alpha) << 0)
        }
    }
    
    func modifyTexture(_ texture: SKMutableTexture, neuronIndex: Int) {
        texture.modifyPixelData { ptr, length in
            let pixelBuffer = ptr?.bindMemory(to: UInt32.self, capacity: canvasSize * canvasSize)
            for (index, value) in self.outputMap.reduce([], +).enumerated() {
                let value = tanh(value[neuronIndex]) * 255
                pixelBuffer?[index] = RGBA32(red: UInt8(max(0,min(255, value))), green: 0, blue: UInt8(max(0,min(255, 255 - value))), alpha: 255).color
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
    
    func updateWeights() {
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
                count: canvasSize
            ),
            count: canvasSize
        )
    }
}

public class Dense: Layer {
    
    public init(inputSize: Int, neuronsCount: Int, functionRaw: ActivationFunctionRaw) {
        let function = getActivationFunction(rawValue: functionRaw.rawValue)
        super.init(function: function, neuronsCount: neuronsCount)
        output = .init(size: .init(width: neuronsCount), body: Array(repeating: Float.zero, count: neuronsCount))
        self.neurons = Array(repeating: Neuron(weights: [], weightsDelta: .init(repeating: Float.zero, count: inputSize), bias: 0.0, biasDelta: 0.0), count: neuronsCount)
        for i in 0..<neuronsCount {
            /*var weights = [Float]()
            for _ in 0..<inputSize {
                weights.append(Float.random(in: -1.0 ... 1.0))
            }
            neurons[i].weights = weights*/
            neurons[i].weights = Array(repeating: Float.zero, count: inputSize)
        }
    }
    
    public required init(from decoder: Decoder) throws {
        try super.init(from: decoder)
    }
    
    override func forward(input: DataPiece, dropoutEnabled: Bool, savePoint: CGPoint? = nil) -> DataPiece {
        input.body.withUnsafeBufferPointer { inputPtr in
            output?.body.withUnsafeMutableBufferPointer { outputPtr in
                neurons.withUnsafeBufferPointer { neuronsPtr in
                    DispatchQueue.concurrentPerform(iterations: neuronsPtr.count, execute: { i in
                        var out = neuronsPtr[i].bias
                        neuronsPtr[i].weights.withUnsafeBufferPointer { weightsPtr in
                            DispatchQueue.concurrentPerform(iterations: neuronsPtr[i].weights.count, execute: { i in
                                out += weightsPtr[i] * inputPtr[i]
                            })
                        }
                        outputPtr[i] = function.activation(input: out)
                    })
                }
            }
        }
        if let savePoint = savePoint, let output = output {
            //print(savePoint)
            outputMap[Int(savePoint.x)][Int(savePoint.y)] = output.body
        }
        return output ?? input
    }
    
    override func backward(input: DataPiece, previous: Layer?) -> DataPiece {
        var errors = Array(repeating: Float.zero, count: neurons.count)
        if let previous = previous {
            for j in 0..<neurons.count {
                for neuron in previous.neurons {
                    errors[j] += neuron.weights[j]*neuron.biasDelta
                }
            }
        } else {
            for j in 0..<neurons.count {
                errors[j] = input.body[j] - output!.body[j]
            }
        }
        for j in 0..<neurons.count {
            neurons[j].biasDelta = errors[j] * function.derivative(output: output!.body[j])
        }
        return output ?? input
    }
    
    override func deltaWeights(input: DataPiece, learningRate: Float) -> DataPiece {
        neurons.withUnsafeMutableBufferPointer { neuronsPtr in
            input.body.withUnsafeBufferPointer { inputPtr in
                DispatchQueue.concurrentPerform(iterations: neuronsPtr.count, execute: { i in
                    neuronsPtr[i].weightsDelta.withUnsafeMutableBufferPointer { deltaPtr in
                        DispatchQueue.concurrentPerform(iterations: deltaPtr.count, execute: { j in
                            deltaPtr[j] += learningRate * neuronsPtr[i].biasDelta * inputPtr[j]
                        })
                        neuronsPtr[i].bias += learningRate * neuronsPtr[i].biasDelta
                    }
                })
            }
        }
        return output ?? input
    }
    
    override func updateWeights() {
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
                neuron.synapses[i].strokeColor = weightToColor(CGFloat((neuron.weights[i] - lastMinWeight) / lastMaxWeight))
            }
        }
    }
    
}

public class Dropout: Layer {
    var probability: Int
    var cache: [Bool]
    
    private enum CodingKeys: String, CodingKey {
        case probability
        case cache
    }
    
    public override func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(probability, forKey: .probability)
        try container.encode(cache, forKey: .cache)
        try super.encode(to: encoder)
    }
    
    public init(inputSize: Int, probability: Int) {
        self.probability = probability
        self.cache = Array(repeating: true, count: inputSize)
        super.init(function: Plain(), neuronsCount: 0)
        self.neurons = Array(repeating: Neuron(weights: [], weightsDelta: [], bias: 0.0, biasDelta: 0.0), count: inputSize)
        output = DataPiece(size: .init(width: inputSize), body: Array(repeating: Float.zero, count: inputSize))
    }
    
    public required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.probability = try container.decode(Int.self, forKey: .probability)
        self.cache = try container.decode([Bool].self, forKey: .cache)
        try super.init(from: decoder)
    }
    
    override func forward(input: DataPiece, dropoutEnabled: Bool, savePoint: CGPoint? = nil) -> DataPiece {
        output = input
        if dropoutEnabled {
            cache.withUnsafeMutableBufferPointer { cachePtr in
                output?.body.withUnsafeMutableBufferPointer { outputPtr in
                    DispatchQueue.concurrentPerform(iterations: outputPtr.count, execute: { i in
                        if Int.random(in: 0...100) < probability {
                            cachePtr[i] = false
                            outputPtr[i] = 0
                        } else {
                            cachePtr[i] = true
                        }
                    })
                }
            }
        }
        return output ?? input
    }
    
    override func backward(input: DataPiece, previous: Layer?) -> DataPiece {
        return output ?? input
    }
    
    override func deltaWeights(input: DataPiece, learningRate: Float) -> DataPiece {
        return output ?? input
    }
}

fileprivate func getActivationFunction(rawValue: Int) -> ActivationFunction {
    switch rawValue {
    case ActivationFunctionRaw.reLU.rawValue:
        return ReLU()
    case ActivationFunctionRaw.sigmoid.rawValue:
        return Sigmoid()
    default:
        return Plain()
    }
}

public enum ActivationFunctionRaw: Int {
    case sigmoid = 0
    case reLU
    case plain
}

protocol ActivationFunction: Codable {
    var rawValue: Int { get }
    func activation(input: Float) -> Float
    func derivative(output: Float) -> Float
}

fileprivate struct Sigmoid: ActivationFunction, Codable {
    public var rawValue: Int = 0
    
    public func activation(input: Float) -> Float {
        return 1.0/(1.0+exp(-input))
    }
    
    public func derivative(output: Float) -> Float {
        return output * (1.0-output)
    }
}

fileprivate struct ReLU: ActivationFunction, Codable {
    public var rawValue: Int = 1
    
    public func activation(input: Float) -> Float {
        return max(Float.zero, input)
    }
    
    public func derivative(output: Float) -> Float {
        return output < 0 ? 0 : 1
    }
}

fileprivate struct Plain: ActivationFunction, Codable {
    public var rawValue: Int = 2
    
    public func activation(input: Float) -> Float {
        return input
    }
    
    public func derivative(output: Float) -> Float {
        return 1
    }
}

#if DEBUG
func getActivationFunctionMirror(rawValue: Int) -> ActivationFunction {
    getActivationFunction(rawValue: rawValue)
}
#endif
