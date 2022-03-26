import Foundation

public protocol NNPreset {
    var neuralNetwork: NeuralNetwork { get }
    var dataset: Dataset { get }
}

extension NNPreset {
    func train() {
        neuralNetwork.train(set: dataset)
    }
}

public struct LinearPreset: NNPreset {
    static let inputs: [InputType] = [.x, .y, .x2, .y2]
    
    public var neuralNetwork = NeuralNetwork(
        inputs: inputs,
        layers: [
            Dense(inputSize: inputs.count, neuronsCount: 2, function: .sigmoid),
            Dense(inputSize: 2, neuronsCount: 1, function: .sigmoid)
        ],
        lossFunction: .binary,
        learningRate: 5,
        epochs: 100,
        batchSize: 8,
        delay: 100
    )
    
    public var dataset: Dataset = Dataset(predicator: { ($0.x + $0.y < 15) ? 1 : 0 }, inputs: inputs)

    public init() {

    }
}

public struct QuadPreset: NNPreset {
    static let inputs: [InputType] = [.x, .y, .x2, .y2, .sinx, .siny]
    
    public var neuralNetwork = NeuralNetwork(
        inputs: inputs,
        layers: [
            Dense(inputSize: inputs.count, neuronsCount: 4, function: .tanh),
            Dense(inputSize: 4, neuronsCount: 2, function: .tanh),
            Dense(inputSize: 2, neuronsCount: 1, function: .tanh)
        ],
        lossFunction: .binary,
        learningRate: 5,
        epochs: 100,
        batchSize: 8,
        delay: 100
    )
    
    public var dataset: Dataset = Dataset(predicator: { (pow($0.x, 2) + pow($0.y, 2) < 100) ? 1 : 0 }, inputs: inputs)

    public init() {

    }
}

public struct LinePreset: NNPreset {
    static let inputs: [InputType] = [.x, .y, .x2, .y2]
    
    public var neuralNetwork = NeuralNetwork(
        inputs: inputs,
        layers: [
            Dense(inputSize: inputs.count, neuronsCount: 4, function: .sigmoid),
            Dense(inputSize: 4, neuronsCount: 4, function: .sigmoid),
            Dense(inputSize: 4, neuronsCount: 1, function: .sigmoid)
        ],
        lossFunction: .binary,
        learningRate: 5,
        epochs: 100,
        batchSize: 8,
        delay: 100
    )
    
    public var dataset: Dataset = Dataset(predicator: { (abs($0.x-$0.y) < 3) ? 1 : 0 }, inputs: inputs)

    public init() {

    }
}
