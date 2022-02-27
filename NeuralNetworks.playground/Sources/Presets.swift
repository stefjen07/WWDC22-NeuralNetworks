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

//Tic-Tac-Toe preset

public struct TicTacToePreset: NNPreset {
    public var neuralNetwork: NeuralNetwork = {
        let network = NeuralNetwork()
        network.layers = [
            Dense(inputSize: 4, neuronsCount: 10, functionRaw: .sigmoid),
            Dense(inputSize: 10, neuronsCount: 10, functionRaw: .sigmoid),
            Dense(inputSize: 10, neuronsCount: 1, functionRaw: .sigmoid)
        ]
        network.learningRate = 5
        network.batchSize = 5
        network.epochs = 200
        network.delay = 100
        return network
    }()
    public var dataset: Dataset = Dataset(items: [
        .init(input: [1, 0, 0, 0], inputSize: .init(width: 9), output: [0], outputSize: .init(width: 1)),
        .init(input: [0, 1, 0, 0], inputSize: .init(width: 9), output: [0], outputSize: .init(width: 1)),
        .init(input: [0, 0, 1, 0], inputSize: .init(width: 9), output: [1], outputSize: .init(width: 1)),
        .init(input: [0, 0, 0, 1], inputSize: .init(width: 9), output: [1], outputSize: .init(width: 1)),
        .init(input: [1, 1, 1, 1], inputSize: .init(width: 9), output: [0], outputSize: .init(width: 1)),
        .init(input: [1, 0, 0, 1], inputSize: .init(width: 9), output: [0], outputSize: .init(width: 1)),
    ])
    
    public init() {
        
    }
}

public struct PrimePreset: NNPreset {
    public var neuralNetwork: NeuralNetwork = {
        let network = NeuralNetwork()
        network.layers = [
            Dense(inputSize: 4, neuronsCount: 10, functionRaw: .sigmoid),
            Dense(inputSize: 10, neuronsCount: 10, functionRaw: .sigmoid),
            Dense(inputSize: 10, neuronsCount: 1, functionRaw: .sigmoid)
        ]
        network.learningRate = 5
        network.batchSize = 5
        network.epochs = 200
        network.delay = 100
        return network
    }()
    public var dataset: Dataset = Dataset(items: [
        .init(input: [1, 0, 0, 0], inputSize: .init(width: 4), output: [0], outputSize: .init(width: 1)),
        .init(input: [0, 1, 0, 0], inputSize: .init(width: 4), output: [0], outputSize: .init(width: 1)),
        .init(input: [0, 0, 1, 0], inputSize: .init(width: 4), output: [1], outputSize: .init(width: 1)),
        .init(input: [0, 0, 0, 1], inputSize: .init(width: 4), output: [1], outputSize: .init(width: 1)),
        .init(input: [1, 1, 1, 1], inputSize: .init(width: 4), output: [0], outputSize: .init(width: 1)),
        .init(input: [1, 0, 0, 1], inputSize: .init(width: 4), output: [0], outputSize: .init(width: 1)),
    ])
    
    public init() {
        
    }
}

public struct ParityPreset: NNPreset {
    public var neuralNetwork: NeuralNetwork = {
        let network = NeuralNetwork()
        network.layers = [
            Dense(inputSize: 4, neuronsCount: 10, functionRaw: .sigmoid),
            Dense(inputSize: 10, neuronsCount: 10, functionRaw: .sigmoid),
            Dense(inputSize: 10, neuronsCount: 1, functionRaw: .sigmoid)
        ]
        network.learningRate = 5
        network.batchSize = 5
        network.epochs = 200
        network.delay = 100
        return network
    }()
    
    public var dataset: Dataset = Dataset(items: [
        .init(input: [1, 0, 0, 0], inputSize: .init(width: 4), output: [1], outputSize: .init(width: 1)),
        .init(input: [0, 1, 0, 0], inputSize: .init(width: 4), output: [1], outputSize: .init(width: 1)),
        .init(input: [0, 0, 1, 0], inputSize: .init(width: 4), output: [1], outputSize: .init(width: 1)),
        .init(input: [0, 0, 0, 1], inputSize: .init(width: 4), output: [0], outputSize: .init(width: 1)),
        .init(input: [1, 1, 1, 1], inputSize: .init(width: 4), output: [0], outputSize: .init(width: 1)),
        .init(input: [1, 0, 0, 1], inputSize: .init(width: 4), output: [0], outputSize: .init(width: 1)),
    ])
    
    public init() {
        
    }
}

public struct CorrectParityPreset: NNPreset {
    public var neuralNetwork: NeuralNetwork = {
        let network = NeuralNetwork()
        network.layers = [
            Dense(inputSize: 4, neuronsCount: 1, functionRaw: .sigmoid)
        ]
        network.learningRate = 5
        network.batchSize = 5
        network.epochs = 200
        network.delay = 100
        return network
    }()
    
    public var dataset: Dataset = Dataset(items: [
        .init(input: [1, 0, 0, 0], inputSize: .init(width: 4), output: [1], outputSize: .init(width: 1)),
        .init(input: [0, 1, 0, 0], inputSize: .init(width: 4), output: [1], outputSize: .init(width: 1)),
        .init(input: [0, 0, 1, 0], inputSize: .init(width: 4), output: [1], outputSize: .init(width: 1)),
        .init(input: [0, 0, 0, 1], inputSize: .init(width: 4), output: [0], outputSize: .init(width: 1)),
        .init(input: [1, 1, 1, 1], inputSize: .init(width: 4), output: [0], outputSize: .init(width: 1)),
        .init(input: [1, 0, 0, 1], inputSize: .init(width: 4), output: [0], outputSize: .init(width: 1)),
    ])
    
    public init() {
        
    }
}
