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

public struct FibonacciPreset: NNPreset {
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
        .init(decimal: 1, output: true),
        .init(decimal: 2, output: true),
        .init(decimal: 3, output: true),
        .init(decimal: 5, output: true),
        .init(decimal: 8, output: true),
        .init(decimal: 13, output: true),
        .init(decimal: 144, output: true),
        .init(decimal: 4, output: false),
        .init(decimal: 7, output: false),
        .init(decimal: 11, output: false),
        .init(decimal: 16, output: false),
        .init(decimal: 28, output: false),
        .init(decimal: 45, output: false),
        .init(decimal: 120, output: false),
    ])
    
    public init() {
        
    }
}
