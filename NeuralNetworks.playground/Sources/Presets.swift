import Foundation
import CoreGraphics

public protocol NNPreset {
    var neuralNetwork: NeuralNetwork { get }
    var dataset: Dataset { get }
}

extension NNPreset {
    func train() {
        neuralNetwork.train(set: dataset)
    }
}

public struct GaussianPreset: NNPreset {
    static let inputs: [InputType] = [.x, .y]
    
    public var neuralNetwork = NeuralNetwork(
        inputs: inputs,
        layers: [
            FullyConnectedLayer(inputSize: inputs.count, neuronsCount: 8, function: .sigmoid),
            FullyConnectedLayer(inputSize: 8, neuronsCount: 1, function: .sigmoid)
        ],
        lossFunction: .binary,
        learningRate: 3,
        epochs: 100,
        batchSize: 8,
        delay: 100
    )
    
    public var dataset: Dataset = Dataset(
        firstGenerator: {
            let x = CGFloat.random(in: canvasRect.minX...0)
            let y = CGFloat.random(in: canvasRect.minY...0)
            return CGPoint(x: x, y: y)
        },
        secondGenerator: {
            let x = CGFloat.random(in: 0...canvasRect.maxX)
            let y = CGFloat.random(in: 0...canvasRect.maxY)
            return CGPoint(x: x, y: y)
        },
        count: 100,
        inputs: inputs
    )

    public init() {

    }
}

public struct CircleInCirclePreset: NNPreset {
    static let inputs: [InputType] = [.x, .y]
    
    public var neuralNetwork = NeuralNetwork(
        inputs: inputs,
        layers: [
            FullyConnectedLayer(inputSize: inputs.count, neuronsCount: 4, function: .sigmoid),
            FullyConnectedLayer(inputSize: 4, neuronsCount: 4, function: .sigmoid),
            FullyConnectedLayer(inputSize: 4, neuronsCount: 1, function: .sigmoid)
        ],
        lossFunction: .binary,
        learningRate: 3,
        epochs: 100,
        batchSize: 8,
        delay: 100
    )
    
    public var dataset: Dataset = Dataset(
        firstGenerator: {
            let x = CGFloat.random(in: -canvasRect.width/4...canvasRect.width/4)
            let yLimit = sqrt(pow(canvasRect.width/4, 2) - pow(x, 2))
            let y = CGFloat.random(in: -yLimit...yLimit)
            return CGPoint(x: x, y: y)
        },
        secondGenerator: {
            let x = CGFloat.random(in: -canvasRect.width/2...canvasRect.width/2)
            let yUpperbound = sqrt(pow(canvasRect.width/2, 2) - pow(x, 2))
            let yLowerbound = (abs(x) > abs(canvasRect.width/4)) ? 0 : sqrt(pow(canvasRect.width/4, 2) - pow(x, 2))
            print(yLowerbound, yUpperbound)
            var y = CGFloat.random(in: yLowerbound...yUpperbound)
            if Bool.random() {
                y = -y
            }
            return CGPoint(x: x, y: y)
        },
        count: 100,
        inputs: inputs
    )

    public init() {

    }
}

public struct QuartersPreset: NNPreset {
    static let inputs: [InputType] = [.x, .y]
    
    public var neuralNetwork = NeuralNetwork(
        inputs: inputs,
        layers: [
            FullyConnectedLayer(inputSize: inputs.count, neuronsCount: 4, function: .sigmoid),
            FullyConnectedLayer(inputSize: 4, neuronsCount: 4, function: .sigmoid),
            FullyConnectedLayer(inputSize: 4, neuronsCount: 1, function: .sigmoid)
        ],
        lossFunction: .binary,
        learningRate: 3,
        epochs: 100,
        batchSize: 8,
        delay: 100
    )
    
    public var dataset: Dataset = Dataset(
        firstGenerator: {
            let x = CGFloat.random(in: canvasRect.minX...canvasRect.maxX)
            let absY = CGFloat.random(in: 0..<canvasRect.maxY)
            return CGPoint(x: x, y: x < 0 ? absY : -absY)
        },
        secondGenerator: {
            let x = CGFloat.random(in: canvasRect.minX...canvasRect.maxX)
            let absY = CGFloat.random(in: 0..<canvasRect.maxY)
            return CGPoint(x: x, y: x < 0 ? -absY : absY)
        },
        count: 100,
        inputs: inputs
    )

    public init() {

    }
}

public struct SpiralPreset: NNPreset {
    static let inputs: [InputType] = [.x, .y, .x2, .y2, .sinx, .siny]
    
    public var neuralNetwork = NeuralNetwork(
        inputs: inputs,
        layers: [
            FullyConnectedLayer(inputSize: inputs.count, neuronsCount: 8, function: .sigmoid),
            FullyConnectedLayer(inputSize: 8, neuronsCount: 1, function: .sigmoid)
        ],
        lossFunction: .binary,
        learningRate: 3,
        epochs: 100,
        batchSize: 1,
        delay: 100
    )
    
    public var dataset: Dataset = Dataset(
        firstGenerator: {
            let radius = CGFloat.random(in: canvasRect.minX...canvasRect.maxX)
            let a: CGFloat = 0
            let b: CGFloat = 1
            let theta = (radius - a) / b
            return CGPoint(x: radius * cos(theta), y: radius * sin(theta))
        },
        secondGenerator: {
            let radius = CGFloat.random(in: canvasRect.minX...canvasRect.maxX)
            let a: CGFloat = 0
            let b: CGFloat = 1
            let theta = (radius - a) / b
            return CGPoint(x: -radius * cos(theta), y: radius * sin(theta))
        },
        count: 100,
        inputs: inputs
    )

    public init() {

    }
}
