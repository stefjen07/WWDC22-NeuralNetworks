import Foundation
import CoreGraphics
import SpriteKit

public protocol NNPreset {
    var neuralNetwork: NeuralNetwork { get }
    var dataset: Dataset { get }
}

extension CGSize {
    var square: CGFloat {
        width * height
    }
}

extension NNPreset {
    func train() {
        neuralNetwork.train(set: dataset)
    }
    
    func datasetImage() -> SKTexture {
        var datasetMap = Array(
            repeating: Array(
                repeating: Float.nan,
                count: Int(canvasRect.width)
            ),
            count: Int(canvasRect.height)
        )
        
        print("Start filling map")
        
        for item in dataset.items {
            let x = Int(item.point!.x - canvasRect.minX)
            let y = Int(item.point!.y - canvasRect.minY)
            
            datasetMap[y][x] = item.output.body[0]
        }
        
        print("End filling map")
        
        let dataLength = 4 * Int(canvasRect.size.square)
        var data = Data(count: dataLength)
        data.withUnsafeMutableBytes { ptr in
            let pixelBuffer = ptr.bindMemory(to: RGBA.self)
            let pixelCount = Int(dataLength / MemoryLayout<RGBA>.stride)
            
            for (index, value) in datasetMap.reduce([], +).enumerated() {
                if value.isNaN {
                    pixelBuffer[index] = .white
                } else {
                    let normalizedValue = tanh(value) * 255
                    pixelBuffer[index] = RGBA(
                        red: UInt8(max(0, min(255, normalizedValue))),
                        green: 0,
                        blue: UInt8(max(0, min(255, 255 - normalizedValue))),
                        alpha: 255
                    )
                }
            }
        }
        
        let texture = SKTexture(data: data, size: canvasRect.size)
        return texture
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
    static let inputs: [InputType] = [.x, .y, .sinx, .siny]
    
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
            var y = CGFloat.random(in: yLowerbound...yUpperbound)
            if Bool.random() {
                y = -y
            }
            return CGPoint(x: x, y: y)
        },
        count: 300,
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
            FullyConnectedLayer(inputSize: inputs.count, neuronsCount: 8, function: .tanh),
            FullyConnectedLayer(inputSize: 8, neuronsCount: 1, function: .tanh)
        ],
        lossFunction: .meanSquared,
        learningRate: 0.03,
        epochs: 100,
        batchSize: 8,
        delay: 100
    )
    
    public var dataset: Dataset = Dataset(
        firstGenerator: {
            let radius = CGFloat.random(in: 0...canvasRect.width/2)
            let a: CGFloat = 0
            let b: CGFloat = 1
            let theta = (radius - a) / b
            return CGPoint(x: radius * cos(theta), y: radius * sin(theta))
        },
        secondGenerator: {
            let radius = CGFloat.random(in: 0...canvasRect.width/2)
            let a: CGFloat = 0
            let b: CGFloat = 1
            let theta = (radius - a) / b
            return CGPoint(x: -radius * cos(theta), y: -radius * sin(theta))
        },
        count: 400,
        inputs: inputs
    )

    public init() {

    }
}
