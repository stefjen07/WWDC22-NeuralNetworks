import Foundation
import CoreGraphics
import SpriteKit

public protocol NNPreset {
    var neuralNetwork: NeuralNetwork { get }
    var canvasRect: CGRect { get }
    var nodeCanvasSize: CGSize { get }
    var resultMultiplier: CGFloat { get }
    var dataset: Dataset { get }
}

extension NNPreset {
    var padding: CGFloat { 
        5 + nodeCanvasSize.width/2
    }
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
        
        for item in dataset.items {
            let x = Int(item.point!.x - canvasRect.minX)
            let y = Int(item.point!.y - canvasRect.minY)
            
            datasetMap[y][x] = item.output.body[0]
        }
        
        let dataLength = 4 * Int(canvasRect.size.square)
        var data = Data(count: dataLength)
        data.withUnsafeMutableBytes { ptr in
            let pixelBuffer = ptr.bindMemory(to: RGBA.self)
            let pixelCount = Int(dataLength / MemoryLayout<RGBA>.stride)
            
            for (index, value) in datasetMap.reduce([], +).enumerated() {
                if index < pixelCount {
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
        }
        
        let texture = SKTexture(data: data, size: canvasRect.size)
        return texture
    }
}

public struct GaussianPreset: NNPreset {
    public var canvasRect: CGRect
    public let nodeCanvasSize: CGSize = .init(width: 50, height: 50)
    public var resultMultiplier: CGFloat = 2

    let inputs: [InputType] = [.x, .y]
    public let neuralNetwork: NeuralNetwork
    public let dataset: Dataset
    let scene: SKScene
    
    public init(scene: SKScene) {
        self.scene = scene
        
        let canvasRect = CGRect(x: -10, y: -10, width: 20, height: 20) 
        
        self.neuralNetwork = NeuralNetwork(
            scene: scene, canvasRect: canvasRect,
            inputs: inputs,
            layers: [
                FullyConnectedLayer(inputSize: inputs.count, neuronsCount: 2, function: .tanh),
                FullyConnectedLayer(inputSize: 2, neuronsCount: 1, function: .sigmoid)
            ],
            lossFunction: .meanSquared,
            learningRate: 0.0003,
            epochs: 1000,
            batchSize: 1,
            delay: 0
        )
        
        self.dataset = Dataset(
            canvasRect: canvasRect,
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
        
        self.canvasRect = canvasRect
    }
}

public struct CircleInCirclePreset: NNPreset {
    public var canvasRect: CGRect
    public let nodeCanvasSize: CGSize = .init(width: 50, height: 50)
    public var resultMultiplier: CGFloat = 2
    
    let inputs: [InputType] = [.x, .y, .sinx, .siny]
    public let neuralNetwork: NeuralNetwork
    public let dataset: Dataset
    let scene: SKScene
    
    public init(scene: SKScene) {
        self.scene = scene
        
        let canvasRect = CGRect(x: -10, y: -10, width: 20, height: 20) 
        
        self.neuralNetwork = NeuralNetwork(
            scene: scene,
            canvasRect: canvasRect,
            inputs: inputs,
            layers: [
                FullyConnectedLayer(inputSize: inputs.count, neuronsCount: 4, function: .tanh),
                FullyConnectedLayer(inputSize: 4, neuronsCount: 4, function: .sigmoid),
                FullyConnectedLayer(inputSize: 4, neuronsCount: 1, function: .sigmoid)
            ],
            lossFunction: .meanSquared,
            learningRate: 0.3,
            epochs: 1000,
            batchSize: 16,
            delay: 0
        )
        
        self.dataset = Dataset(
            canvasRect: canvasRect,
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
            count: 400,
            inputs: inputs
        )
        
        self.canvasRect = canvasRect
    }
}

public struct QuartersPreset: NNPreset {
    public var canvasRect: CGRect
    public var resultMultiplier: CGFloat = 2

    public let canvasSize: CGSize = .init(width: 20, height: 20)
    public let nodeCanvasSize: CGSize = .init(width: 50, height: 50)
    
    let inputs: [InputType] = [.x, .y, .sinx, .siny]
    public let neuralNetwork: NeuralNetwork
    public let dataset: Dataset
    let scene: SKScene
    
    public init(scene: SKScene) {
        self.scene = scene
        
        let canvasRect = CGRect(x: -10, y: -10, width: 20, height: 20) 
        
        self.neuralNetwork = NeuralNetwork(
            scene: scene, 
            canvasRect: canvasRect,
            inputs: inputs,
            layers: [
                FullyConnectedLayer(inputSize: inputs.count, neuronsCount: 4, function: .tanh),
                FullyConnectedLayer(inputSize: 4, neuronsCount: 2, function: .tanh),
                FullyConnectedLayer(inputSize: 2, neuronsCount: 1, function: .sigmoid)
            ],
            lossFunction: .meanSquared,
            learningRate: 0.03,
            epochs: 1000,
            batchSize: 16,
            delay: 0
        )
        
        self.dataset = Dataset(
            canvasRect: canvasRect,
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
            count: 400,
            inputs: inputs
        )
        
        self.canvasRect = canvasRect
    }
}

public struct SpiralPreset: NNPreset {
    public let canvasRect: CGRect
    public var resultMultiplier: CGFloat = 4
    
    public let canvasSize: CGSize = .init(width: 20, height: 20)
    public let nodeCanvasSize: CGSize = .init(width: 20, height: 20)
    
    let inputs: [InputType] = [.x, .y, .x2, .y2, .sinx, .siny]
    public let neuralNetwork: NeuralNetwork
    public let dataset: Dataset
    let scene: SKScene
    
    public init(scene: SKScene) {
        self.scene = scene
        
        let canvasRect = CGRect(x: -10, y: -10, width: 20, height: 20) 
        
        self.neuralNetwork = NeuralNetwork(
            scene: scene, canvasRect: canvasRect,
            inputs: inputs,
            layers: [
                FullyConnectedLayer(inputSize: inputs.count, neuronsCount: 16, function: .tanh),
                FullyConnectedLayer(inputSize: 16, neuronsCount: 4, function: .tanh),
                FullyConnectedLayer(inputSize: 4, neuronsCount: 1, function: .sigmoid)
            ],
            lossFunction: .meanSquared,
            learningRate: 0.03,
            epochs: 1000,
            batchSize: 16,
            delay: 0
        )
        
        self.dataset = Dataset(
            canvasRect: canvasRect,
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
        
        self.canvasRect = canvasRect
    }
}
