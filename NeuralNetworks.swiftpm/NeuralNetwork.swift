import SwiftUI
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

final public class NeuralNetwork {
    internal var layers: [Layer]
    private var lossFunction: LossFunction
    private var learningRate: Float
    private var epochs: Int
    private var batchSize: Int
    internal var trainScene: SKScene
    private var delay: Int
    internal var inputNeurons: [Neuron] = []
    private var inputs: [InputType]
    
    private var canvasRect: CGRect = .zero
    public var nodeCanvasSize: CGSize = .zero
    public var padding: CGFloat = .zero
    public var resultMultiplier: CGFloat = .zero
    
    public var isTraining: Bool = false
    public var safeAction: (() -> ())?
    public var statDelegate: (((accuracy: Double, epoch: Int)) -> ())?
    
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
        for neuronIndex in 0..<inputNeurons.count {
            inputNeurons[neuronIndex].texture.modifyPixelData { ptr, length in
                let pixelPtr = ptr?.assumingMemoryBound(to: RGBA.self)
                let pixelCount = Int(length / MemoryLayout<RGBA>.stride)
                let pixelBuffer = UnsafeMutableBufferPointer(start: pixelPtr, count: pixelCount)
                for (index, value) in self.normalMap(neuronIndex: neuronIndex).enumerated() {
                    let value = value * 255
                    if index < pixelCount {
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
    }
    
    func generateOutputMaps() {
        for point in pointsToCheck {
            let input = DataPiece(size: .init(width: inputs.count), body: inputs.map { $0.inputForPoint(point) })
            _ = forward(
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
                print("Fully connected layer: \(layer.neurons.count) neurons")
            default:
                break
            }
        }
    }
    
    public init(scene: SKScene, canvasRect: CGRect, inputs: [InputType], layers: [Layer], lossFunction: LossFunction, learningRate: Float, epochs: Int, batchSize: Int, delay: Int) {
        self.trainScene = scene
        self.inputs = inputs
        self.inputNeurons = (0..<inputs.count).map { _ in
            Neuron(weights: [], weightsDelta: [], bias: 0, biasDelta: 0)
        }
        self.canvasRect = canvasRect
        self.layers = layers
        self.lossFunction = lossFunction
        self.learningRate = learningRate
        self.epochs = epochs
        self.batchSize = batchSize
        self.delay = delay
        
        inputNeurons.forEach { neuron in
            neuron.canvasSize = canvasRect.size
        }
        layers.forEach { layer in
            layer.canvasSize = canvasRect.size
        }
    }
    
    public func train(set: Dataset) -> Float {
        if(isTraining) {
            return 0
        }
        
        isTraining = true
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
                        guessed += (predictions.body[i].rounded() == item.output.body[i].rounded()) ? 1 : 0
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
            for layer in layers {
                layer.updateSynapses()
            }
            error = lossFunction.cost(sum: error, outputSize: outputSize)
            statDelegate?((accuracy: Double(100 * guessed / Float(outputSize)), epoch: epoch+1))
            //print("Epoch \(epoch+1), error \(error), accuracy \(guessed / Float(outputSize))")
            
            if let safeAction = safeAction {
                safeAction()
                if !isTraining {
                    break
                }
            }
        }
        isTraining = false
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
