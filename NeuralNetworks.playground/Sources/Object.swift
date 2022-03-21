import Foundation
import SpriteKit

extension NeuralNetwork {
    
    public func generateInputNeurons() {
        inputNeurons = []
        if let layer = layers.first {
            if let neuron = layer.neurons.first {
                inputNeurons = Array(repeating: Neuron(weights: [], weightsDelta: [], bias: 0, biasDelta: 0), count: neuron.weights.count)
            }
        }
    }
    
    public func generatePositions() {
        let padding: CGFloat = 10
        let width = trainScene.size.width - padding * 2
        let height = trainScene.size.height - padding * 2
        let newHeight = height / CGFloat(layers.count + 1 - 1) // + 1 because of input neurons layer
        var y = -height / 2
        let inputWidth = width / CGFloat(inputNeurons.count - 1)
        var x = -width / 2
        for j in 0..<inputNeurons.count {
            inputNeurons[j].position = .init(x: x, y: -y)
            x += inputWidth
        }
        y += newHeight
        for i in 0..<layers.count {
            let newWidth = width / CGFloat(layers[i].neurons.count - 1)
            var x = -width / 2
            for j in 0..<layers[i].neurons.count {
                layers[i].neurons[j].position = .init(x: x, y: -y)
                x += newWidth
            }
            y += newHeight
        }
    }
    
    public func generateObjects() {
        for j in 0..<inputNeurons.count {
            if let position = inputNeurons[j].position {
                let node = SKShapeNode(circleOfRadius: 1)
                node.position = position
                node.fillColor = .black
                trainScene.addChild(node)
                inputNeurons[j].object = node
            }
        }
        for i in 0..<layers.count {
            for j in 0..<layers[i].neurons.count {
                if let position = layers[i].neurons[j].position {
                    let node = SKShapeNode(circleOfRadius: 20)
                    node.zPosition = 1
                    node.position = position
                    node.fillColor = .black
                    trainScene.addChild(node)
                    
                    let imageNode = SKSpriteNode(color: .black, size: .init(width: 40, height: 40))
                    imageNode.zPosition = 2
                    imageNode.position = .init(x: 0, y: 0)
                    imageNode.texture = layers[i].neurons[j].texture
                    node.addChild(imageNode)
                    
                    layers[i].neurons[j].object = node
                    layers[i].neurons[j].imageObject = imageNode
                }
            }
        }
    }
    
    public func generateSynapses() {
        if let layer = layers.first {
            for j in 0..<inputNeurons.count {
                for k in 0..<layer.neurons.count {
                    let path = CGMutablePath()
                    if let position = inputNeurons[j].position {
                        path.move(to: position)
                    }
                    if let position = layer.neurons[k].position {
                        path.addLine(to: position)
                    }
                    let node = SKShapeNode(path: path)
                    node.strokeColor = SKColor.red
                    node.lineWidth = 1
                    trainScene.addChild(node)
                    layer.neurons[k].synapses.append(node)
                }
            }
        }
        for i in 0..<layers.count-1 {
            for j in 0..<layers[i].neurons.count {
                for k in 0..<layers[i+1].neurons.count {
                    let path = CGMutablePath()
                    if let position = layers[i].neurons[j].position {
                        path.move(to: position)
                    }
                    if let position = layers[i+1].neurons[k].position {
                        path.addLine(to: position)
                    }
                    let node = SKShapeNode(path: path)
                    node.strokeColor = SKColor.red
                    node.lineWidth = 1
                    trainScene.addChild(node)
                    layers[i+1].neurons[k].synapses.append(node)
                }
            }
        }
    }
    
    public func generateScene() {
        trainScene.backgroundColor = .white
        generateInputNeurons()
        generatePositions()
        generateObjects()
        generateSynapses()
    }
    
}

#if os(iOS)
typealias SystemColor = UIColor
#else
typealias SystemColor = NSColor
#endif

func weightToColor(_ value: CGFloat) -> SystemColor {
    return SystemColor(red: 1-value, green: value, blue: 0, alpha: 1)
}
