import Foundation
import SpriteKit

extension NeuralNetwork {
    public func generatePositions() {
        let padding: CGFloat = 20
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
                let node = SKSpriteNode(texture: inputNeurons[j].texture)
                node.zPosition = 2
                node.position = position
                trainScene.addChild(node)
                inputNeurons[j].imageObject = node
            }
        }
        for i in 0..<layers.count {
            for j in 0..<layers[i].neurons.count {
                if let position = layers[i].neurons[j].position {
                    let imageNode = SKSpriteNode(texture: layers[i].neurons[j].texture)
                    imageNode.zPosition = 2
                    imageNode.position = position
                    trainScene.addChild(imageNode)
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
        generatePositions()
        generateObjects()
        generateInputMaps()
        generateSynapses()
    }

}

#if os(iOS)
typealias SystemColor = UIColor
#else
typealias SystemColor = NSColor
#endif

func weightToColor(_ value: CGFloat) -> SystemColor {
    return SystemColor(red: value, green: 0, blue: 1-value, alpha: 1)
}
