import SwiftUI
import SpriteKit

extension SKSpriteNode {
    func drawBorder(color: SystemColor, width: CGFloat) {
        let shapeNode = SKShapeNode(rect: .init(origin: .init(x: -frame.size.width / 2, y: -frame.size.height / 2), size: frame.size))
        shapeNode.fillColor = .clear
        shapeNode.strokeColor = color
        shapeNode.lineWidth = width
        addChild(shapeNode)
    }
}

extension NeuralNetwork {
    public func generatePositions() {
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
                node.size = nodeCanvasSize
                trainScene.addChild(node)
                inputNeurons[j].imageObject = node
                node.drawBorder(color: .black, width: 0.25)
            }
        }
        for i in 0..<layers.count {
            for j in 0..<layers[i].neurons.count {
                if let position = layers[i].neurons[j].position {
                    let imageNode = SKSpriteNode(texture: layers[i].neurons[j].texture)
                    imageNode.zPosition = 2
                    
                    if i == layers.count - 1 {
                        imageNode.size = .init(width: nodeCanvasSize.width * resultMultiplier, height: nodeCanvasSize.height * resultMultiplier)
                        imageNode.position = .init(x: position.x + imageNode.size.width/2 - padding, y: position.y + imageNode.size.height/2 - padding)
                    } else {
                        imageNode.position = position
                        imageNode.size = nodeCanvasSize
                    }
                    
                    trainScene.addChild(imageNode)
                    layers[i].neurons[j].imageObject = imageNode
                    imageNode.drawBorder(color: .black, width: 0.25)
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
        trainScene.backgroundColor = .systemBackground
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

func valueToColor(_ value: Float) -> SystemColor {
    let value = CGFloat(value)
    if value > 0 {
        let normalized = 1 - min(1, value)
        return SystemColor(red: 1, green: normalized, blue: normalized, alpha: 1)
    } else {
        let normalized = 1 - min(1, abs(value))
        return SystemColor(red: normalized, green: normalized, blue: 1, alpha: 1)
    }
}
