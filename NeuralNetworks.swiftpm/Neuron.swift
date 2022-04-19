import SwiftUI
import SpriteKit

class Neuron: Codable {
    public enum CodingKeys: String, CodingKey {
        case weights
        case weightsDelta
        case bias
        case biasDelta
        case totalBiasDelta
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(weights, forKey: .weights)
        try container.encode(weightsDelta, forKey: .weightsDelta)
        try container.encode(bias, forKey: .bias)
        try container.encode(biasDelta, forKey: .biasDelta)
    }
    
    required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        weights = try container.decode([Float].self, forKey: .weights)
        weightsDelta = try container.decode([Float].self, forKey: .weightsDelta)
        bias = try container.decode(Float.self, forKey: .bias)
        biasDelta = try container.decode(Float.self, forKey: .biasDelta)
    }
    
    init(weights: [Float], weightsDelta: [Float], bias: Float, biasDelta: Float) {
        
        self.weights = weights
        self.weightsDelta = weightsDelta
        self.bias = bias
        self.biasDelta = biasDelta
    }
    
    var canvasSize: CGSize = .zero {
        didSet {
            self.outputMap = Array(
                repeating: Array(
                    repeating: 0,
                    count: Int(canvasSize.width)
                ),
                count: Int(canvasSize.height)
            )
            self.texture = SKMutableTexture(
                size: canvasSize,
                pixelFormat: Int32(kCVPixelFormatType_32RGBA)
            )
        }
    }
    var output: Float = 0
    var outputMap: [[Float]] = .init()
    var weights: [Float]
    var weightsDelta: [Float]
    var bias: Float
    var biasDelta: Float
    var totalBiasDelta: Float = 0
    var imageObject: SKSpriteNode?
    var texture: SKMutableTexture = .init()
    var synapses: [SKShapeNode] = []
    var position: CGPoint?
}
