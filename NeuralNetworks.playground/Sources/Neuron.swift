import Foundation
import SpriteKit

class Neuron: Codable {
    public enum CodingKeys: String, CodingKey {
        case weights
        case weightsDelta
        case bias
        case biasDelta
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

    var output: Float = 0
    var weights: [Float]
    var weightsDelta: [Float]
    var bias: Float
    var biasDelta: Float
    var imageObject: SKSpriteNode?
    var texture: SKMutableTexture = SKMutableTexture(
        size: canvasRect.size,
        pixelFormat: Int32(kCVPixelFormatType_32RGBA)
    )
    var synapses: [SKShapeNode] = []
    var position: CGPoint?
}
