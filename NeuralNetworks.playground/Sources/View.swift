import Foundation
import SpriteKit

public class NNView: SKView {
    var preset: NNPreset
    
    func setup() {
        let camera = SKCameraNode()
        camera.zPosition = 100
        
        preset.neuralNetwork.trainScene.size = self.frame.size
        preset.neuralNetwork.trainScene.addChild(camera)
        preset.neuralNetwork.trainScene.camera = camera
        
        preset.neuralNetwork.testInput = preset.dataset.items.first?.input
        
        self.presentScene(preset.neuralNetwork.trainScene)
    }
    
    public func showScene() {
        preset.neuralNetwork.generateScene()
        
        let queue = DispatchQueue(label: "networkQueue")

        queue.async {
            self.preset.train()
        }
    }
    
    public init(size: CGSize, preset: NNPreset) {
        self.preset = preset
        
        super.init(frame: .init(origin: .zero, size: size))
        
        setup()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}
