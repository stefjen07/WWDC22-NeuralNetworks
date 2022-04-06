import Foundation
import SpriteKit

public class NNViewManager {
    var preset: NNPreset
    
    private func showDatasetImage() {
        let node = SKSpriteNode(texture: preset.datasetImage())
        node.drawBorder(color: .black, width: 0.25)
        
        let sceneRect = preset.neuralNetwork.trainScene.frame
        node.position = .init(x: sceneRect.width/2 - canvasRect.width, y: canvasRect.height - sceneRect.height/2)
        
        preset.neuralNetwork.trainScene.addChild(node)
    }
    
    public func showScene() {
        preset.neuralNetwork.generateScene()
        showDatasetImage()
        
        let queue = DispatchQueue(label: "networkQueue")
        queue.async {
            self.preset.train()
        }
    }
    
    public init(size: CGSize, preset: NNPreset) {
        self.preset = preset
        let camera = SKCameraNode()
        camera.zPosition = 100
        
        //preset.neuralNetwork.trainScene.size = .init(width: frame.size.width * 0.25, height: frame.size.height * 0.25)
        preset.neuralNetwork.trainScene.addChild(camera)
        preset.neuralNetwork.trainScene.camera = camera
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}
