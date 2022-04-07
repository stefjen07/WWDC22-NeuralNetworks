import Foundation
import SpriteKit

public enum PresetType: String, Identifiable, CaseIterable {
    public var id: Int {
        hashValue
    }
    
    func preset(with scene: SKScene) -> NNPreset {
        switch self {
        case .gaussian:
            return GaussianPreset(scene: scene)
        case .quarters:
            return QuartersPreset(scene: scene)
        case .circleInCircle:
            return CircleInCirclePreset(scene: scene)
        case .spiral:
            return SpiralPreset(scene: scene)
        }
    }
    
    case gaussian = "Gaussian"
    case quarters = "Quarters"
    case circleInCircle = "Circle-in-circle"
    case spiral = "Spiral"
}

public class NNSceneManager: ObservableObject {
    @Published var presetType: PresetType {
        didSet {
            self.preset.neuralNetwork.isTraining = false
            self.scene.removeAllChildren()
            preset.neuralNetwork.safeAction = {
                self.preset = self.presetType.preset(with: self.scene)
                self.startTraining()
            }
            
            if !preset.neuralNetwork.isTraining {
                preset.neuralNetwork.safeAction?()
            }
        }
    }
    
    @Published var epoch: Int = 0
    @Published var accuracy: Double = 0
    
    var scene: SKScene = .init(size: .init(width: 400, height: 800))
    
    private var preset: NNPreset
    
    private func showDatasetImage() {
        let node = SKSpriteNode(texture: preset.datasetImage())
        node.size = nodeCanvasSize
        
        let sceneRect = preset.neuralNetwork.trainScene.frame
        node.position = .init(x: sceneRect.width/2 - padding, y: -sceneRect.height/2 + padding)
        
        node.drawBorder(color: .black, width: 0.25)
        
        preset.neuralNetwork.trainScene.addChild(node)
    }
    
    public func showScene() {
        preset.neuralNetwork.generateScene()
        showDatasetImage()
        
        let queue = DispatchQueue(label: "networkQueue")
        queue.async { [weak self] in
            self?.preset.train()
        }
    }
    
    public func startTraining() {
        let camera = SKCameraNode()
        camera.zPosition = 100
        preset.neuralNetwork.trainScene.addChild(camera)
        preset.neuralNetwork.trainScene.camera = camera
        
        preset.neuralNetwork.statDelegate = { (stat) in
            (self.accuracy, self.epoch) = stat
        }
        
        showScene()
    }
    
    public init(presetType: PresetType) {
        self.presetType = presetType
        self.preset = presetType.preset(with: scene)
        self.scene = preset.neuralNetwork.trainScene
        
        startTraining()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}
