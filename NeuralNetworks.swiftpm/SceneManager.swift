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
            preset.neuralNetwork.safeAction = {
                self.scene.removeAllChildren()
                self.preset = self.presetType.preset(with: self.scene)
                self.startTraining()
            }
            
            if !preset.neuralNetwork.isTraining {
                preset.neuralNetwork.safeAction?()
            } else {
                preset.neuralNetwork.isTraining = false
            }
        }
    }
    
    @Published var epoch: Int = 0
    @Published var accuracy: Double = 0
    
    var scene: SKScene = .init(size: .init(width: 400, height: 800))
    
    private var preset: NNPreset
    
    private func showDatasetImage() {
        let node = SKSpriteNode(texture: preset.datasetImage())
        node.size = .init(width: preset.nodeCanvasSize.width * preset.resultMultiplier, height: preset.nodeCanvasSize.height * preset.resultMultiplier)
        
        let sceneRect = preset.neuralNetwork.trainScene.frame
        node.position = .init(x: sceneRect.width/2 - node.size.width/2, y: -sceneRect.height/2 + node.size.height/2)
        
        node.drawBorder(color: .black, width: 0.25)
        
        preset.neuralNetwork.trainScene.addChild(node)
    }
    
    public func showScene() {
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
        
        preset.neuralNetwork.padding = preset.padding
        preset.neuralNetwork.nodeCanvasSize = preset.nodeCanvasSize
        preset.neuralNetwork.resultMultiplier = preset.resultMultiplier
        
        preset.neuralNetwork.generateScene()
        
        preset.neuralNetwork.statDelegate = { (stat) in
            DispatchQueue.main.async {
                (self.accuracy, self.epoch) = stat
            }
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
