import SpriteKit
import PlaygroundSupport
import Darwin


//Epilepsy warning

let view = NNView(size: CGSize(width: 400, height: 800), preset: CorrectParityPreset())

PlaygroundSupport.PlaygroundPage.current.setLiveView(view)
PlaygroundSupport.PlaygroundPage.current.needsIndefiniteExecution = true

view.showScene()
