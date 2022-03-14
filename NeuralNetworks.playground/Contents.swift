import SpriteKit
import PlaygroundSupport
import Darwin


//Epilepsy warning

let view = NNView(size: CGSize(width: 800, height: 800), preset: ParityPreset())

PlaygroundSupport.PlaygroundPage.current.setLiveView(view)
PlaygroundSupport.PlaygroundPage.current.needsIndefiniteExecution = true

view.showScene()
