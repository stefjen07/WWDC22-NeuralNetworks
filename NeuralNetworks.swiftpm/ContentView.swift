import SwiftUI
import SpriteKit

struct ContentView: View {
    let manager = NNViewManager(size: .init(width: 1000, height: 1000), preset: GaussianPreset())
    
    var body: some View {
        SpriteView(scene: manager.preset.neuralNetwork.trainScene)
            .onAppear {
                manager.showScene()
            }
    }
} 
