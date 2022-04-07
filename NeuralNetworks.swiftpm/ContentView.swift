import SwiftUI
import SpriteKit

struct ContentView: View {
    @ObservedObject var manager = NNSceneManager(presetType: .gaussian)
    
    var body: some View {
        VStack {
            Picker("Neural network preset", selection: $manager.presetType) {
                ForEach(PresetType.allCases) { presetType in
                    Text(presetType.rawValue)
                        .tag(presetType)
                }
            }
            SpriteView(scene: manager.scene)
            Text("Epoch \(manager.epoch), training accuracy: \(manager.accuracy)%")
        }
    }
} 
