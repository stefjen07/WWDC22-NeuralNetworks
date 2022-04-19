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
            .pickerStyle(.menu)
            SpriteView(scene: manager.scene)
                .aspectRatio(0.5, contentMode: .fit)
            Text("Epoch \(manager.epoch), training accuracy: \(manager.accuracy)%")
        }.padding(.all, 15)
    }
} 
