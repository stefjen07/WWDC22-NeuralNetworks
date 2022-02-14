import SpriteKit
import PlaygroundSupport
import Darwin

let view = SKView(frame: .init(origin: .zero, size: .init(width: 400, height: 800)))

let neuralNetwork = NeuralNetwork()
neuralNetwork.layers = [
    Dense(inputSize: 4, neuronsCount: 10, functionRaw: .sigmoid),
    Dense(inputSize: 10, neuronsCount: 10, functionRaw: .sigmoid),
    Dense(inputSize: 10, neuronsCount: 1, functionRaw: .sigmoid)
]
neuralNetwork.printSummary()
neuralNetwork.learningRate = 5
neuralNetwork.batchSize = 5
neuralNetwork.epochs = 200

/*let dataset = Dataset(items: [
    .init(input: [1, 0, 0, 0], inputSize: .init(width: 4), output: [0], outputSize: .init(width: 1)),
    .init(input: [0, 1, 0, 0], inputSize: .init(width: 4), output: [0], outputSize: .init(width: 1)),
    .init(input: [0, 0, 1, 0], inputSize: .init(width: 4), output: [1], outputSize: .init(width: 1)),
    .init(input: [0, 0, 0, 1], inputSize: .init(width: 4), output: [1], outputSize: .init(width: 1)),
    .init(input: [1, 1, 1, 1], inputSize: .init(width: 4), output: [0], outputSize: .init(width: 1)),
    .init(input: [1, 0, 0, 1], inputSize: .init(width: 4), output: [0], outputSize: .init(width: 1)),
])*/
let mnist = MNISTDataset(isTrain: true, url: playgroundSharedDataDirectory)
let mnistTest = MNISTDataset(isTrain: false, url: playgroundSharedDataDirectory)

do {
    try mnist.load()
    mnist.fillSet()

    try mnistTest.load()
    mnistTest.fillSet()
} catch {
    print(error.localizedDescription)
}


let camera = SKCameraNode()
camera.zPosition = 100
neuralNetwork.trainScene.addChild(camera)
neuralNetwork.trainScene.camera = camera

neuralNetwork.delay = 100 //ms

view.presentScene(neuralNetwork.trainScene)

PlaygroundSupport.PlaygroundPage.current.setLiveView(view)
PlaygroundSupport.PlaygroundPage.current.needsIndefiniteExecution = true

neuralNetwork.generateScene()

let queue = DispatchQueue(label: "networkQueue")

queue.async {
    neuralNetwork.train(set: mnist.set)
    print(neuralNetwork.predict(input: .init(size: .init(width: 4), body: [1, 0, 1, 0])))
}
