import Foundation
import CoreImage

public class MNISTDataset {
    public var set: Dataset = .init(items: [])
    private static let baseURL = "http://yann.lecun.com/exdb/mnist/"
    private static let names = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte" , "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
    let imagePrefixLength = 16
    let labelPrefixLength = 8
    var isTrain = false
    var url: URL
    public init(isTrain: Bool = true, url: URL) {
        self.isTrain = isTrain
        self.url = url
    }
    
    func dataPath(with name: String) -> String {
        return url.path + "/" + name
    }
    
    private func downloadFile(name: String, completion: @escaping (Data)->Void) throws {
        let path = dataPath(with: name)
        if FileManager.default.fileExists(atPath: path) {
            let data =  try Data(contentsOf: URL(fileURLWithPath: path))
            completion(data)
        } else {
            let url = URL(string: "\(Self.baseURL)\(name).gz")!
            print("Downloading \(url.absoluteString)")
            let request = URLRequest(url: url, timeoutInterval: 3600)
            URLSession.shared.dataTask(with: request) { (data, response, error) in
                if error == nil, let data = data {
                    FileManager.default.createFile(atPath: path + ".gz", contents: data, attributes: nil)
                    _ = execute(launchPath: gzipPath(), currentDirectory: Bundle.main.bundlePath, arguments: ["-d", path + ".gz"])
                    print("Save to \(path)")
                    let data =  try! Data(contentsOf: URL(fileURLWithPath: path))
                    completion(data)
                } else {
                    print("Error:\(error)")
                }
            }.resume()
            
        }
    }
    
    var imageData: Data!
    var labelData: Data!
    
    public func load() throws {
        let imageIndex = isTrain ? 0 : 2
        let labelIndex = isTrain ? 1 : 3
        let imageFileName = Self.names[imageIndex]
        let labelFileName = Self.names[labelIndex]
        let group = DispatchGroup()
        print("Downloading...")
        group.enter()
        try downloadFile(name: imageFileName) { [weak self] (data) in
            self?.imageData = data
            group.leave()
        }
        group.wait()
        group.enter()
        try downloadFile(name: labelFileName, completion: { [weak self] data in
            self?.labelData = data
            group.leave()
        })
        group.wait()
        print("Downloads ended")
    }
    
    var count: Int {
        return (imageData.count - imagePrefixLength) / (28 * 28)
    }
    
    public func fillSet() {
        var start = 0
        var end = 0
        for index in 0..<count {
            start = imagePrefixLength + index * 28 * 28
            end = start + 28*28
            let label = Int(labelData[labelPrefixLength + index])
            
            print("Image processing")
            if let cgImage = CIImage(bitmapData: imageData[start..<end], bytesPerRow: 28, size: .init(width: 28, height: 28), format: .R8, colorSpace: .init(name: CGColorSpace.linearGray)!).inverted.convertedCGImage {
                let sample = DataItem(input: DataPiece(image: cgImage), output: DataPiece(label: label, count: 10))
                set.items.append(sample)
            }
        }
    }
}

func execute(launchPath: String, currentDirectory: String? = nil, arguments: [String] = []) -> String {
    let pipe = Pipe()
    let file = pipe.fileHandleForReading
    let task = Process()
    task.launchPath = launchPath
    task.arguments = arguments
    task.standardOutput = pipe
    if let currentDirectory = currentDirectory  {
        task.currentDirectoryURL = URL(fileURLWithPath: currentDirectory)
    }
    task.launch()
    let data = file.readDataToEndOfFile()
    return String(data: data, encoding: .utf8)!
}

func gzipPath() -> String {
    return execute(launchPath: "/usr/bin/which", arguments: ["gzip"]).components(separatedBy: .newlines).first!
}

extension CIImage {
    var convertedCGImage: CGImage? {
        let context = CIContext(options: nil)
        return context.createCGImage(self, from: self.extent)
    }
    
    var inverted: CIImage {
        let filter = CIFilter(name: "CIColorInvert")!
        filter.setValue(self, forKey: kCIInputImageKey)
        
        return filter.outputImage ?? self
    }
    
    func resize(targetSize: CGSize) -> CIImage {
        let resizeFilter = CIFilter(name:"CILanczosScaleTransform")!

        let scale = targetSize.height / self.extent.height
        let aspectRatio = targetSize.width/(self.extent.width * scale)

        resizeFilter.setValue(self, forKey: kCIInputImageKey)
        resizeFilter.setValue(scale, forKey: kCIInputScaleKey)
        resizeFilter.setValue(aspectRatio, forKey: kCIInputAspectRatioKey)
        return resizeFilter.outputImage ?? self
    }
}
