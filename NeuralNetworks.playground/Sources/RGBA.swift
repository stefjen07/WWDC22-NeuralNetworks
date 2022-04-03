import Foundation

struct RGBA {
    static let white = RGBA(red: 255, green: 255, blue: 255, alpha: 255)
    
    var red: UInt8
    var green: UInt8
    var blue: UInt8
    var alpha: UInt8

    init(red: UInt8, green: UInt8, blue: UInt8, alpha: UInt8) {
        let alphaScale = Float(alpha) / Float(UInt8.max)
        self.red = red.scaled(by: alphaScale)
        self.blue = blue.scaled(by: alphaScale)
        self.green = green.scaled(by: alphaScale)
        self.alpha = alpha
    }
}

extension UInt8 {
    func scaled(by scale: Float) -> UInt8 {
        var newValue = UInt(round(Float(self) * scale))
        newValue = Swift.min(newValue, UInt(UInt8.max))
        return UInt8(newValue)
    }
}
