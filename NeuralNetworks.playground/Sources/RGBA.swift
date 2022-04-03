import Foundation

#if os(iOS)
import UIKit

extension UIColor {
    var redValue: CGFloat{ return CIColor(color: self).red }
    var greenValue: CGFloat{ return CIColor(color: self).green }
    var blueValue: CGFloat{ return CIColor(color: self).blue }
    var alphaValue: CGFloat{ return CIColor(color: self).alpha }
}
#endif

struct RGBA {
    static let white = RGBA(red: 255, green: 255, blue: 255, alpha: 255)
    
    var red: UInt8
    var green: UInt8
    var blue: UInt8
    var alpha: UInt8
    
    init(color: SystemColor) {
        #if os(iOS)
        self.init(red: UInt8(min(255, color.redValue * 255)), green: UInt8(min(255, color.greenValue * 255)), blue: UInt8(min(255, color.blueValue * 255)), alpha: UInt8(min(255, color.alphaValue * 255)))
        #else
        self.init(red: UInt8(min(255, color.redComponent * 255)), green: UInt8(min(255, color.greenComponent * 255)), blue: UInt8(min(255, color.blueComponent * 255)), alpha: UInt8(min(255, color.alphaComponent * 255)))
        #endif
    }
    
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
