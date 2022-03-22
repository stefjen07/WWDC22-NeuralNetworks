//
//  LossFunctions.swift
//  NeuralNetworksApp
//
//  Created by Yauheni on 3/22/22.
//

import Foundation

public enum LossFunction: Int, Codable {
    case meanSquared = 0
    case binary
    
    func loss(prediction: Float, expectation: Float) -> Float {
        switch self {
        case .meanSquared:
            return pow(expectation - prediction, 2)
        case .binary:
            return -(expectation * log(prediction) + (1 - expectation) * log(1 - prediction))
        }
    }
    
    func cost(sum: Float, outputSize: Int) -> Float {
        switch self {
        case .meanSquared:
            return sum / Float(outputSize)
        case .binary:
            return -sum / Float(outputSize)
        }
    }
}
