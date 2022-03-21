//
//  ActivationFunctions.swift
//  NeuralNetworksApp
//
//  Created by Yauheni on 3/22/22.
//

import Foundation

public enum ActivationFunction: Int, Codable {
    case sigmoid = 0
    case reLU
    case plain

    func activation(input: Float) -> Float {
        switch self {
        case .sigmoid:
            return 1.0/(1.0+exp(-input))
        case .reLU:
            return max(Float.zero, input)
        case .plain:
            return input
        }
    }

    func derivative(output: Float) -> Float {
        switch self {
        case .sigmoid:
            return output * (1.0-output)
        case .reLU:
            return output <= 0 ? 0 : 1
        case .plain:
            return 1
        }
    }
}
