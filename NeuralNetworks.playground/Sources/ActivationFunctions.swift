//
//  ActivationFunctions.swift
//  NeuralNetworksApp
//
//  Created by Yauheni on 3/22/22.
//

import Foundation

extension Float {
    func tanh() -> Float {
        return Darwin.tanh(self)
    }
}

public enum ActivationFunction: Int, Codable {
    case sigmoid = 0
    case tanh
    case reLU
    case plain

    func transfer(input: Float) -> Float {
        switch self {
        case .sigmoid:
            return 1.0/(1.0+exp(-input))
        case .tanh:
            return input.tanh()
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
        case .tanh:
            return 1 - pow(output.tanh(), 2)
        case .reLU:
            return output < 0 ? 0 : 1
        case .plain:
            return 1
        }
    }
}
