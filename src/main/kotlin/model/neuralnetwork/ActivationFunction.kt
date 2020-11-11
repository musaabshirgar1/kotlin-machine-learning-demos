package model.neuralnetwork

import kotlin.math.exp

enum class ActivationFunction {

    IDENTITY {
        override fun invoke(x: Double, otherValues: () -> DoubleArray) = x
    },
    SIGMOID {
        override fun invoke(x: Double, otherValues: () -> DoubleArray) = 1.0 / (1.0 + exp(-x))
    },
    TANH {
        override fun invoke(x: Double, otherValues: () -> DoubleArray) = kotlin.math.tanh(x)
    },
    RELU {
        override fun invoke(x: Double, otherValues: () -> DoubleArray) = if (x < 0.0) 0.0 else x
    },
    MAX {
        override fun invoke(x: Double, otherValues: () -> DoubleArray) = if (x == otherValues().maxOrNull()) x else 0.0
    },
    SOFTMAX {
        override fun invoke(x: Double, otherValues: () -> DoubleArray) =
            (exp(x) / otherValues().asSequence().map { exp(it) }.sum())
    };

    abstract fun invoke(x: Double, otherValues: () -> DoubleArray): Double
}