package model.neuralnetwork

import model.predictor.randomWeightValue
import tornadofx.singleAssign

sealed class Layer<N : Node> : Iterable<N> {
    abstract val nodes: List<N>
    override fun iterator() = nodes.iterator()
}
/**
 * A `CalculatedLayer` is used for the hidden and output layers,
 * and is derived off weights and values off each previous layer
 */
class CalculatedLayer(
    val index: Int,
    nodeCount: Int,
    val activationFunction: ActivationFunction
) : Layer<CalculatedNode>() {

    var feedingLayer: Layer<out Node> by singleAssign()

    override val nodes by lazy {
        (0 until nodeCount).asSequence()
            .map { CalculatedNode(it, this) }
            .toList()
    }
    /**
     * Weights are paired for feeding layer and this layer
     */
    val weights by lazy {
        (0 until feedingLayer.nodes.count())
            .asSequence()
            .flatMap { feedingNodeIndex ->
                (0 until nodeCount).asSequence()
                    .map { nodeIndex ->
                        WeightKey(
                            calculatedLayerIndex = index,
                            feedingNodeIndex = feedingNodeIndex,
                            nodeIndex = nodeIndex
                        ) to randomWeightValue()
                    }
            }.toMap().toMutableMap()
    }
    val biases by lazy {
        (0 until nodeCount).asSequence()
            .map {
                it to 0.0
            }.toMap().toMutableMap()
    }

    fun modifyWeight(
        key: WeightKey,
        adjustment: Double
    ) = weights.compute(key) { _, v ->
        v!! + adjustment
    }

    fun modifyBias(
        nodeId: Int,
        adjustment: Double
    ) = biases.compute(nodeId) { _, v ->
        v!! + adjustment
    }
}
/**
 * An `InputLayer` belongs to the first layer and accepts the
 * input values for each `model.neuralnetwork.InputNode`
 */
class InputLayer(nodeCount: Int) : Layer<InputNode>() {
    override val nodes = (0 until nodeCount).asSequence()
        .map { InputNode(it) }
        .toList()
}
