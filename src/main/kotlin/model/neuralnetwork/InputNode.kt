package model.neuralnetwork

sealed class Node(val index: Int) {
    abstract val value: Double
}

class InputNode(index: Int) : Node(index) {
    override var value = 0.0
}

class CalculatedNode(
    index: Int,
    val layer: CalculatedLayer
) : Node(index) {
    override val value: Double
        get() = layer.feedingLayer.asSequence()
            .map { feedingNode ->
                val weightKey = WeightKey(
                    calculatedLayerIndex = layer.index,
                    feedingNodeIndex = feedingNode.index,
                    nodeIndex = index
                )
                layer.weights[weightKey]!! * feedingNode.value
            }.plus(layer.biases[index]!!)
            .sum()
            .let { v ->
                layer.activationFunction.invoke(v) {
                    layer.asSequence().map { node ->
                        node.layer.feedingLayer.asSequence()
                            .map { feedingNode ->
                                val weightKey = WeightKey(
                                    calculatedLayerIndex = layer.index,
                                    feedingNodeIndex = feedingNode.index,
                                    nodeIndex = node.index
                                )
                                layer.weights[weightKey]!! * feedingNode.value
                            }.plus(layer.biases[node.index]!!).sum()
                    }.toList()
                        .toDoubleArray()
                }
            }
}