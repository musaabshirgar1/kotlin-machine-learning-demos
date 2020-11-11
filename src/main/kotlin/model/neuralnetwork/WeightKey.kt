package model.neuralnetwork

data class WeightKey(
    val calculatedLayerIndex: Int,
    val feedingNodeIndex: Int,
    val nodeIndex: Int
)