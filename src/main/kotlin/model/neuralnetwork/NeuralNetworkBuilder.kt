package model.neuralnetwork

fun neuralNetwork(op: NeuralNetworkBuilder.() -> Unit): NeuralNetwork {
    val nn = NeuralNetworkBuilder()
    nn.op()
    return nn.build()
}

class NeuralNetworkBuilder {
    var input = 0
    var hidden = mutableListOf<HiddenLayerBuilder>()
    var output: HiddenLayerBuilder = HiddenLayerBuilder(0, ActivationFunction.RELU)

    class HiddenLayerBuilder(
        val nodeCount: Int,
        val activationFunction: ActivationFunction
    )

    fun inputLayer(nodeCount: Int) {
        input = nodeCount
    }

    fun hiddenLayer(
        nodeCount: Int,
        activationFunction: ActivationFunction
    ) {
        hidden.add(HiddenLayerBuilder(nodeCount, activationFunction))
    }

    fun outputLayer(
        nodeCount: Int,
        activationFunction: ActivationFunction
    ) {
        output = HiddenLayerBuilder(nodeCount, activationFunction)
    }

    fun build() = NeuralNetwork(input, hidden, output)
}