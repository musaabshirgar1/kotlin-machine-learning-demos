package model.neuralnetwork

import builder.NeuralNetworkBuilder
import org.apache.commons.math3.distribution.TDistribution
import org.nield.kotlinstatistics.randomFirst
import org.nield.kotlinstatistics.weightedCoinFlip
import kotlin.math.exp
import kotlin.math.pow

class NeuralNetwork(
    inputNodeCount: Int,
    hiddenLayers: List<NeuralNetworkBuilder.HiddenLayerBuilder>,
    outputLayer: NeuralNetworkBuilder.HiddenLayerBuilder
) {
    private val inputLayer = InputLayer(inputNodeCount)

    private val hiddenLayers = hiddenLayers.asSequence()
        .mapIndexed { index, hiddenLayer ->
            CalculatedLayer(
                index = index,
                nodeCount = hiddenLayer.nodeCount,
                activationFunction = hiddenLayer.activationFunction
            )
        }.toList().also { layers ->
            layers.withIndex().forEach { (i, layer) ->
                layer.feedingLayer = (if (i == 0) inputLayer else layers[i - 1])
            }
        }

    private val outputLayer =
        CalculatedLayer(
            index = hiddenLayers.count(),
            nodeCount = outputLayer.nodeCount,
            activationFunction = outputLayer.activationFunction
        ).also {
            it.feedingLayer = (if (this.hiddenLayers.isNotEmpty()) this.hiddenLayers.last() else inputLayer)
        }

    private val calculatedLayers = this.hiddenLayers.plusElement(this.outputLayer)

    /**
     * Input a set of training values for each node
     */
    fun trainEntriesHillClimbing(
        inputsAndTargets: Iterable<Pair<DoubleArray, DoubleArray>>
    ) {
        val entries = inputsAndTargets.toList()

        /**
         * Use simple hill climbing
         */
        var bestLoss = Double.MAX_VALUE

        val tDistribution = TDistribution(3.0)

        val allCalculatedNodes = calculatedLayers
            .asSequence()
            .flatMap {
                it.nodes.asSequence()
            }.toList()

        println("Training with ${entries.count()}")

        val learningRate = .1

        val weightsPlusBiasesIndices = calculatedLayers
            .asSequence()
            .map { it.weights.count() + it.biases.count() }
            .sum()
            .let { 0 until it }
            .toList().toIntArray()

        val weightCutOff = calculatedLayers
            .asSequence()
            .map { it.weights.count() }
            .sum() - 1

        repeat(1000) { epoch ->
            val randomVariableIndex = weightsPlusBiasesIndices.random()
            val randomlySelectedNode = allCalculatedNodes.randomFirst()
            val randomlySelectedFeedingNode = randomlySelectedNode.layer.feedingLayer.nodes.randomFirst()
            val selectedWeightKey = WeightKey(
                calculatedLayerIndex = randomlySelectedNode.layer.index,
                feedingNodeIndex = randomlySelectedFeedingNode.index,
                nodeIndex = randomlySelectedNode.index
            )

            val randomAdjust = if (randomVariableIndex <= weightCutOff) {
                val currentWeightValue = randomlySelectedNode.layer.weights[selectedWeightKey]!!
                val randomAdjust = (tDistribution.sample() * learningRate).let {
                    when {
                        currentWeightValue + it < -1.0 -> -1.0 - currentWeightValue
                        currentWeightValue + it > 1.0 -> 1.0 - currentWeightValue
                        else -> it
                    }
                }
                randomlySelectedNode.layer.modifyWeight(selectedWeightKey, randomAdjust)
                randomAdjust
            } else {
                val currentBiasValue = randomlySelectedNode.layer.biases[randomlySelectedNode.index]!!
                val randomAdjust = (tDistribution.sample() * learningRate).let {
                    when {
                        currentBiasValue + it < 0.0 -> 0.0 - currentBiasValue
                        currentBiasValue + it > 1.0 -> 1.0 - currentBiasValue
                        else -> it
                    }
                }
                randomlySelectedNode.layer.modifyBias(randomlySelectedNode.index, randomAdjust)
                randomAdjust
            }

            val totalLoss = entries
                .asSequence()
                .flatMap { (input, label) ->
                    label.asSequence()
                        .zip(predictEntry(input).asSequence()) { actual, predicted ->
                            (actual - predicted).pow(2)
                        }
                }.sum()

            if (totalLoss < bestLoss) {
                println("epoch $epoch: $bestLoss -> $totalLoss")
                bestLoss = totalLoss
            } else {
                if (randomVariableIndex <= weightCutOff) {
                    randomlySelectedNode.layer.modifyWeight(selectedWeightKey, -randomAdjust)
                } else {
                    randomlySelectedNode.layer.modifyBias(randomlySelectedNode.index, -randomAdjust)
                }
            }
        }
        calculatedLayers.forEach { println(it.weights) }
    }

    fun trainEntriesSimulatedAnnealing(
        inputsAndTargets: Iterable<Pair<DoubleArray, DoubleArray>>
    ) {
        val entries = inputsAndTargets.toList()

        /**
         *  Use simulated annealing
         */
        var bestLoss = Double.MAX_VALUE
        var currentLoss = bestLoss
        var bestConfig = calculatedLayers.map {
            it.index to it.weights.toMap()
        }.toMap()

        val tDistribution = TDistribution(3.0)

        val allCalculatedNodes = calculatedLayers
            .asSequence()
            .flatMap {
                it.nodes.asSequence()
            }.toList()

        println("Training with ${entries.count()}")

        val learningRate = .1

        val weightsPlusBiasesIndices = calculatedLayers
            .asSequence()
            .map { it.weights.count() + it.biases.count() }
            .sum()
            .let { 0 until it }
            .toList().toIntArray()

        val weightCutOff = calculatedLayers
            .asSequence()
            .map {
                it.weights.count()
            }.sum() - 1

        sequenceOf(
            generateSequence(80.0) { t ->
                t - .005
            }.takeWhile {
                it >= 0
            }).flatMap { it }.forEach { temp ->
            val randomVariableIndex = weightsPlusBiasesIndices.random()
            val randomlySelectedNode = allCalculatedNodes.randomFirst()
            val randomlySelectedFeedingNode = randomlySelectedNode.layer.feedingLayer.nodes.randomFirst()
            val selectedWeightKey = WeightKey(
                calculatedLayerIndex = randomlySelectedNode.layer.index,
                feedingNodeIndex = randomlySelectedFeedingNode.index,
                nodeIndex = randomlySelectedNode.index
            )

            val randomAdjust = if (randomVariableIndex <= weightCutOff) {
                val currentWeightValue = randomlySelectedNode.layer.weights[selectedWeightKey]!!
                val randomAdjust = (tDistribution.sample() * learningRate).let {
                    when {
                        currentWeightValue + it < -1.0 -> -1.0 - currentWeightValue
                        currentWeightValue + it > 1.0 -> 1.0 - currentWeightValue
                        else -> it
                    }
                }
                randomlySelectedNode.layer.modifyWeight(selectedWeightKey, randomAdjust)
                randomAdjust
            } else {
                val currentBiasValue = randomlySelectedNode.layer.biases[randomlySelectedNode.index]!!
                val randomAdjust = (tDistribution.sample() * learningRate).let {
                    when {
                        currentBiasValue + it < 0.0 -> 0.0 - currentBiasValue
                        currentBiasValue + it > 1.0 -> 1.0 - currentBiasValue
                        else -> it
                    }
                }
                randomlySelectedNode.layer.modifyBias(randomlySelectedNode.index, randomAdjust)
                randomAdjust
            }

            val newLoss = entries
                .asSequence()
                .flatMap { (input, label) ->
                    label.asSequence()
                        .zip(predictEntry(input).asSequence()) { actual, predicted ->
                            (actual - predicted).pow(2)
                        }
                }.sum()

            if (newLoss < currentLoss) {
                currentLoss = newLoss
                if (newLoss < bestLoss) {
                    println("temp $temp: $bestLoss -> $newLoss")
                    bestLoss = newLoss
                    bestConfig = calculatedLayers.asSequence().map {
                        it.index to it.weights.toMap()
                    }.toMap()
                }
            } else if (weightedCoinFlip(exp((-(newLoss - currentLoss)) / temp))) {
                currentLoss = newLoss
            } else {
                if (randomVariableIndex <= weightCutOff) {
                    randomlySelectedNode.layer.modifyWeight(selectedWeightKey, -randomAdjust)
                } else {
                    randomlySelectedNode.layer.modifyBias(randomlySelectedNode.index, -randomAdjust)
                }
            }
        }
        calculatedLayers.forEach { cl ->
            bestConfig[cl.index]?.forEach { w ->
                cl.weights[w.key] = w.value
            }
        }
        calculatedLayers.forEach {
            println(it.weights)
        }
    }

    fun predictEntry(inputValues: DoubleArray): DoubleArray {
        /**
         * Assign input values to input nodes
         */
        inputValues.withIndex().forEach { (i, v) ->
            inputLayer.nodes[i].value = v
        }
        /**
         * Calculate new hidden and output node values
         */
        return outputLayer.map { it.value }.toDoubleArray()
    }
}

