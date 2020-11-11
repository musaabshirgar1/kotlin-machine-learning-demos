package model.predictor

import builder.neuralNetwork
import javafx.beans.property.SimpleObjectProperty
import javafx.collections.FXCollections
import javafx.scene.paint.Color
import model.neuralnetwork.ActivationFunction
import model.neuralnetwork.NeuralNetwork
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Nesterovs
import org.ojalgo.ann.ArtificialNeuralNetwork
import org.ojalgo.array.Primitive64Array
import java.net.URL
import java.util.concurrent.ThreadLocalRandom

object PredictorModel {
    val inputs = FXCollections.observableArrayList<LabeledColor>()!!

    /**
     * Default Predictor
     */
    val selectedPredictor = SimpleObjectProperty(Predictor.OJALGO_NEURAL_NETWORK)

    fun predict(color: Color) = selectedPredictor.get().predict(color)

    operator fun plusAssign(
        labeledColor: LabeledColor
    ) {
        inputs += labeledColor
        Predictor.values().forEach {
            it.retrainFlag = true
        }
    }

    operator fun plusAssign(
        categorizedInput: Pair<Color, FontShade>
    ) {
        inputs += categorizedInput.let {
            LabeledColor(
                color = it.first,
                fontShade = it.second
            )
        }
        Predictor.values().forEach {
            it.retrainFlag = true
        }
    }

    fun preTrainData() {
        println("Pre Training Data With: ${selectedPredictor.value}")
        URL("https://tinyurl.com/y2qmhfsr")
            .readText()
            .split(Regex("\\r?\\n"))
            .asSequence()
            .drop(1)
            .filter {
                it.isNotBlank()
            }
            .map { s ->
                s.split(",").map {
                    it.toInt()
                }
            }
            .map {
                Color.rgb(
                    it[0],
                    it[1],
                    it[2]
                )
            }
            .map {
                LabeledColor(
                    color = it,
                    fontShade = Predictor.FORMULAIC.predict(it)
                )
            }.toList()
            .forEach {
                inputs += it
            }
        println("Input: $inputs")
        Predictor.values().forEach {
            it.retrainFlag = true
        }
    }

    enum class Predictor {

        /**
         * Uses a simple formula to classify colors as LIGHT or DARK
         */
        FORMULAIC {
            override fun predict(color: Color) = (0.299 * color.red + 0.587 * color.green + 0.114 * color.blue)
                .let { if (it > .5) FontShade.DARK else FontShade.LIGHT }
        },

//        LINEAR_REGRESSION_HILL_CLIMBING {
//            override fun predict(color: Color): FontShade {
//                var redWeightCandidate = 0.0
//                var greenWeightCandidate = 0.0
//                var blueWeightCandidate = 0.0
//
//                var currentLoss = Double.MAX_VALUE
//
//                val normalDistribution = NormalDistribution(0.0, 1.0)
//
//                fun predict(color: Color) =
//                    (redWeightCandidate * color.red + greenWeightCandidate * color.green + blueWeightCandidate * color.blue)
//
//                repeat(10000) {
//
//                    val selectedColor = (0..2).asSequence().randomFirst()
//                    val adjust = normalDistribution.sample()
//
//                    /**
//                     *  Make random adjustment to two of the colors
//                     */
//
//                    when (selectedColor) {
//                        0 -> redWeightCandidate += adjust
//                        1 -> greenWeightCandidate += adjust
//                        2 -> blueWeightCandidate += adjust
//                    }
//
//                    /**
//                     *  Calculate the loss, which is sum of squares
//                     */
//                    val newLoss = inputs.asSequence()
//                        .map { (color, fontShade) ->
//                            (predict(color) - fontShade.intValue).pow(2)
//                        }.sum()
//
//                    /**
//                     * If improvement doesn't happen, undo the move
//                     */
//                    if (newLoss < currentLoss) {
//                        currentLoss = newLoss
//                    } else {
//                        /**
//                         * revert if no improvement happens
//                         */
//                        when (selectedColor) {
//                            0 -> redWeightCandidate -= adjust
//                            1 -> greenWeightCandidate -= adjust
//                            2 -> blueWeightCandidate -= adjust
//                        }
//                    }
//                }
//
//                println("${redWeightCandidate}R + ${greenWeightCandidate}G + ${blueWeightCandidate}B")
//
//                val formulasLoss = inputs.asSequence()
//                    .map { (color, fontShade) ->
//                        ((0.299 * color.red + 0.587 * color.green + 0.114 * color.blue) - fontShade.intValue).pow(2)
//                    }.average()
//
//                println("BEST LOSS: $currentLoss, FORMULA'S LOSS: $formulasLoss \r\n")
//
//                return predict(color)
//                    .let { if (it > .5) FontShade.DARK else FontShade.LIGHT }
//            }
//        },

//        LOGISTIC_REGRESSION_HILL_CLIMBING {
//
//            var b0 = .01 // constant
//            var b1 = .01 // red beta
//            var b2 = .01 // green beta
//            var b3 = .01 // blue beta
//
//            private fun predictProbability(color: Color) =
//                1.0 / (1 + exp(-(b0 + b1 * color.red + b2 * color.green + b3 * color.blue)))
//
//            /**
//             * Helpful Resources:
//             * StatsQuest on YouTube: https://www.youtube.com/watch?v=yIYKR4sgzI8&list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe
//             * Brandon Foltz on YouTube: https://www.youtube.com/playlist?list=PLIeGtxpvyG-JmBQ9XoFD4rs-b3hkcX7Uu
//             */
//            override fun predict(color: Color): FontShade {
//                if (retrainFlag) {
//                    var bestLikelihood = -10_000_000.0
//
//                    /**
//                     * Use hill climbing for optimization
//                     */
//                    val normalDistribution = NormalDistribution(0.0, 1.0)
//
//                    b0 = .01 // constant
//                    b1 = .01 // red beta
//                    b2 = .01 // green beta
//                    b3 = .01 // blue beta
//
//                    /**
//                     * 1 = DARK FONT, 0 = LIGHT FONT
//                     */
//
//                    repeat(50000) {
//                        val selectedBeta = (0..3).asSequence().randomFirst()
//                        val adjust = normalDistribution.sample()
//
//                        /**
//                         * Make random adjustment to two of the colors
//                         */
//                        when (selectedBeta) {
//                            0 -> b0 += adjust
//                            1 -> b1 += adjust
//                            2 -> b2 += adjust
//                            3 -> b3 += adjust
//                        }
//
//                        /**
//                         * Calculate maximum likelihood
//                         */
//                        val darkEstimates = inputs.asSequence()
//                            .filter { it.fontShade == FontShade.DARK }
//                            .map { ln(predictProbability(it.color)) }
//                            .sum()
//
//                        val lightEstimates = inputs.asSequence()
//                            .filter { it.fontShade == FontShade.LIGHT }
//                            .map { ln(1 - predictProbability(it.color)) }
//                            .sum()
//
//                        val likelihood = darkEstimates + lightEstimates
//
//                        if (bestLikelihood < likelihood) {
//                            bestLikelihood = likelihood
//                        } else {
//                            /**
//                             * Revert if no improvement happens
//                             */
//                            when (selectedBeta) {
//                                0 -> b0 -= adjust
//                                1 -> b1 -= adjust
//                                2 -> b2 -= adjust
//                                3 -> b3 -= adjust
//                            }
//                        }
//                    }
//                    println("1.0 / (1 + exp(-($b0 + $b1*R + $b2*G + $b3*B))")
//                    println("BEST LIKELIHOOD: $bestLikelihood")
//                    retrainFlag = false
//                }
//                return predictProbability(color)
//                    .let { if (it > .5) FontShade.DARK else FontShade.LIGHT }
//            }
//        },

//        DECISION_TREE {
//
//            /**
//             * Helpful Resources:
//             * StatusQuest on YouTube: https://www.youtube.com/watch?v=7VeUPuFGJHk
//             */
//
//            inner class Feature(
//                private val name: String,
//                val mapper: (Color) -> Double
//            ) {
//                override fun toString() = name
//            }
//
//            private val features = listOf(
//                Feature("Red") { it.red * 255.0 },
//                Feature("Green") { it.green * 255.0 },
//                Feature("Blue") { it.blue * 255.0 }
//            )
//
//            private fun giniImpurity(samples: List<LabeledColor>): Double {
//                val totalSampleCount = samples.count().toDouble()
//                return 1.0 - (samples.count {
//                    it.fontShade == FontShade.DARK
//                }.toDouble() / totalSampleCount).pow(2) -
//                        (samples.count {
//                            it.fontShade == FontShade.LIGHT
//                        }.toDouble() / totalSampleCount).pow(2)
//            }
//
//            fun giniImpurityForSplit(feature: Feature, splitValue: Double, samples: List<LabeledColor>): Double {
//                val positiveFeatureSamples = samples.filter {
//                    feature.mapper(it.color) >= splitValue
//                }
//                val negativeFeatureSamples = samples.filter {
//                    feature.mapper(it.color) < splitValue
//                }
//                val positiveImpurity = giniImpurity(samples = positiveFeatureSamples)
//                val negativeImpurity = giniImpurity(samples = negativeFeatureSamples)
//                return (positiveImpurity * (positiveFeatureSamples.count().toDouble() / samples.count().toDouble())) +
//                        (negativeImpurity * (negativeFeatureSamples.count().toDouble() / samples.count().toDouble()))
//            }
//
//            private fun splitContinuousVariable(
//                feature: Feature,
//                samples: List<LabeledColor>
//            ): Double? {
//                val featureValues = samples.asSequence().map {
//                    feature.mapper(it.color)
//                }.distinct()
//                    .toList()
//                    .sorted()
//                return featureValues.asSequence().zipWithNext { value1, value2 ->
//                    (value1 + value2) / 2.0
//                }.minByOrNull {
//                    giniImpurityForSplit(feature = feature, splitValue = it, samples = samples)
//                }
//            }
//
//            inner class FeatureAndSplit(val feature: Feature, val split: Double)
//
//            fun buildLeaf(
//                samples: List<LabeledColor>,
//                previousLeaf: TreeLeaf? = null,
//                featureSampleSize: Int? = null
//            ): TreeLeaf? {
//                val fs = (if (featureSampleSize == null) features else features.random(featureSampleSize))
//                    .asSequence()
//                    .filter { splitContinuousVariable(it, samples) != null }
//                    .map { feature ->
//                        FeatureAndSplit(feature, splitContinuousVariable(feature, samples)!!)
//                    }.minByOrNull { fs ->
//                        giniImpurityForSplit(fs.feature, fs.split, samples)
//                    }
//
//                return if (previousLeaf == null ||
//                    (fs != null && giniImpurityForSplit(fs.feature, fs.split, samples) < previousLeaf.giniImpurity)
//                )
//                    TreeLeaf(fs!!.feature, fs.split, samples)
//                else
//                    null
//            }
//
//            inner class TreeLeaf(
//                private val feature: Feature,
//                private val splitValue: Double,
//                private val samples: List<LabeledColor>
//            ) {
//
//                private val goodWeatherItems = samples.filter {
//                    it.fontShade == FontShade.DARK
//                }
//                private val badWeatherItems = samples.filter {
//                    it.fontShade == FontShade.LIGHT
//                }
//
//                private val positiveItems = samples.filter {
//                    feature.mapper(it.color) >= splitValue
//                }
//                private val negativeItems = samples.filter {
//                    feature.mapper(it.color) < splitValue
//                }
//
//                val giniImpurity = giniImpurityForSplit(feature = feature, splitValue = splitValue, samples = samples)
//
//                val featurePositiveLeaf: TreeLeaf? =
//                    buildLeaf(samples.filter {
//                        feature.mapper(it.color) >= splitValue
//                    }, this)
//                val featureNegativeLeaf: TreeLeaf? =
//                    buildLeaf(samples.filter {
//                        feature.mapper(it.color) < splitValue
//                    }, this)
//
//
//                fun predict(color: Color): Double {
//                    val featureValue = feature.mapper(color)
//                    return when {
//                        featureValue >= splitValue ->
//                            when (featurePositiveLeaf) {
//                                null -> (goodWeatherItems.count {
//                                    feature.mapper(it.color) >= splitValue
//                                }.toDouble() / samples.count {
//                                    feature.mapper(it.color) >= splitValue
//                                }.toDouble())
//                                else -> featurePositiveLeaf.predict(color)
//                            }
//                        else ->
//                            when (featureNegativeLeaf) {
//                                null -> (goodWeatherItems.count {
//                                    feature.mapper(it.color) < splitValue
//                                }.toDouble() / samples.count {
//                                    feature.mapper(it.color) < splitValue
//                                }.toDouble())
//                                else -> featureNegativeLeaf.predict(color)
//                            }
//                    }
//                }
//
//                override fun toString() =
//                    "$feature split on $splitValue, ${negativeItems.count()}|${positiveItems.count()}, Impurity: $giniImpurity"
//            }
//
//            private fun recurseAndPrintTree(
//                leaf: TreeLeaf?,
//                depth: Int = 0
//            ) {
//                if (leaf != null) {
//                    println("\t".repeat(depth) + "($depth): $leaf")
//                    recurseAndPrintTree(leaf = leaf.featureNegativeLeaf, depth = depth + 1)
//                    recurseAndPrintTree(leaf = leaf.featurePositiveLeaf, depth = depth + 1)
//                }
//            }
//
//            override fun predict(color: Color): FontShade {
//                val tree = buildLeaf(inputs)
//                recurseAndPrintTree(tree)
//                return if (tree!!.predict(color) >= .5) FontShade.DARK else FontShade.LIGHT
//            }
//        },

//        RANDOM_FOREST {
//            /**
//             * Helpful Resources:
//             * StatusQuest on YouTube: https://www.youtube.com/watch?v=7VeUPuFGJHk
//             */
//
//            inner class Feature(
//                private val name: String,
//                val mapper: (Color) -> Double
//            ) {
//                override fun toString() = name
//            }
//
//            private val features = listOf(
//                Feature("Red") { it.red * 255.0 },
//                Feature("Green") { it.green * 255.0 },
//                Feature("Blue") { it.blue * 255.0 }
//            )
//
//            private fun giniImpurity(
//                samples: List<LabeledColor>
//            ): Double {
//                val totalSampleCount = samples.count().toDouble()
//                return 1.0 - (samples.count { it.fontShade == FontShade.DARK }.toDouble() / totalSampleCount).pow(2) -
//                        (samples.count { it.fontShade == FontShade.LIGHT }.toDouble() / totalSampleCount).pow(2)
//            }
//
//            fun giniImpurityForSplit(feature: Feature, splitValue: Double, samples: List<LabeledColor>): Double {
//                val positiveFeatureSamples = samples.filter {
//                    feature.mapper(it.color) >= splitValue
//                }
//                val negativeFeatureSamples = samples.filter {
//                    feature.mapper(it.color) < splitValue
//                }
//                val positiveImpurity = giniImpurity(samples = positiveFeatureSamples)
//                val negativeImpurity = giniImpurity(samples = negativeFeatureSamples)
//                return (positiveImpurity * (positiveFeatureSamples.count().toDouble() / samples.count().toDouble())) +
//                        (negativeImpurity * (negativeFeatureSamples.count().toDouble() / samples.count().toDouble()))
//            }
//
//            private fun splitContinuousVariable(feature: Feature, samples: List<LabeledColor>): Double? {
//                val featureValues = samples
//                    .asSequence()
//                    .map {
//                        feature.mapper(it.color)
//                    }.distinct()
//                    .toList()
//                    .sorted()
//
//                return featureValues.asSequence().zipWithNext { value1, value2 ->
//                    (value1 + value2) / 2.0
//                }.minByOrNull {
//                    giniImpurityForSplit(feature = feature, splitValue = it, samples = samples)
//                }
//            }
//
//            inner class FeatureAndSplit(
//                val feature: Feature,
//                val split: Double
//            )
//
//            fun buildLeaf(
//                samples: List<LabeledColor>,
//                previousLeaf: TreeLeaf? = null,
//                featureSampleSize: Int? = null
//            ): TreeLeaf? {
//
//                val fs = (if (featureSampleSize == null) features else features.random(featureSampleSize))
//                    .asSequence()
//                    .filter { splitContinuousVariable(it, samples) != null }
//                    .map { feature ->
//                        FeatureAndSplit(feature, splitContinuousVariable(feature, samples)!!)
//                    }.minByOrNull { fs ->
//                        giniImpurityForSplit(
//                            feature = fs.feature,
//                            splitValue = fs.split,
//                            samples = samples
//                        )
//                    }
//
//                return if (previousLeaf == null ||
//                    (fs != null && giniImpurityForSplit(fs.feature, fs.split, samples) < previousLeaf.giniImpurity)
//                )
//                    TreeLeaf(fs!!.feature, fs.split, samples)
//                else
//                    null
//            }
//
//            inner class TreeLeaf(
//                private val feature: Feature,
//                private val splitValue: Double,
//                private val samples: List<LabeledColor>
//            ) {
//
//                private val darkItems = samples.filter {
//                    it.fontShade == FontShade.DARK
//                }
//                private val lightItems = samples.filter {
//                    it.fontShade == FontShade.LIGHT
//                }
//
//                private val positiveItems = samples.filter {
//                    feature.mapper(it.color) >= splitValue
//                }
//                private val negativeItems = samples.filter {
//                    feature.mapper(it.color) < splitValue
//                }
//
//                val giniImpurity = giniImpurityForSplit(feature, splitValue, samples)
//
//                val featurePositiveLeaf: TreeLeaf? =
//                    buildLeaf(samples.filter { feature.mapper(it.color) >= splitValue }, this)
//                val featureNegativeLeaf: TreeLeaf? =
//                    buildLeaf(samples.filter { feature.mapper(it.color) < splitValue }, this)
//
//                fun predict(color: Color): Double {
//                    val featureValue = feature.mapper(color)
//                    return when {
//                        featureValue >= splitValue ->
//                            when (featurePositiveLeaf) {
//                                null -> (darkItems.count {
//                                    feature.mapper(it.color) >= splitValue
//                                }.toDouble() / samples.count {
//                                    feature.mapper(it.color) >= splitValue
//                                }.toDouble())
//                                else -> featurePositiveLeaf.predict(color)
//                            }
//                        else ->
//                            when (featureNegativeLeaf) {
//                                null -> (darkItems.count {
//                                    feature.mapper(it.color) < splitValue
//                                }.toDouble() / samples.count {
//                                    feature.mapper(it.color) < splitValue
//                                }.toDouble())
//                                else -> featureNegativeLeaf.predict(color)
//                            }
//                    }
//                }
//
//                override fun toString() =
//                    "$feature split on $splitValue, ${negativeItems.count()}|${positiveItems.count()}, Impurity: $giniImpurity"
//
//            }
//
//            fun recurseAndPrintTree(leaf: TreeLeaf?, depth: Int = 0) {
//                if (leaf != null) {
//                    println("\t".repeat(depth) + "($leaf)")
//                    recurseAndPrintTree(leaf = leaf.featureNegativeLeaf, depth = depth + 1)
//                    recurseAndPrintTree(leaf = leaf.featurePositiveLeaf, depth = depth + 1)
//                }
//            }
//
//            lateinit var randomForest: List<TreeLeaf>
//
//            override fun predict(color: Color): FontShade {
//                val bootStrapSampleCount = (inputs.count() * (2.0 / 3.0)).toInt()
//                if (retrainFlag) {
//                    randomForest = (1..300).asSequence()
//                        .map {
//                            buildLeaf(samples = inputs.random(bootStrapSampleCount), featureSampleSize = 2)!!
//                        }.toList()
//                    retrainFlag = false
//                }
//                val votes = randomForest.asSequence().countBy {
//                    if (it.predict(color) >= .5) FontShade.DARK else FontShade.LIGHT
//                }
//                println(votes)
//                return votes.maxByOrNull { it.value }!!.key
//            }
//        },

        NEURAL_NETWORK_HILL_CLIMBING {
            lateinit var artificialNeuralNetwork: NeuralNetwork

            override fun predict(color: Color): FontShade {
                println("Predicting Using NEURAL_NETWORK_HILL_CLIMBING")
                if (retrainFlag) {
                    println("Re-Train Flag: $retrainFlag")
                    artificialNeuralNetwork = neuralNetwork {
                        inputLayer(nodeCount = 3)
                        hiddenLayer(nodeCount = 3, activationFunction = ActivationFunction.TANH)
                        outputLayer(nodeCount = 2, activationFunction = ActivationFunction.SOFTMAX)
                    }
                    val trainingData = inputs.map {
                        colorAttributes(it.color) to it.fontShade.outputArray
                    }
                    artificialNeuralNetwork.trainEntriesHillClimbing(trainingData)
                    retrainFlag = false
                }
                return artificialNeuralNetwork.predictEntry(colorAttributes(color)).let {
                    println("${it[0]} ${it[1]}")
                    if (it[0] > it[1]) FontShade.LIGHT else FontShade.DARK
                }
            }
        },

        NEURAL_NETWORK_SIMULATED_ANNEALING {
            lateinit var artificialNeuralNetwork: NeuralNetwork
            override fun predict(color: Color): FontShade {
                println("Predicting Using NEURAL_NETWORK_SIMULATED_ANNEALING")
                if (retrainFlag) {
                    println("Re-Train Flag: $retrainFlag")
                    artificialNeuralNetwork = neuralNetwork {
                        inputLayer(nodeCount = 3)
                        hiddenLayer(nodeCount = 3, activationFunction = ActivationFunction.TANH)
                        outputLayer(nodeCount = 2, activationFunction = ActivationFunction.SOFTMAX)
                    }
                    val trainingData = inputs.map {
                        colorAttributes(it.color) to it.fontShade.outputArray
                    }
                    artificialNeuralNetwork.trainEntriesSimulatedAnnealing(trainingData)
                    retrainFlag = false
                }
                return artificialNeuralNetwork.predictEntry(colorAttributes(color)).let {
                    println("${it[0]} ${it[1]}")
                    if (it[0] > it[1]) FontShade.LIGHT else FontShade.DARK
                }
            }
        },

        OJALGO_NEURAL_NETWORK {
            lateinit var artificialNeuralNetwork: ArtificialNeuralNetwork
            override fun predict(color: Color): FontShade {
                println("Predicting Using OJALGO_NEURAL_NETWORK")
                if (retrainFlag) {
                    println("Re-Train Flag: $retrainFlag")
                    artificialNeuralNetwork = ArtificialNeuralNetwork
                        .builder(3, 3, 2)
                        .apply {
                            activator(0, ArtificialNeuralNetwork.Activator.RECTIFIER)
                            activator(1, ArtificialNeuralNetwork.Activator.SOFTMAX)

                            rate(.05)

                            error(ArtificialNeuralNetwork.Error.CROSS_ENTROPY)

                            val inputValues = inputs
                                .asSequence()
                                .map {
                                    Primitive64Array.FACTORY.copy(* colorAttributes(it.color))
                                }.toList()

                            val outputValues = inputs
                                .asSequence()
                                .map {
                                    Primitive64Array.FACTORY.copy(*it.fontShade.outputArray)
                                }.toList()

                            train(inputValues, outputValues)
                        }.get()

                    retrainFlag = false
                }

                return artificialNeuralNetwork.invoke(
                    Primitive64Array.FACTORY.copy(
                        *colorAttributes(color)
                    )
                ).let {
                    println("${it[0]} ${it[1]}")
                    if (it[0] > it[1]) FontShade.LIGHT else FontShade.DARK
                }
            }
        },

        /**
         * Uses DeepLearning4J, a heavyweight neural network library that is probably overkill for this toy problem.
         * However, DL4J is a good library to use for large real-world projects.
         */
        DL4J_NEURAL_NETWORK {
            override fun predict(color: Color): FontShade {

                val dl4jNN = NeuralNetConfiguration.Builder()
                    .weightInit(WeightInit.UNIFORM)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(Nesterovs(.006, .9))
                    .l2(1e-4)
                    .list(
                        DenseLayer.Builder().nIn(3).nOut(3).activation(Activation.RELU).build(),
                        OutputLayer.Builder().nIn(3).nOut(2).activation(Activation.SOFTMAX).build()
                    ).pretrain(false)
                    .backprop(true)
                    .build()
                    .let(::MultiLayerNetwork).apply { init() }

                val examples = inputs.asSequence()
                    .map { colorAttributes(it.color) }
                    .toList().toTypedArray()
                    .let { Nd4j.create(it) }

                val outcomes = inputs.asSequence()
                    .map { it.fontShade.outputArray }
                    .toList().toTypedArray()
                    .let { Nd4j.create(it) }

                /**
                 * Train for 1000 iterations (epochs)
                 */
                repeat(1000) {
                    dl4jNN.fit(examples, outcomes)
                }
                /**
                 * Test the input color and predict it as LIGHT or DARK
                 */
                val result = dl4jNN.output(Nd4j.create(colorAttributes(color))).toDoubleVector()
                println(result.joinToString(",  "))
                return if (result[0] > result[1]) FontShade.LIGHT else FontShade.DARK
            }
        };

        var retrainFlag = true

        abstract fun predict(color: Color): FontShade
        override fun toString() = name.replace("_", " ")
    }

}

/**
 *  UTILITIES
 */
fun randomInt(lower: Int, upper: Int) = ThreadLocalRandom.current().nextInt(lower, upper + 1)

fun randomWeightValue() = ThreadLocalRandom.current().nextDouble(-1.0, 1.0)

fun randomColor(): Color = (1..3).asSequence()
    .map { randomInt(0, 255) }
    .toList()
    .let { Color.rgb(it[0], it[1], it[2]) }

fun colorAttributes(c: Color) = doubleArrayOf(
    c.red,
    c.green,
    c.blue
)
