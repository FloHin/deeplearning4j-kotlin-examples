package examples.a_mnist

import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.evaluation.classification.ROCMultiClass
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
 * https://deeplearning4j.konduit.ai/getting-started/tutorials/quickstart-with-mnist
 *
 * In this quickstart, you will create a deep neural network using Deeplearning4j
 * and train a model capable of classifying random handwriting digits.
 * While handwriting recognition has been attempted by different machine
 * learning algorithms over the years, deep learning performs remarkably
 * well and achieves an accuracy of over 99.7% on the MNIST dataset. For this tutorial, we will classify digits in EMNIST, the “next generation” of MNIST and a larger dataset.
 *
 * 1. Load a dataset for a neural network.
 * 2. Format EMNIST for image recognition.
 * 3. Create a deep neural network.
 * 4. Train a model.
 * 5. Evaluate the performance of your model.
 *
 */
class MNISTExample : Example {

    override fun run() {
        val batchSize = 128 // how many examples to simultaneously train in the network
        val emnistSet = EmnistDataSetIterator.Set.BALANCED
        val emnistTrain = EmnistDataSetIterator(emnistSet, batchSize, true)
        val emnistTest = EmnistDataSetIterator(emnistSet, batchSize, false)

        val conf = logTime("1. Build Network") {
            buildNetwork(emnistSet)
        }

        val network = logTime("2. Train model") {
            trainModel(conf, emnistTrain, 5)
        }

        logTime("3. Evaluate model") {
            evaluateModel(network, emnistTest)
        }
    }

    private fun buildNetwork(emnistSet: EmnistDataSetIterator.Set): MultiLayerConfiguration {
        //Build the neural network
        val outputNum = EmnistDataSetIterator.numLabels(emnistSet) // total output classes
        val rngSeed = 123L // integer for reproducability of a random number generator
        val numRows = 28 // number of "pixel rows" in an mnist digit
        val numColumns = 28
        val conf = NeuralNetConfiguration.Builder()
            .seed(rngSeed)
            .updater(Adam())
            .l2(1e-4)
            .list()
            .layer(
                DenseLayer.Builder()
                    .nIn(numRows * numColumns) // Number of input datapoints.
                    .nOut(1000) // Number of output datapoints.
                    .activation(Activation.RELU) // Activation function.
                    .weightInit(WeightInit.XAVIER) // Weight initialization.
                    .build()
            )
            .layer(
                OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nIn(1000)
                    .nOut(outputNum)
                    .activation(Activation.SOFTMAX)
                    .weightInit(WeightInit.XAVIER)
                    .build()
            )
            .build()

        return conf
    }

    private fun trainModel(
        conf: MultiLayerConfiguration,
        emnistTrain: EmnistDataSetIterator,
        epochs: Int
    ): MultiLayerNetwork {
        // create the MLN
        val network = MultiLayerNetwork(conf)
        network.init()

        // pass a training listener that reports score every 10 iterations
        val eachIterations = 10
        network.addListeners(ScoreIterationListener(eachIterations))

        // fit a dataset for a single epoch
        // network.fit(emnistTrain)

        // fit for multiple epochs
        // val numEpochs = 2
        // network.fit(emnistTrain, numEpochs)

        // or simply use for loop
        for (i in 1..epochs) {
            println("Epoch $i / $epochs")
            network.fit(emnistTrain)
        }

        return network
    }

    fun evaluateModel(network: MultiLayerNetwork, emnistTest: EmnistDataSetIterator) {
        // evaluate basic performance
        val eval: Evaluation = network.evaluate(emnistTest)
        println("accuracy: ${eval.accuracy()}")
        println("precision: ${eval.precision()}")
        println("recall: ${eval.recall()}")

        // evaluate ROC and calculate the Area Under Curve
        val roc: ROCMultiClass = network.evaluateROCMultiClass(emnistTest, 0)
        //        roc.calculateAUC(classIndex)

        // optionally, you can print all stats from the evaluations
        print(eval.stats())
        print(roc.stats())
    }


}

fun main() {
    val example = MNISTExample()
    example.run()
}