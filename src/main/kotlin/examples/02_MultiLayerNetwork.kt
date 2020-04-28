package examples

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
 * https://deeplearning4j.konduit.ai/getting-started/tutorials/multilayernetwork-and-computationgraph
 *
 * DL4J provides the following classes to configure networks:
 *
 * 1. MultiLayerNetwork
 * 2. ComputationGraph
 *
 * MultiLayerNetwork consists of a single input layer and a single output layer with a stack of layers in between them.
 * ComputationGraph is used for constructing networks with a more complex architecture than MultiLayerNetwork. It can have multiple input layers, multiple output layers and the layers in between can be connected through a direct acyclic graph.
 */
class MultiLayerNetworkExample() : Example {

    override fun run() {
        logTime {
            val multiLayerNetwork: MultiLayerNetwork = buildMultiLayerNetwork()
        }

        logTime {
            val buildComputationGraph = buildComputationGraph()
        }
    }

    private fun buildComputationGraph(): ComputationGraphConfiguration {
        val computationGraphConf: ComputationGraphConfiguration = NeuralNetConfiguration.Builder()
            .seed(123)
            .updater(Nesterovs(0.1, 0.9)) //High Level Configuration
            .graphBuilder()  //For configuring ComputationGraph we call the graphBuilder method
            .addInputs("input") //Configuring Layers
            .addLayer(
                "L1",
                DenseLayer.Builder().nIn(3).nOut(4).build(), "input"
            )
            .addLayer(
                "out1",
                OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nIn(4).nOut(3)
                    .build(),
                "L1"
            )
            .addLayer(
                "out2",
                OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(4).nOut(2).build(),
                "L1"
            )
            .setOutputs("out1", "out2")
            .build() //Building configuration

        println("Sanity checking for our ComputationGraphConfiguration")
        println(computationGraphConf.toJson())

        val computationGraph: ComputationGraph = ComputationGraph(computationGraphConf)
        return computationGraphConf
    }

    private fun buildMultiLayerNetwork(): MultiLayerNetwork {
        val multiLayerConf: MultiLayerConfiguration = NeuralNetConfiguration.Builder()
            // For keeping the network outputs reproducible during runs by initializing weights and other network randomizations through a seed
            .seed(123)
            .updater(Nesterovs(0.1, 0.9)) //High Level Configuration
            .list() //For configuring MultiLayerNetwork we call the list method
            .layer(
                0, DenseLayer.Builder()
                    .nIn(784)
                    .nOut(100)
                    .weightInit(WeightInit.XAVIER)
                    .activation(Activation.RELU).build()
            ) //Configuring Layers
            .layer(
                1,
                OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                    .nIn(100)
                    .nOut(10).weightInit(WeightInit.XAVIER)
                    .activation(Activation.SIGMOID).build()
            )
            .build() //Building Configuration

        println("Sanity checking for our MultiLayerConfiguration")
        println(multiLayerConf.toJson())

        return MultiLayerNetwork(multiLayerConf)
    }
}


fun main() {
    val example = MultiLayerNetworkExample()
    example.run()
}