package com.deepnetts.examples.classification.spam;

import deepnetts.core.DeepNetts;
import deepnetts.data.DataSets;
import deepnetts.eval.Evaluators;
import javax.visrec.ml.eval.EvaluationMetrics;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.util.DeepNettsException;
import java.io.IOException;
import javax.visrec.ml.data.DataSet;
import deepnetts.data.TrainTestPair;
import deepnetts.data.norm.MaxScaler;

/**
 * Email spam  Classification example.
 * This example shows how to create a binary classifier for spam classification, using Feed Forward neural network.
 * Data is given in CSV file.
 *
 * For the best performance and accuracy the highly recommended way to run this example is to use Deep Netts Pro.
 * https://www.deepnetts.com/download
 *
 * Step-by-step guide for setting up Deep Netts is available at
 * https://www.deepnetts.com/quickstart
 * 
 * @see FeedForwardNetwork
 */
public class BuildAndTrainModel {

    public static void main(String[] args) throws DeepNettsException, IOException {

        // number of inputs and outputs in data set 
        int numInputs = 57;
        int numOutputs = 1;
        
        // load data from csv file
        DataSet dataSet = DataSets.readCsv("email_spam.csv", numInputs, numOutputs, true);             

        // randomly split data set into train and test set: 70% for training and rest 30% for testing
        TrainTestPair trainTest = DataSets.trainTestSplit(dataSet, 0.7);
        
        // scale training and test data (prepare for training)
        MaxScaler scaler = new MaxScaler(trainTest.getTrainingeSet()); // initialize scaler using training data
        scaler.apply(trainTest.getTrainingeSet()); // scale training data
        scaler.apply(trainTest.getTestSet()); // apply same scaler on test data
        
        // create an instance of the feed forward neural network using builder
        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(numInputs)
                .addFullyConnectedLayer(25, ActivationType.RELU)
                .addOutputLayer(numOutputs, ActivationType.SIGMOID)
                .lossFunction(LossType.CROSS_ENTROPY)
                .randomSeed(123)    // can be any number, usefull for repeating the training
                .build();

        // set training settings
        neuralNet.getTrainer().setStopError(0.3f)       // stop training when this error is reached
                              .setLearningRate(0.001f); // adjust step size for internal parameters change
        
        // start training
        neuralNet.train(trainTest.getTrainingeSet());
        
        // test trained network /  evaluate classifier on data that it has not seen during the training
        EvaluationMetrics em = neuralNet.test(trainTest.getTestSet());        
        System.out.println(em); // print evaluation results
        
        // Save trained network into a file for later use
        neuralNet.save("SpamClassifier.dnet");
        
        // shutdown all threads
        DeepNetts.shutdown();        
    }
    

}