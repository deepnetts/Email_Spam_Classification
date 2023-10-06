package com.deepnetts.examples.classification.spam;

import deepnetts.core.DeepNetts;
import deepnetts.data.DataSets;
import deepnetts.data.MLDataItem;
import deepnetts.data.TabularDataSet;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.util.FileIO;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.visrec.ml.classification.BinaryClassifier;
import javax.visrec.ml.data.DataSet;
import javax.visrec.ri.ml.classification.FeedForwardNetBinaryClassifier;
import javax.visrec.ri.ml.classification.ImageClassifierNetwork;

/**
 * This example shows how to load and create instance of trained network from file.
 * 
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class UseTrainedModel {
    
    public static void main(String[] args) {  
        try {
            // load trained convolutioal network from file
            FeedForwardNetwork neuralNet =  FileIO.createFromFile("SpamClassifier.dnet", FeedForwardNetwork.class);

            // create classifier using loaded trained network
            BinaryClassifier<float[]> spamClassifier = new FeedForwardNetBinaryClassifier(neuralNet);
            
            // load data for testing purposes
            TabularDataSet<MLDataItem> dataSet = DataSets.readCsv("email_spam.csv", 57, 1, true);             
            
            // get a single feature vector/array for testing
            float[] testEmail = dataSet.get(0).getInput().getValues();
            
            // feed the classifer and get a result (spam probability)
            Float spamProbability = spamClassifier.classify(testEmail);
            System.out.println("Spam probability for the given email is: "+spamProbability);      

            // shutdown the thread pool
            DeepNetts.shutdown();
        } catch (IOException | ClassNotFoundException ioe) {
            Logger.getLogger(UseTrainedModel.class.getName()).log(Level.SEVERE, null, ioe);
        }
     
    }    
}
