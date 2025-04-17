package com.mycompany.optproj;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

public class LinearReg {
    List<double[]> X = new ArrayList<>();
        private List<Integer> y = new ArrayList<>();
        private List<double[]> X_train = new ArrayList<>();
        private List<Integer> y_train = new ArrayList<>();
        private List<double[]> X_test = new ArrayList<>();
        private List<Integer> y_test = new ArrayList<>();
        
        private double[] weights; // Model weights
        private double bias; // Bias term

    public LinearReg(double[] weights,double bias) {
        this.weights = weights;
        this.bias = bias;
    }


    
    public void linearRegression() throws FileNotFoundException, IOException{
        

        BufferedReader reader = new BufferedReader(new FileReader("synthetic_data.csv"));
        String line;
        reader.readLine(); // Skip header
    try {
        while ((line = reader.readLine()) != null) {
            String[] values = line.split(",");
            double[] features = new double[values.length - 1];
            for (int i = 0; i < features.length; i++) {
                features[i] = Double.parseDouble(values[i]);
            }
            X.add(features);
            y.add(Integer.parseInt(values[values.length - 1])); // Last column is label
        }
    } catch (IOException ex) {
        Logger.getLogger(LinearReg.class.getName()).log(Level.SEVERE, null, ex);
    }
        reader.close();

    }
    
    
    
    
    
    
    
    
    public void splitDataset(double trainRatio) {
        int trainSize = (int) (trainRatio * X.size());
        

        // Shuffle indices to randomly split data
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < X.size(); i++) {
            indices.add(i);
        }
        Collections.shuffle(indices);

        // Split data into training and testing sets
        for (int i = 0; i < X.size(); i++) {
            if (i < trainSize) {
                X_train.add(X.get(indices.get(i)));
                y_train.add(y.get(indices.get(i)));
            } else {
                X_test.add(X.get(indices.get(i)));
                y_test.add(y.get(indices.get(i)));
            }
        }

        // Print sizes for verification
        System.out.println("Training Set Size: " + X_train.size());
        System.out.println("Testing Set Size: " + X_test.size());
    }

    
    
    
    
    
    
    public double sigmoid(double z) {
    return 1.0 / (1.0 + Math.exp(-z)); // Apply sigmoid
}

    
public void displayConfusionMatrix() {
    int TP = 0, FP = 0, TN = 0, FN = 0;

    for (int i = 0; i < X_test.size(); i++) {
        double[] features = X_test.get(i);
        int actualLabel = y_test.get(i);

        // Compute z = w^T x + b
        double z = bias;
        for (int j = 0; j < features.length; j++) {
            z += weights[j] * features[j];
        }

        // Apply sigmoid function and classify
        double probability = sigmoid(z);
        int predictedLabel = (probability >= 0.5) ? 1 : 0;

        // Update confusion matrix counts
        if (predictedLabel == 1 && actualLabel == 1) TP++; // True Positive
        else if (predictedLabel == 1 && actualLabel == 0) FP++; // False Positive
        else if (predictedLabel == 0 && actualLabel == 0) TN++; // True Negative
        else FN++; // False Negative
    }

    // Print Confusion Matrix
    System.out.println("\nConfusion Matrix:");
    System.out.println("        Predicted 0  Predicted 1");
    System.out.println("Actual 0    " + TN + "           " + FP);
    System.out.println("Actual 1    " + FN + "           " + TP);
}
    
    
    
    
    
public void train(int epochs, double learningRate) {
        int m = X_train.size(); // Number of samples

        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;

            for (int i = 0; i < m; i++) {
                double[] features = X_train.get(i);
                int actualLabel = y_train.get(i);

                // Compute z = w^T x + b
                double z = bias;
                for (int j = 0; j < features.length; j++) {
                    z += weights[j] * features[j];
                }

                // Apply sigmoid function
                double prediction = sigmoid(z);
                double error = actualLabel - prediction;

                // Compute loss (Squared Error for simplicity)
                totalLoss += error * error;

                // Update weights & bias using Gradient Descent
                bias += learningRate * error;
                for (int j = 0; j < features.length; j++) {
                    weights[j] += learningRate * error * features[j];
                }
            }

            System.out.println("Epoch " + (epoch + 1) + " - Loss: " + (totalLoss / m));
        }
 
        System.out.println("\nFinal Weights:");
    for (int i = 0; i < weights.length; i++) {
        System.out.println("Weight " + (i + 1) + ": " + weights[i]);
    }
    System.out.println("Final Bias: " + bias);
    
    saveWeightsToCSV();

    }




public void trainBatch(int epochs, double learningRate) {
    int m = X_train.size(); // Total samples
    int numFeatures = X_train.get(0).length; // Number of features

    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalLoss = 0;
        double biasGradient = 0;
        double[] weightGradients = new double[numFeatures]; // Stores accumulated gradients

        // **Step 1: Accumulate Gradients for All Samples**
        for (int i = 0; i < m; i++) {
            double[] features = X_train.get(i);
            int actualLabel = y_train.get(i);

            // Compute z = w^T x + b
            double z = bias;
            for (int j = 0; j < features.length; j++) {
                z += weights[j] * features[j];
            }

            // Apply sigmoid function
            double prediction = sigmoid(z);
            double error = actualLabel - prediction;

            // **Accumulate loss and gradients**
            totalLoss += error * error; 
            biasGradient += error;
            for (int j = 0; j < numFeatures; j++) {
                weightGradients[j] += error * features[j]; // Sum all gradients for features
            }
        }

        // **Step 2: Apply Averaged Gradient Update**
        bias += learningRate * (biasGradient / m);
        for (int j = 0; j < numFeatures; j++) {
            weights[j] += learningRate * (weightGradients[j] / m); // Average gradients
        }

        System.out.println("Epoch " + (epoch + 1) + " - Loss: " + (totalLoss / m));
    }

    // **Save the updated weights**
    saveWeightsToCSV();

    System.out.println("\nFinal Weights:");
    for (int i = 0; i < weights.length; i++) {
        System.out.println("Weight " + (i + 1) + ": " + weights[i]);
    }
    System.out.println("Final Bias: " + bias);
}




public void trainMiniBatch(int epochs, double learningRate, int batchSize) {
    int m = X_train.size(); // Total samples
    int numFeatures = X_train.get(0).length; // Feature count

    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalLoss = 0;

        // **Shuffle dataset to ensure batches vary each epoch**
        Collections.shuffle(X_train);
        Collections.shuffle(y_train);

        for (int i = 0; i < m; i += batchSize) {
            double biasGradient = 0;
            double[] weightGradients = new double[numFeatures];
            int currentBatchSize = Math.min(batchSize, m - i); // Handle last batch size

            // **Compute gradients for mini-batch**
            for (int j = i; j < i + currentBatchSize; j++) {
                double[] features = X_train.get(j);
                int actualLabel = y_train.get(j);

                // Compute z = w^T x + b
                double z = bias;
                for (int k = 0; k < features.length; k++) {
                    z += weights[k] * features[k];
                }

                // Apply sigmoid function
                double prediction = sigmoid(z);
                double error = actualLabel - prediction;

                // Accumulate gradients
                totalLoss += error * error;
                biasGradient += error;
                for (int k = 0; k < numFeatures; k++) {
                    weightGradients[k] += error * features[k];
                }
            }

            // **Apply mini-batch averaged gradient update**
            bias += learningRate * (biasGradient / currentBatchSize);
            for (int k = 0; k < numFeatures; k++) {
                weights[k] += learningRate * (weightGradients[k] / currentBatchSize);
            }
        }

        System.out.println("Epoch " + (epoch + 1) + " - Loss: " + (totalLoss / m));
    }

    // Save final weights
    saveWeightsToCSV();

    System.out.println("\nFinal Weights:");
    for (int i = 0; i < weights.length; i++) {
        System.out.println("Weight " + (i + 1) + ": " + weights[i]);
    }
    System.out.println("Final Bias: " + bias);
}



public double testModel() {
    int totalSamples = X_test.size();
    int correctPredictions = 0;
    int TP = 0, FP = 0, TN = 0, FN = 0;

    for (int i = 0; i < totalSamples; i++) {
        double[] features = X_test.get(i);
        int actualLabel = y_test.get(i);

        // Compute z = w^T x + b
        double z = bias;
        for (int j = 0; j < features.length; j++) {
            z += weights[j] * features[j];
        }

        // Apply sigmoid function
        double probability = sigmoid(z);
        int predictedLabel = (probability >= 0.5) ? 1 : 0; // Classification decision

        // Compare prediction with actual label
        if (predictedLabel == actualLabel) {
            correctPredictions++;
            if (predictedLabel == 1) TP++; // True Positive
            else TN++; // True Negative
        } else {
            if (predictedLabel == 1) FP++; // False Positive
            else FN++; // False Negative
        }
    }

    // Calculate metrics
    double accuracy = (double) correctPredictions / totalSamples;
    double precision = (TP + FP) > 0 ? (double) TP / (TP + FP) : 0;
    double recall = (TP + FN) > 0 ? (double) TP / (TP + FN) : 0;
    double f1Score = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;

    // Print results
    System.out.println("\nModel Evaluation Metrics:");
    System.out.println("Accuracy: " + accuracy);
    System.out.println("Precision: " + precision);
    System.out.println("Recall: " + recall);
    System.out.println("F1 Score: " + f1Score);
    
    
    displayConfusionMatrix();
    return accuracy;
}




public void saveWeightsToCSV() {
    String fileName = "final_weights.csv"; // CSV file name

    try (PrintWriter pw = new PrintWriter(new FileWriter(fileName))) {
        // **Write header dynamically based on number of features**
        pw.print("Weight1");
        for (int i = 1; i < weights.length; i++) {
            pw.print(",Weight" + (i + 1));
        }
        pw.print(",Bias"); // Bias as the last column
        pw.println(); // New line after header

        // **Write weights and bias in a single row**
        pw.print(weights[0]);
        for (int i = 1; i < weights.length; i++) {
            pw.print("," + weights[i]);
        }
        pw.print("," + bias); // Append bias
        pw.println(); // New line for separation

        System.out.println("\nFinal weights saved to: " + fileName);
    } catch (IOException e) {
        System.err.println("Error saving weights: " + e.getMessage());
    }
}

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }


    
    
    
    
    
    
    
    
    
    
    
}
