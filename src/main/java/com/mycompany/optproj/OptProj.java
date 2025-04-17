
package com.mycompany.optproj;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Arrays;


import java.util.Random;
import java.util.Scanner;

public class OptProj {

      public static int countFeatures(String filePath) {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            br.readLine(); // Skip the header row

            String firstDataRow = br.readLine(); // Read first actual data row
            if (firstDataRow != null) {
                String[] columns = firstDataRow.split(","); // Split by comma
                return columns.length - 1; // Exclude label column
            } else {
                System.out.println("CSV file is empty after header.");
                return -1; // Indicating an error
            }
        } catch (IOException e) {
            System.err.println("Error reading the file: " + e.getMessage());
            return -1; // Indicating an error
        }
    }
      
      
      public static double[] loadWeights(String filePath) {
        double[] buffer = null;

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String header = br.readLine(); // Read the header (Weight1, Weight2, ..., Bias)
            String valueRow = br.readLine(); // Read the row with actual weights and bias

            if (header != null && valueRow != null) {
                String[] values = valueRow.split(","); // Extract weights & bias
                buffer = new double[values.length]; // Allocate array dynamically

                // Convert each value to a double
                for (int i = 0; i < values.length; i++) {
                    buffer[i] = Double.parseDouble(values[i]);
                }
            }
        } catch (IOException e) {
            System.err.println("Error reading weights file: " + e.getMessage());
        }

        // **Print Loaded Weights & Bias for Verification**
        System.out.println("\nLoaded Weights & Bias:");
        System.out.println(Arrays.toString(buffer)); // Display all values
        
          
        return buffer;
    }






    public static void main(String[] args) throws IOException {
        
        
        Scanner scanner = new Scanner(System.in);
        int answer;
        int answer1;
        String filePath = "synthetic_data.csv"; // CSV file location
        int numFeatures = countFeatures(filePath);
        
            System.out.println("Number of Features: " + numFeatures);
        

        
        System.out.println("Generate new data?(1 for yes 0 for no)");
                answer = scanner.nextInt();
                if(answer != 1 && answer != 0){
               System.out.println("Invalid input exiting..");
               return;
               
           }
                
        if(answer == 1){
             
            int numSamples;
            System.out.println("Please enter number of samples and features");
            numSamples = scanner.nextInt();
            numFeatures = scanner.nextInt();
            
        Random rand = new Random();
        DecimalFormat df = new DecimalFormat("#.##");
        FileWriter writer = new FileWriter("synthetic_data.csv");

        
        writer.append("features and labels:\n");

        
        for (int i = 0; i < numSamples; i++) {
            StringBuilder row = new StringBuilder();
            double sumFeatures = 0;

            for (int j = 0; j < numFeatures; j++) {
                double feature = -1 + rand.nextDouble() * 2; // Generate feature value between 0 and 1
                sumFeatures += feature;
                row.append(df.format(feature)).append(",");

            }
            int label = (Math.sin(sumFeatures) + Math.cos(sumFeatures) > 0.3) ? 1 : 0;
            row.append(label);
            writer.append(row).append("\n");

        }

        writer.flush();
        writer.close();
        System.out.println("Synthetic data with "+ numFeatures +" features generated successfully!");
    }
        
       
        
        
                
        
        int epochs;
        double[] weights = new double[numFeatures];
        double bias = 0;
        double lr;
        
        
        
        
        
        
        
        System.out.println("Test or train (1 to test, 0 to train)");
           answer = scanner.nextInt();
           answer1 = answer;
           if(answer != 1 && answer != 0){
               System.out.println("Invalid input exiting..");
               return;
               
           }
        if(answer == 1){
            System.out.println("Test with last outputed weights?(1 for yes 0 for no)");
            answer = scanner.nextInt();
            
            if(answer != 1 && answer != 0){
               System.out.println("Invalid input exiting..");
               return;
               
           }
            if(answer == 1){
            
            
                    
         double[] buffer = loadWeights("final_weights.csv");

        for(int i = 0;i < buffer.length;i++){
              
              
              if(i == buffer.length -1){
                  bias = buffer[i];
              }
              else{
                  weights[i] = buffer[i];
              }
          }
            }
            else{
            System.out.println("Please enter weights("+numFeatures+")");
            for(int i = 0;i<numFeatures;i++){
                weights[i] = scanner.nextDouble();
            }
            }
        
        
        
        }
        
        else{
            
            
        for(int i = 0;i<numFeatures ;i++){
            
            Random rand = new Random();
        weights = new double[numFeatures];

        for (i = 0; i < numFeatures; i++) {
            weights[i] = rand.nextGaussian() * 0.01; // Small random normal values
        }


            
        }
        
        bias = 0;
        
        
        }
        
        
        
        
        
        for(int i = 0;i<numFeatures ;i++){
            
            System.out.println(weights[i]);
            
        }
        LinearReg reg = new LinearReg(weights,bias);
        
       
        reg.linearRegression();
        reg.splitDataset(0.8);
        if(answer1 == 1){
        reg.testModel();
        }
        
        else{
            System.out.println("Please enter number of epochs and learning rate");
        epochs = scanner.nextInt();
        lr = scanner.nextDouble();
        System.out.println("Please choose the optimiser (GD/1) (SGD/2) (MBGD/3) (ALL/4)");
        
        int answer3 = scanner.nextInt();
        if(answer3 != 1 && answer3 != 2 && answer3 != 3 && answer3 != 4){
               System.out.println("Invalid input exiting..");
               return;
               
           }
        if(answer3 == 2){
            reg.train(epochs, lr);
            reg.testModel();
        }
        if(answer3 == 1){
            reg.trainBatch(epochs, lr);
            reg.testModel();
        }
        if(answer3 == 3){
            System.out.println("Enter Batch size");
            int batchS = scanner.nextInt();
            reg.trainMiniBatch(epochs, lr,batchS);
            reg.testModel();
        }
        if(answer3 == 4){
             System.out.println("Enter Batch size");
            int batchS = scanner.nextInt();
            
            reg.train(epochs, lr);
            double SGD = reg.testModel();
            reg.trainBatch(epochs, lr);
            double GD = reg.testModel();
            reg.trainMiniBatch(epochs, lr,batchS);
            double MBGD = reg.testModel();
            
            if(SGD>GD){
                if(MBGD>SGD){
                    System.out.println("MBGD is most accurate");
                }
                else{
                    System.out.println("SGD is most accurate");
                }
            }
            else{
            if(GD>MBGD){
                System.out.println("GD is most accurate");
                
            }
            
            
            
            }
           
            
        }
        
        }
        
        
    }
    
    
    
   
    

}
