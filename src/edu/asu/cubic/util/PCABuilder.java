package edu.asu.cubic.util;

import edu.asu.cubic.dimensionality_reduction.PCA;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.Properties;

import weka.classifiers.functions.SMOreg;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;;;

/**
 * Given a set of training files the class builds a PCA model 
 * @author prasanthl
 *
 */
public class PCABuilder {


	public static void extractPCAFeatures(Properties requiredParameters){
		try{
			DecimalFormat fmt= new DecimalFormat("#.####");
			// Load the properties file
			String baseFolder= requiredParameters.getProperty("baseFolder").trim();
			String[] tokens= requiredParameters.getProperty("baseFeature").trim().split(";");
			String[] baseFeature= new String[tokens.length];
			String[] featureType= new String[tokens.length];
			for(int i=0; i<tokens.length; i++){
				baseFeature[i]= tokens[i].split(",")[0];
				featureType[i]= tokens[i].split(",")[1];
			}
			// training files to be used can be passed as a combination of
			// trainingSet,startVideo,endVideo values
			tokens= requiredParameters.getProperty("trainingSets").trim().split(";");
			String[] trainingSets= new String[tokens.length];
			int[][] trainingSetVideos= new int[tokens.length][2];
			for(int i=0; i<tokens.length; i++)
			{
				trainingSets[i]= tokens[i].split(",")[0];
				trainingSetVideos[i][0]= Integer.parseInt(tokens[i].split(",")[1]);// start video
				trainingSetVideos[i][1]= Integer.parseInt(tokens[i].split(",")[2]);// end video
			}
			// if the PCA model does not exist then go ahead and generate training PCA files
			if(!new File(baseFolder+"/PCAModels/").exists()){
				new File(baseFolder+"/PCAModels/").mkdir();
			}
			String modelFilePath= baseFolder+"/PCAModels/"+baseFeature[0]+featureType[0]+".model";
			PCA pcaModel= null;
			if(!new File(modelFilePath).exists()){
				// pool all the training data
				double[][] trainingFeatures= null;
				Instances trainingInstances= null;
				for(int set=0; set< trainingSets.length; set++){
					for(int vid=trainingSetVideos[set][0]; vid<=trainingSetVideos[set][1]; vid++){
						String trainFeaturesFileName=  baseFolder+"/"+baseFeature[0]+featureType[0]+"/"+Utilities.capitalizeFirstLetter(trainingSets[set])+String.format("Seq%03d.csv", vid);
						//System.out.println(trainFeaturesFileName);
						// load training features as weka instances
						try{
							CSVLoader loader= new CSVLoader();
							loader.setFile(new File(trainFeaturesFileName));
							if(trainingInstances==null)
								trainingInstances= loader.getDataSet();
							else{
								Instances currVidInstances= loader.getDataSet();
								for(Instance inst: currVidInstances){
									trainingInstances.add(inst);
								}
							}
						}
						catch(Exception e){
							System.err.println("Error loading "+trainFeaturesFileName);
							e.printStackTrace();
							System.exit(1);
						}
					}
				}
				trainingInstances.deleteAttributeAt(0);
				// load the weka instances to the double array
				trainingFeatures= new double[trainingInstances.numInstances()][trainingInstances.numAttributes()];
				int count=0;
				for(Instance instance: trainingInstances){
					trainingFeatures[count]= instance.toDoubleArray();
					count++;
				}
				// generate the PCA model on training features
				System.out.println("Creating new PCA model");
				pcaModel= new PCA(trainingFeatures);
				// clean the model by removing unnecessary variables
				pcaModel.cleanUpVariables();
				ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelFilePath));
				oos.writeObject(pcaModel);
				oos.close();
			}
			else{
				pcaModel= (PCA)new ObjectInputStream(new FileInputStream(modelFilePath)).readObject();
			}
			double[][] trainingFeatures= null;
			Instances trainingInstances= null;
			for(int set=0; set< trainingSets.length; set++){
				for(int vid=trainingSetVideos[set][0]; vid<=trainingSetVideos[set][1]; vid++){
					String trainFeaturesFileName=  baseFolder+"/"+baseFeature[0]+featureType[0]+"/"+Utilities.capitalizeFirstLetter(trainingSets[set])+String.format("Seq%03d.csv", vid);
					System.out.println(trainFeaturesFileName);
					// load training features as weka instances
					CSVLoader loader= new CSVLoader();
					loader.setFile(new File(trainFeaturesFileName));
					if(trainingInstances==null)
						trainingInstances= loader.getDataSet();
					else{
						Instances currVidInstances= loader.getDataSet();
						for(Instance inst: currVidInstances){
							trainingInstances.add(inst);
						}
					}
				}
			}
			trainingInstances.deleteAttributeAt(0);
			// load the weka instances to the double array
			trainingFeatures= new double[trainingInstances.numInstances()][trainingInstances.numAttributes()];
			int count=0;
			for(Instance instance: trainingInstances){
				trainingFeatures[count]= instance.toDoubleArray();
				count++;
			}
			// generate training PCA files
			for(int set=0; set< trainingSets.length; set++){
				for(int vid=trainingSetVideos[set][0]; vid<=trainingSetVideos[set][1]; vid++){
					String trainFeaturesFolder= baseFolder+"/"+baseFeature[0]+featureType[0]+"PCA";
					if(!new File(trainFeaturesFolder).exists())
						new File(trainFeaturesFolder).mkdir();
					String trainCSVFilePath= trainFeaturesFolder+"/"+Utilities.capitalizeFirstLetter(trainingSets[set])+"Seq"+String.format("%03d",vid)+".csv";
					if(!new File(trainCSVFilePath).exists()){
						System.out.println("Train Video: "+vid);
						trainingFeatures= null;
						String[] docIds= null;
						trainingInstances= null;
						// load training features as weka instances
						String trainFeaturesFileName=  baseFolder+"/"+baseFeature[0]+featureType[0]+"/"+Utilities.capitalizeFirstLetter(trainingSets[set])+String.format("Seq%03d.csv", vid);
						CSVLoader loader= new CSVLoader();
						loader.setFile(new File(trainFeaturesFileName));
						trainingInstances= loader.getDataSet();
						count=0;
						docIds= new String[trainingInstances.numInstances()];
						for(Instance inst: trainingInstances){
							docIds[count]= new String(""+(int)inst.value(0));
							count++;
						}
						trainingInstances.deleteAttributeAt(0);
						trainingFeatures= new double[trainingInstances.numInstances()][];
						count=0;
						for(Instance inst: trainingInstances){
							trainingFeatures[count]= inst.toDoubleArray();
							count++;
						}
						// project the training features onto 98% eigen space
						double[][] projectedData= pcaModel.project(trainingFeatures,98);
						// write all the features and responses to a csv file
						PrintWriter trainingDataCSVFile= new PrintWriter(new File(trainCSVFilePath));
						trainingDataCSVFile.print("DocId,");
						for(int ind=0; ind<projectedData[0].length; ind++ )
							if(ind!=projectedData[0].length-1)
								trainingDataCSVFile.print("Feature"+(ind+1)+",");
							else
								trainingDataCSVFile.print("Feature"+(ind+1));
						trainingDataCSVFile.println();
						for(int m=0; m<projectedData.length; m++){
							trainingDataCSVFile.print(docIds[m]+",");
							for(int ind=0; ind<projectedData[0].length; ind++ )
								if(!Double.isNaN(projectedData[m][ind])){
									if(ind!=projectedData[0].length-1)
										trainingDataCSVFile.print(fmt.format(projectedData[m][ind])+",");
									else
										trainingDataCSVFile.print(fmt.format(projectedData[m][ind]));
								}
								else{
									if(ind!=projectedData[0].length-1)
										trainingDataCSVFile.print("NaN"+",");
									else
										trainingDataCSVFile.print("NaN");
								}
							trainingDataCSVFile.println();
						}
						trainingDataCSVFile.close();
					}
				}
			}
			// load the PCA model
			tokens= requiredParameters.getProperty("testingSets").trim().split(";");
			String[] testingSets= new String[tokens.length];
			int[][] testingSetVideos= new int[tokens.length][2];
			for(int i=0; i<tokens.length; i++)
			{
				testingSets[i]= tokens[i].split(",")[0];
				testingSetVideos[i][0]= Integer.parseInt(tokens[i].split(",")[1]);// start video
				testingSetVideos[i][1]= Integer.parseInt(tokens[i].split(",")[2]);// end video
			}
			for(int set=0; set< testingSets.length; set++){
				for(int vid=testingSetVideos[set][0]; vid<=testingSetVideos[set][1]; vid++){
					String testFeaturesFolder= baseFolder+"/"+baseFeature[0]+featureType[0]+"PCA";
					if(!new File(testFeaturesFolder).exists())
						new File(testFeaturesFolder).mkdir();
					String testCSVFilePath= testFeaturesFolder+"/"+Utilities.capitalizeFirstLetter(testingSets[set])+"Seq"+String.format("%03d",vid)+".csv";
					if(!new File(testCSVFilePath).exists()){
						System.out.println("Test Video: "+vid);
						double[][] testingFeatures= null;
						String[] docIds= null;
						Instances testingInstances= null;
						// load testing features as weka instances
						String testFeaturesFileName=  baseFolder+"/"+baseFeature[0]+featureType[0]+"/"+Utilities.capitalizeFirstLetter(testingSets[set])+String.format("Seq%03d.csv", vid);
						CSVLoader loader= new CSVLoader();
						loader.setFile(new File(testFeaturesFileName));
						testingInstances= loader.getDataSet();
						count=0;
						docIds= new String[testingInstances.numInstances()];
						for(Instance inst: testingInstances){
							docIds[count]= new String(""+(int)inst.value(0));
							count++;
						}
						testingInstances.deleteAttributeAt(0);
						testingFeatures= new double[testingInstances.numInstances()][];
						count=0;
						for(Instance inst: testingInstances){
							testingFeatures[count]= inst.toDoubleArray();
							count++;
						}
						// project the testing features onto 98% eigen space
						double[][] projectedData= pcaModel.project(testingFeatures,98);
						// write all the features and responses to a csv file
						PrintWriter testingDataCSVFile= new PrintWriter(new File(testCSVFilePath));
						testingDataCSVFile.print("DocId,");
						for(int ind=0; ind<projectedData[0].length; ind++ )
							if(ind!=projectedData[0].length-1)
								testingDataCSVFile.print("Feature"+(ind+1)+",");
							else
								testingDataCSVFile.print("Feature"+(ind+1));
						testingDataCSVFile.println();
						for(int m=0; m<projectedData.length; m++){
							testingDataCSVFile.print(docIds[m]+",");
							for(int ind=0; ind<projectedData[0].length; ind++ )
								try{
									if(!Double.isNaN(projectedData[m][ind])){
										if(ind!=projectedData[0].length-1)
											testingDataCSVFile.print(fmt.format(projectedData[m][ind])+",");
										else
											testingDataCSVFile.print(fmt.format(projectedData[m][ind]));
									}
									else{
										if(ind!=projectedData[0].length-1)
											testingDataCSVFile.print("NaN"+",");
										else
											testingDataCSVFile.print("NaN");
									}
								}
							catch(Exception e){
								System.out.println(projectedData[m][ind]);
								e.printStackTrace();
								throw e;
							}
							testingDataCSVFile.println();
						}
						testingDataCSVFile.close();
					}
				}
			}

		}		
		catch(Exception e){e.printStackTrace();System.exit(1);}
	}
}
