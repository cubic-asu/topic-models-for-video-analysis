package edu.asu.cubic.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.Properties;

import edu.asu.cubic.dimensionality_reduction.GMMVB;

/**
 * This class helps in extracting features using a Variational Bayes based GMM model 
 * @author prasanthl
 *
 */
public class GMMVBBuilder {

	public static void extractGMMFeatures(Properties requiredParameters) {
		try{
			DecimalFormat fmt= new DecimalFormat("#.####");
			String baseFolder= requiredParameters.getProperty("baseFolder").trim();
			String[] tokens= requiredParameters.getProperty("baseFeature").trim().split(";");
			String[] baseFeature= new String[tokens.length];
			String[] featureType= new String[tokens.length];
			for(int i=0; i<tokens.length; i++){
				baseFeature[i]= tokens[i].split(",")[0];
				featureType[i]= tokens[i].split(",")[1];
			}
			tokens= requiredParameters.getProperty("dimensionReduction").trim().split("_");
			int numTopics= Integer.parseInt(tokens[1]);
			double alpha= Double.parseDouble(tokens[2]);
			int emIters= Integer.parseInt(tokens[3]);
			double emConverged= 1e-4;//Double.parseDouble(tokens[3]);
			int V = Integer.parseInt(requiredParameters.getProperty("vocabSiz").trim());
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
			String modelFolderPath= baseFolder+"/TopicModels";
			String baseModelName= baseFeature[0]+featureType[0]+"GMMVB"+"_"+numTopics+"_"+alpha+"_"+emIters;
			if(!new File(modelFolderPath+"/"+baseModelName+".model").exists()){
				int[][] docsPerVidPerSet= new int[trainingSets.length][];
				int totalDocs=0;
				// load the total number of docs
				for(int set=0; set< trainingSets.length; set++){
					docsPerVidPerSet[set]= new int[trainingSetVideos[set][1]-trainingSetVideos[set][0]+1];
					String fileName= baseFolder+"/"+baseFeature[0]+featureType[0]+"/"+Utilities.capitalizeFirstLetter(trainingSets[set])+"Frames.txt";
					BufferedReader fileReader= new BufferedReader(new FileReader(fileName));
					String newline= fileReader.readLine();
					int vid=1, count=0;
					while(newline!=null){
						if(vid >= trainingSetVideos[set][0] && vid <= trainingSetVideos[set][1]){
							int numDocs= Integer.parseInt(newline);
							docsPerVidPerSet[set][count]= numDocs;
							totalDocs+= numDocs;
							count++;
						}
						newline= fileReader.readLine();
						vid++;
					}
					fileReader.close();
				}
				// load training documents
				double[][] trainingDocuments= new double[totalDocs][];
				String[] docIds= new String[totalDocs];
				int currDocs=0;
				for(int set=0; set< trainingSets.length; set++){
					for(int vid=trainingSetVideos[set][0]; vid<=trainingSetVideos[set][1]; vid++){
						String trainFeaturesFileName=  baseFolder+"/"+baseFeature[0]+featureType[0]+"/"+Utilities.capitalizeFirstLetter(trainingSets[set])+String.format("Seq%03d.csv", vid);
						String[][] tempFeatures= Utilities.readCSVFile(trainFeaturesFileName, true); 
						for(int d=currDocs; d<currDocs+tempFeatures.length; d++){
							trainingDocuments[d]= new double[tempFeatures[d-currDocs].length-1];
							docIds[d]= tempFeatures[d-currDocs][0];
							for(int n=1; n<tempFeatures[d-currDocs].length; n++){
								trainingDocuments[d][n-1]= Double.parseDouble(tempFeatures[d-currDocs][n]);
							}
						}
						currDocs+= tempFeatures.length;
					}
				}
				double[] alphas = new double[numTopics];
				double[][] betas = new double[numTopics][V];
				for(int i= 0; i<numTopics; i++)
					alphas[i]=alpha;
				for(int i= 0; i<numTopics; i++)
					for(int j=0; j<V; j++)
						betas[i][j]=0.02;
				System.out.println("Training GMM VB model");
				GMMVB trainingModel= new GMMVB(trainingDocuments, numTopics, alpha,modelFolderPath, baseModelName,emConverged,emIters);
				trainingModel.runEM();
				double[][] topicDistributions= trainingModel.getR();
				//Utilities.printArray(topicDistributions);
				// generate topic distributions for training docs
				currDocs=0;
				for(int set=0; set< trainingSets.length; set++){
					for(int vid=trainingSetVideos[set][0]; vid<=trainingSetVideos[set][1]; vid++){
						String trainFeaturesFolder= baseFolder+"/"+baseFeature[0]+featureType[0]+"GMMVB"+"_"+numTopics+"_"+alpha+"_"+emIters;
						if(!new File(trainFeaturesFolder).exists())
							new File(trainFeaturesFolder).mkdir();
						String trainCSVFilePath= trainFeaturesFolder+"/"+Utilities.capitalizeFirstLetter(trainingSets[set])+"Seq"+String.format("%03d",vid)+".csv";
						if(!new File(trainCSVFilePath).exists()){
							System.out.println("Train Video: "+vid);
							PrintWriter featuresFile= new PrintWriter(new File(trainCSVFilePath));
							featuresFile.print("DocId,");
							for(int i=1;i<=numTopics; i++){
								featuresFile.print("Feature"+i);
								if(i!=numTopics)
									featuresFile.print(",");
							}
							featuresFile.println();
							for(int d=currDocs; d< currDocs+docsPerVidPerSet[set][vid-trainingSetVideos[set][0]]; d++){
								featuresFile.print(docIds[d]+",");
								for(int i=0;i<numTopics; i++){
									featuresFile.print(fmt.format(topicDistributions[d][i]));
									if(i!=numTopics-1)
										featuresFile.print(",");
								}
								featuresFile.println();
							}
							featuresFile.close();
						}
						currDocs+=docsPerVidPerSet[set][vid-trainingSetVideos[set][0]];
					}
				}
				// clean the model by removing unnecessary variables
				trainingModel.cleanUpVariables();
				ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelFolderPath+"/"+baseModelName+".model"));
				oos.writeObject(trainingModel);
				oos.close();
			}
			if(!requiredParameters.getProperty("testingSets").trim().equals("")){
				tokens= requiredParameters.getProperty("testingSets").trim().split(";");
				String[] testingSets= new String[tokens.length];
				int[][] testingSetVideos= new int[tokens.length][2];
				for(int i=0; i<tokens.length; i++)
				{
					testingSets[i]= tokens[i].split(",")[0];
					testingSetVideos[i][0]= Integer.parseInt(tokens[i].split(",")[1]);// start video
					testingSetVideos[i][1]= Integer.parseInt(tokens[i].split(",")[2]);// end video
				}
				// if the LDA model exists then generate the testing LDA files 
				GMMVB trainingModel= (GMMVB)new ObjectInputStream(new FileInputStream(modelFolderPath+"/"+baseModelName+".model")).readObject();
				GMMVB testingModel= null;
				for(int set=0; set< testingSets.length; set++){
					for(int vid=testingSetVideos[set][0]; vid<=testingSetVideos[set][1]; vid++){
						String testFeaturesFolder= baseFolder+"/"+baseFeature[0]+featureType[0]+"GMMVB"+"_"+numTopics+"_"+alpha+"_"+emIters;
						if(!new File(testFeaturesFolder).exists())
							new File(testFeaturesFolder).mkdir();
						String testCSVFilePath= testFeaturesFolder+"/"+Utilities.capitalizeFirstLetter(testingSets[set])+"Seq"+String.format("%03d",vid)+".csv";
						if(!new File(testCSVFilePath).exists()){
							System.out.println("Test Video: "+vid);
							String testFeaturesFileName=  baseFolder+"/"+baseFeature[0]+featureType[0]+"/"+Utilities.capitalizeFirstLetter(testingSets[set])+String.format("Seq%03d.csv", vid);
							//int[][] testDocuments=Utilities.loadDocuments(testFeaturesFileName);
							String[][] tempArray= Utilities.readCSVFile(testFeaturesFileName, true);
							double[][] testDocuments= new double[tempArray.length][];
							String[] docIds= new String[tempArray.length];
							for(int d=0; d<tempArray.length; d++){
								docIds[d]= tempArray[d][0];
								testDocuments[d]= new double[tempArray[d].length-1];
								for(int n=1; n<tempArray[d].length; n++){
									testDocuments[d][n-1]= Double.parseDouble(tempArray[d][n]);
								}
							}
							testingModel= new GMMVB(testDocuments,trainingModel, modelFolderPath, baseModelName);
							testingModel.infer();
							double[][] topicDistributions= testingModel.getR();
							//Utilities.printArray(topicDistributions);
							PrintWriter featuresFile= new PrintWriter(new File(testCSVFilePath));
							featuresFile.print("DocId,");
							for(int i=1;i<=numTopics; i++){
								featuresFile.print("Feature"+i);
								if(i!=numTopics)
									featuresFile.print(",");
							}
							featuresFile.println();
							for(int d=0; d< topicDistributions.length; d++){
								featuresFile.print(docIds[d]+",");
								for(int t=0; t< numTopics; t++){
									featuresFile.print(fmt.format(topicDistributions[d][t]));
									if(t!=numTopics-1){
										featuresFile.print(",");
									}
								}
								featuresFile.println();
							}
							featuresFile.close();
						}
					}
				}
			}
		}
		catch(Exception e){e.printStackTrace();System.exit(1);}
	}


}
