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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

import com.google.common.collect.Lists;

import edu.asu.cubic.dimensionality_reduction.LDAGibbs;
import edu.asu.cubic.dimensionality_reduction.LDAVB;

/**
 * This class helps in reducing dimensions using LDA with Variational Bayes inference 
 * @author prasanthl
 *
 */
public class LDAVBBuilder {

	public static void extractLDAFeatures(Properties requiredParameters) {
		try{
			DecimalFormat fmt= new DecimalFormat("#.####");
			// Load the properties file
			String baseFolder= requiredParameters.getProperty("baseFolder").trim();
			String[] tokens= requiredParameters.getProperty("baseFeature").trim().split(";");
			String[] baseFeature= new String[tokens.length];
			String[] featureType= new String[tokens.length];
			for(int i=0; i<tokens.length; i++){
				baseFeature[i]= tokens[i].split(",")[0];
				if(tokens[i].split(",").length >= 2)
					featureType[i]= tokens[i].split(",")[1];
				else
					featureType[i]= "";
			}
			tokens= requiredParameters.getProperty("dimensionReduction").trim().split("_");
			int numTopics= Integer.parseInt(tokens[1]);
			int emIters= Integer.parseInt(tokens[2]);
			double alpha= Double.parseDouble(tokens[3]);
			double beta= Double.parseDouble(tokens[4]);
			double emConverged= 1e-4;//Double.parseDouble(tokens[3]);
			int varIters= -1;//Integer.parseInt(tokens[4]);
			double varConverged= 1e-6;//Double.parseDouble(tokens[5]);
			boolean deleteModelFiles = Boolean.parseBoolean(requiredParameters.getProperty("deleteModelFiles").trim());
			int V = Integer.parseInt(requiredParameters.getProperty("vocabSiz").trim());
			// training files to be used can be passed as a combination of
			// trainingSet,startVideo,endVideo values
			tokens= requiredParameters.getProperty("trainingSets").trim().split(";");
			String[] trainingSets= new String[tokens.length];
			int[][] trainingSetVideos= new int[tokens.length][2];
			for(int i=0; i<tokens.length; i++)
			{
				System.out.println(tokens[i]);
				trainingSets[i]= tokens[i].split(",")[0];
				trainingSetVideos[i][0]= Integer.parseInt(tokens[i].split(",")[1]);// start video
				trainingSetVideos[i][1]= Integer.parseInt(tokens[i].split(",")[2]);// end video
			}
			baseFolder = baseFolder+"/"+baseFeature[0]+featureType[0];
			String modelFolderPath= baseFolder;
			String baseModelName= "LDAVB"+"_"+numTopics+"_"+emIters+"_"+alpha+"_"+beta;
			if(!new File(modelFolderPath+"/"+baseModelName+".model").exists()){
				int[][] docsPerVidPerSet= new int[trainingSets.length][];
				int totalDocs=0;
				// load the total number of docs
				/*for(int set=0; set< trainingSets.length; set++){
					docsPerVidPerSet[set]= new int[trainingSetVideos[set][1]-trainingSetVideos[set][0]+1];
					String fileName= baseFolder+"/"+baseFeature[0]+Utilities.capitalizeFirstLetter(trainingSets[set])+"Frames.txt";
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
				}*/
				// load training documents
				//System.out.println("Total Docs: "+totalDocs);
				List<List<Integer>> trainingDocumentsList = Lists.newArrayList();
				
				//String[] docIds= new String[totalDocs];
				List<String> docIds = Lists.newArrayList();
				int currDocs=0;
				for(int set=0; set< trainingSets.length; set++){
					docsPerVidPerSet[set]= new int[trainingSetVideos[set][1]-trainingSetVideos[set][0]+1];
					System.out.println("Loading "+Utilities.capitalizeFirstLetter(trainingSets[set]) +" "+trainingSetVideos[set][0]+ " "+trainingSetVideos[set][1]);
					int count = 0;
					for(int vid=trainingSetVideos[set][0]; vid<=trainingSetVideos[set][1]; vid++){
						String trainFeaturesFileName=  baseFolder+"/docs/"+Utilities.capitalizeFirstLetter(trainingSets[set])+String.format("Seq%03d.csv", vid);
						//System.out.println("Loading from: "+trainFeaturesFileName);
						int[][] tempDocuments=Utilities.loadDocuments(trainFeaturesFileName);
						String[][] tempDocIds= Utilities.readCSVFile(trainFeaturesFileName, false);
						
						for(int d=0; d<tempDocuments.length; d++){
							List<Integer> tempList = Lists.newArrayList();
							for(int w=0; w<tempDocuments[d].length; w++){
								tempList.add(tempDocuments[d][w]);
							}
							trainingDocumentsList.add(tempList);
							docIds.add(tempDocIds[d][0]);
						}
						currDocs+= tempDocuments.length;
						docsPerVidPerSet[set][count]= tempDocuments.length;
						count++;
					}
				}
				totalDocs = trainingDocumentsList.size();
				int[][] trainingDocuments= new int[totalDocs][];
				for(int d=0; d<totalDocs; d++){
					trainingDocuments[d] = new int[trainingDocumentsList.get(d).size()];
					for(int w=0; w<trainingDocumentsList.get(d).size(); w++){
						trainingDocuments[d][w] = trainingDocumentsList.get(d).get(w);
					}
				}
				double[] alphas = new double[numTopics];
				double[][] betas = new double[numTopics][V];
				for(int i= 0; i<numTopics; i++)
					alphas[i]=alpha;
				for(int i= 0; i<numTopics; i++)
					for(int j=0; j<V; j++)
						betas[i][j]=beta;
				System.out.println("Training LDA VB model");
				LDAVB trainingModel= new LDAVB(trainingDocuments, V,numTopics, alphas[0], beta,modelFolderPath, baseModelName,emIters,emConverged, varIters,varConverged,false);
				trainingModel.runParallelEM();
				double[][] topicDistributions= trainingModel.getGamma();
				// generate topic distributions for training docs
				currDocs=0;
				for(int set=0; set< trainingSets.length; set++){
					for(int vid=trainingSetVideos[set][0]; vid<=trainingSetVideos[set][1]; vid++){
						String trainFeaturesFolder= baseFolder+"/"+"LDAVB"+"_"+numTopics+"_"+emIters+"_"+alpha+"_"+beta;
						if(!new File(trainFeaturesFolder).exists())
							new File(trainFeaturesFolder).mkdir();
						String trainCSVFilePath= trainFeaturesFolder+"/"+Utilities.capitalizeFirstLetter(trainingSets[set])+"Seq"+String.format("%03d",vid)+".csv";
						if(vid==2){
							System.out.println();
						}
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
								featuresFile.print(docIds.get(d)+",");
								for(int i=0;i<numTopics; i++){
									if(!Double.isNaN(topicDistributions[d][i]))
										featuresFile.print(fmt.format(topicDistributions[d][i]));
									else
										featuresFile.print("NaN");
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
				LDAVB trainingModel= (LDAVB)new ObjectInputStream(new FileInputStream(modelFolderPath+"/"+baseModelName+".model")).readObject();
				LDAVB testingModel= null;
				for(int set=0; set< testingSets.length; set++){
					for(int vid=testingSetVideos[set][0]; vid<=testingSetVideos[set][1]; vid++){
						String testFeaturesFolder= baseFolder+"/"+"LDAVB"+"_"+numTopics+"_"+emIters+"_"+alpha+"_"+beta;
						if(!new File(testFeaturesFolder).exists())
							new File(testFeaturesFolder).mkdir();
						String testCSVFilePath= testFeaturesFolder+"/"+Utilities.capitalizeFirstLetter(testingSets[set])+"Seq"+String.format("%03d",vid)+".csv";
						if(!new File(testCSVFilePath).exists()){
						System.out.println("Test Video: "+vid);
						String testFeaturesFileName=  baseFolder+"/"+"docs/"+Utilities.capitalizeFirstLetter(testingSets[set])+String.format("Seq%03d.csv", vid);
						int[][] testDocuments=Utilities.loadDocuments(testFeaturesFileName);
						String[][] tempArray= Utilities.readCSVFile(testFeaturesFileName, false);
						String[] docIds= new String[tempArray.length];
						for(int d=0; d<tempArray.length; d++){
							docIds[d]= tempArray[d][0];
						}
						testingModel= new LDAVB(testDocuments,trainingModel, modelFolderPath, baseModelName, varIters,varConverged);
						testingModel.infer("Test");
						double[][] topicDistributions= testingModel.getGamma();
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
									if(!Double.isNaN(topicDistributions[d][t]))
										featuresFile.print(fmt.format(topicDistributions[d][t]));
									else
										featuresFile.print("NaN");
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
				if(deleteModelFiles){
					new File(modelFolderPath+"/"+baseModelName+".model").delete();
					new File(modelFolderPath+"/"+baseModelName+"Beta.csv").delete();
				}
			}
		}
		catch(Exception e){e.printStackTrace();System.exit(1);}
	}

}
