package edu.asu.cubic.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.Properties;

import edu.asu.cubic.dimensionality_reduction.SLDAGibbs;

public class SLDABuilder {

	public static void extractSLDAFeatures(Properties requiredParameters) {
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
			String response= requiredParameters.getProperty("response").trim();
			int numTopics= Integer.parseInt(requiredParameters.getProperty("dimensionReduction").trim().split("_")[1]);
			int m_Iterations= Integer.parseInt(requiredParameters.getProperty("dimensionReduction").trim().split("_")[2]);
			int e_Iterations= Integer.parseInt(requiredParameters.getProperty("dimensionReduction").trim().split("_")[3]);
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
			String modelFolderPath= baseFolder+"/TopicModels";//++gibbsIterations+".model";
			String baseModelName= baseFeature[0]+featureType[0]+Utilities.capitalizeFirstLetter(response)+"SLDA"+"_"+numTopics;
			if(!new File(modelFolderPath+"/"+baseModelName+"_"+m_Iterations+"_"+e_Iterations+".model").exists()){
				int[][] docsPerVidPerSet= new int[trainingSets.length][];
				int totalDocs=0;
				// load the total number of docs
				for(int set=0; set< trainingSets.length; set++){
					docsPerVidPerSet[set]= new int[trainingSetVideos[set][1]-trainingSetVideos[set][0]+1];
					String fileName= baseFolder+"/"+baseFeature[0]+featureType[0]+"Docs/"+Utilities.capitalizeFirstLetter(trainingSets[set])+"Frames.txt";
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
				// load training documents and annotations
				int[][] trainingDocuments= new int[totalDocs][];
				double[] trainingAnnotations= new double[totalDocs];
				String[] docIds= new String[totalDocs];
				int currDocs=0;
				for(int set=0; set< trainingSets.length; set++){
					for(int vid=trainingSetVideos[set][0]; vid<=trainingSetVideos[set][1]; vid++){
						String trainFeaturesFileName=  baseFolder+"/"+baseFeature[0]+featureType[0]+"Docs/"+Utilities.capitalizeFirstLetter(trainingSets[set])+String.format("Seq%03d.csv", vid);
						int[][] tempDocuments=Utilities.loadDocuments(trainFeaturesFileName);
						String[][] tempDocIds= Utilities.readCSVFile(trainFeaturesFileName, false); 
						for(int d=currDocs; d<currDocs+tempDocuments.length; d++){
							trainingDocuments[d]= tempDocuments[d-currDocs];
							docIds[d]= tempDocIds[d-currDocs][0];
						}
						String trainAnnotationsFileName=  baseFolder+"/"+baseFeature[0]+featureType[0]+"Responses/"+Utilities.capitalizeFirstLetter(trainingSets[set])+String.format("%s%03d.csv", Utilities.capitalizeFirstLetter(response),vid);
						String[][] tempAnnotations= Utilities.readCSVFile(trainAnnotationsFileName, false);
						for(int d=currDocs; d<currDocs+tempDocuments.length; d++){
							trainingAnnotations[d]= Double.parseDouble(tempAnnotations[d-currDocs][1]);
						}
						currDocs+= tempDocuments.length;
					}
				}
				double[] alphas = new double[numTopics];
				double[][] betas = new double[numTopics][V];
				for(int i= 0; i<numTopics; i++)
					alphas[i]=50/numTopics;
				for(int i= 0; i<numTopics; i++)
					for(int j=0; j<V; j++)
						betas[i][j]=0.02;
				System.out.println("Training SLDA model");
				SLDAGibbs trainingModel= new SLDAGibbs(trainingDocuments, trainingAnnotations,V,numTopics, alphas, betas,modelFolderPath, baseModelName,1.0E-3);
				trainingModel.setIterations(e_Iterations,m_Iterations,0);
				trainingModel.initialStateForTraining();
				trainingModel.gibbs();
				double[][] topicDistributions= trainingModel.getTheta();
				// generate topic distributions for training docs
				currDocs=0;
				for(int set=0; set< trainingSets.length; set++){
					for(int vid=trainingSetVideos[set][0]; vid<=trainingSetVideos[set][1]; vid++){
						String trainFeaturesFolder= baseFolder+"/"+baseFeature[0]+featureType[0]+Utilities.capitalizeFirstLetter(response)+"SLDA"+"_"+numTopics+"_"+m_Iterations+"_"+e_Iterations;
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
				// write the model to file
				// clean the model by removing unnecessary variables
				/*trainingModel.cleanUpVariables();
				ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelFilePath));
				oos.writeObject(trainingModel);
				oos.close();*/
			}
			// Generate the topic features for test data
			tokens= requiredParameters.getProperty("testingSets").trim().split(";");
			String[] testingSets= new String[tokens.length];
			int[][] testingSetVideos= new int[tokens.length][2];
			for(int i=0; i<tokens.length; i++)
			{
				testingSets[i]= tokens[i].split(",")[0];
				testingSetVideos[i][0]= Integer.parseInt(tokens[i].split(",")[1]);// start video
				testingSetVideos[i][1]= Integer.parseInt(tokens[i].split(",")[2]);// end video
			}
			if(testingSets.length!=0){
				System.out.println("Generating SLDA Test features");
				SLDAGibbs trainingModel= (SLDAGibbs)new ObjectInputStream(new FileInputStream(modelFolderPath+"/"+baseModelName+"_"+m_Iterations+"_"+e_Iterations+".model")).readObject();
				tokens= requiredParameters.getProperty("testingSets").trim().split(";");
				SLDAGibbs testingModel= null;
				for(int set=0; set< testingSets.length; set++){
					for(int vid=testingSetVideos[set][0]; vid<=testingSetVideos[set][1]; vid++){
						String testFeaturesFolder= baseFolder+"/"+baseFeature[0]+featureType[0]+Utilities.capitalizeFirstLetter(response)+"SLDA"+"_"+numTopics+"_"+m_Iterations+"_"+e_Iterations;
						if(!new File(testFeaturesFolder).exists())
							new File(testFeaturesFolder).mkdir();
						String testCSVFilePath= testFeaturesFolder+"/"+Utilities.capitalizeFirstLetter(testingSets[set])+"Seq"+String.format("%03d",vid)+".csv";
						if(!new File(testCSVFilePath).exists()){
							System.out.println("Test Video: "+vid);
							String testFeaturesFileName=  baseFolder+"/"+baseFeature[0]+featureType[0]+"Docs/"+Utilities.capitalizeFirstLetter(testingSets[set])+String.format("Seq%03d.csv", vid);
							int[][] testDocuments=Utilities.loadDocuments(testFeaturesFileName);
							String[][] tempArray= Utilities.readCSVFile(testFeaturesFileName, false);
							String[] docIds= new String[tempArray.length];
							for(int d=0; d<tempArray.length; d++){
								docIds[d]= tempArray[d][0];
							}
							testingModel= new SLDAGibbs(testDocuments,trainingModel,numTopics);
							testingModel.initialStateForUnseenDocs();
							testingModel.configure(20, 4, 1, 1);
							testingModel.gibbs();
							double[][] topicDistributions= testingModel.getTheta();
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

	/*
	 * This method is just a utility function that takes a set of SLDA features and a SLDA model, extracts the 
	 * coefficients, multiplies them with the features and generates predictions
	 */
	public static double[] generateSLDAPredictions(String trainingModelFilePath, String testingFilePath) throws Exception
	{
		String[][] featuresAsStrings= Utilities.readCSVFile(testingFilePath, true);
		double[] predictions= new double[featuresAsStrings.length];
		SLDAGibbs trainingModel= (SLDAGibbs)new ObjectInputStream(new FileInputStream(trainingModelFilePath)).readObject();
		double[] coeffs= trainingModel.getB();
		for(int sample=0; sample< featuresAsStrings.length; sample++)
		{	predictions[sample]= coeffs[coeffs.length-1];
			for(int feat=1;feat <featuresAsStrings[0].length-1; feat++){ // ignore first and last features 
				//System.out.print(coeffs[feat-1]*Double.parseDouble(featuresAsStrings[sample][feat])+",");//coeffs[feat-1]+"*"+Double.parseDouble(featuresAsStrings[sample][feat])+"= "+
				predictions[sample]+= coeffs[feat-1]*Double.parseDouble(featuresAsStrings[sample][feat]); 
			}
			//System.out.println();
		}
		Utilities.printArray("",predictions);
		return predictions;
	}

}
