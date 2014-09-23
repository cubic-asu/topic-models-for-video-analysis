package edu.asu.cubic.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Properties;
import java.util.concurrent.Callable;

import org.apache.commons.math3.analysis.interpolation.LoessInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.apache.commons.math3.stat.inference.TestUtils;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.unsupervised.attribute.RandomProjection;
import edu.asu.cubic.regression.BaggedRegression;
import edu.asu.cubic.regression.BoostingRegression;
import edu.asu.cubic.regression.DecisionStumpRegression;
import edu.asu.cubic.regression.DecisionTreeRegression;
import edu.asu.cubic.regression.KNNRegression;
import edu.asu.cubic.regression.LRegression;
import edu.asu.cubic.regression.RandomRegression;
import edu.asu.cubic.regression.RegressionResults;
import edu.asu.cubic.regression.SVMRegression;

public class RegressorEvaluator implements Callable<RegressionResults>{

	Properties requiredParameters;
	String regressorName;
	String[] regressorParams;
	String[] trainingSets;
	int[][] trainingSetVideos;
	String[] testingSets;
	int[][] testingSetVideos;
	int totalTestVideos;
	String trainingDataInfo;
	String testingDataInfo;
	String[] baseFeature;
	String[] featureType;
	boolean writePredictions;

	public RegressorEvaluator(Properties parameters, String rName, String[] params, String trainSets, String testSets, String bFeature, String fType, boolean writePreds){
		requiredParameters= parameters;
		regressorName= rName;
		regressorParams= params;
		baseFeature= new String[1];
		baseFeature[0]= bFeature;
		featureType= new String[1];
		featureType[0]= fType;
		writePredictions= writePreds;
		// training files to be used can be passed as a combination of
		// trainingSet,startVideo,endVideo values
		String[] tokens= trainSets.trim().split(";");
		trainingSets= new String[tokens.length];
		trainingSetVideos= new int[tokens.length][2];
		trainingDataInfo= "";
		for(int i=0; i<tokens.length; i++)
		{
			trainingSets[i]= tokens[i].split(",")[0];
			trainingSetVideos[i][0]= Integer.parseInt(tokens[i].split(",")[1]);// start video
			trainingSetVideos[i][1]= Integer.parseInt(tokens[i].split(",")[2]);// end video
			trainingDataInfo+=trainingSets[i]+"_"+trainingSetVideos[i][0]+"_"+trainingSetVideos[i][1];
		}
		tokens= testSets.split(";");
		testingSets= new String[tokens.length];
		testingSetVideos= new int[tokens.length][2];
		testingDataInfo= "";
		totalTestVideos=0;
		for(int i=0; i<tokens.length; i++)
		{
			testingSets[i]= tokens[i].split(",")[0];
			testingSetVideos[i][0]= Integer.parseInt(tokens[i].split(",")[1]);// start video
			testingSetVideos[i][1]= Integer.parseInt(tokens[i].split(",")[2]);// end video
			testingDataInfo+=testingSets[i]+"_"+testingSetVideos[i][0]+"_"+testingSetVideos[i][1];
			totalTestVideos+= testingSetVideos[i][1]-testingSetVideos[i][0]+1;
		}
		Utilities.printArray(testingSets);
		//Utilities.printArray(array)
	}

	public RegressorEvaluator(Properties parameters, String rName, String[] params, String trainSets, String testSets, String[] bFeature, String[] fType, boolean writePreds){
		requiredParameters= parameters;
		regressorName= rName;
		regressorParams= params;
		baseFeature= bFeature;
		featureType= fType;
		writePredictions= writePreds;
		// training files to be used can be passed as a combination of
		// trainingSet,startVideo,endVideo values
		String[] tokens= trainSets.trim().split(";");
		trainingSets= new String[tokens.length];
		trainingSetVideos= new int[tokens.length][2];
		trainingDataInfo= "";
		for(int i=0; i<tokens.length; i++)
		{
			trainingSets[i]= tokens[i].split(",")[0];
			trainingSetVideos[i][0]= Integer.parseInt(tokens[i].split(",")[1]);// start video
			trainingSetVideos[i][1]= Integer.parseInt(tokens[i].split(",")[2]);// end video
			trainingDataInfo+=trainingSets[i]+"_"+trainingSetVideos[i][0]+"_"+trainingSetVideos[i][1];
		}
		tokens= testSets.split(";");
		testingSets= new String[tokens.length];
		testingSetVideos= new int[tokens.length][2];
		testingDataInfo= "";
		totalTestVideos=0;
		for(int i=0; i<tokens.length; i++)
		{
			testingSets[i]= tokens[i].split(",")[0];
			testingSetVideos[i][0]= Integer.parseInt(tokens[i].split(",")[1]);// start video
			testingSetVideos[i][1]= Integer.parseInt(tokens[i].split(",")[2]);// end video
			testingDataInfo+=testingSets[i]+"_"+testingSetVideos[i][0]+"_"+testingSetVideos[i][1];
			totalTestVideos+= testingSetVideos[i][1]-testingSetVideos[i][0]+1;
		}
		Utilities.printArray(testingSets);
		//Utilities.printArray(array)
	}

	public RegressionResults call() {

		RegressionResults resultsObject= new RegressionResults();
		DecimalFormat fmt= new DecimalFormat("#.####");
		try{
			/*System.out.println("Evaluating "+regressorName);
			if(regressorParams!=null)
				Utilities.printArray(regressorParams);*/
			String regressorDetails="";
			// Load the properties file
			String baseFolder= requiredParameters.getProperty("baseFolder").trim();
			boolean dsc= Boolean.parseBoolean(requiredParameters.getProperty("dsc").trim()); // depression challenge
			boolean combineFeatures= Boolean.parseBoolean(requiredParameters.getProperty("combineFeatures").trim()); // depression challenge
			boolean smoothPredictions = Boolean.parseBoolean(requiredParameters.getProperty("smoothPredictions").trim());
			String dimensionReduction= requiredParameters.getProperty("dimensionReduction").trim();
			String response= requiredParameters.getProperty("response").trim();
			String featureTransformation= requiredParameters.getProperty("transformation").trim();
			String featureSelection= requiredParameters.getProperty("featSelection").trim();
			String samplingAlgo= requiredParameters.getProperty("sampling").trim();
			String capitalizedResponse= Utilities.capitalizeFirstLetter(response);
			boolean deleteModelFiles = Boolean.parseBoolean(requiredParameters.getProperty("deleteModelFiles").trim());
			boolean normalizeFeatures= Boolean.parseBoolean(requiredParameters.getProperty("normalizeFeatures").trim());
			if(normalizeFeatures)
				trainingDataInfo+="Norm";
			else
				trainingDataInfo+="UnNorm";
			// for each regression model do this
			String regressionModelFolder= baseFolder+"/"+baseFeature[0]+featureType[0]+"/"+response+"/"+dimensionReduction;
			if(combineFeatures || dimensionReduction.equals("MMDGMM")){
				if(dimensionReduction.equals("MMDGMM") && baseFeature.length<2){
					throw new Exception("For MMDGMM you have to supply atleast 2 different feature types. ");
				}
				regressionModelFolder= baseFolder+"/"+baseFeature[0]+featureType[0]+"/"+response+"/";
				for(int feat=0; feat< baseFeature.length; feat++){
					regressionModelFolder+= baseFeature[feat]+featureType[feat];
				}
				regressionModelFolder+= dimensionReduction;
			}
			String regressorModelPath= null;
			if(regressorName.equals("SVR")){
				// unpack the svr parameters
				/*double cParam= Double.parseDouble(regressorParams[0].trim());
				String kernelParams= regressorParams[1].trim();*/
				regressorDetails= regressorName+"_"+regressorParams[0].trim()+"_"+regressorParams[1].trim()+featureTransformation+featureSelection+trainingDataInfo;
				regressorModelPath= regressionModelFolder+"/"+regressorName+"_"+regressorParams[0].trim()+"_"+regressorParams[1].trim()+featureTransformation+featureSelection+trainingDataInfo+samplingAlgo+".model";
			}
			else if(regressorName.contains("LR")){
				boolean onlineFlag= Boolean.parseBoolean(regressorParams[0].trim());
				if(onlineFlag){
					regressorDetails= regressorName+"_"+regressorParams[0].trim()+"_"+regressorParams[1].trim()+"_"+regressorParams[2].trim()+featureTransformation+featureSelection+trainingDataInfo;
					regressorModelPath= regressionModelFolder+"/"+regressorName+"_"+regressorParams[0].trim()+"_"+regressorParams[1].trim()+"_"+regressorParams[2].trim()+featureTransformation+featureSelection+trainingDataInfo+samplingAlgo+".model";
				}
				else{
					regressorDetails= regressorName+"_"+regressorParams[0].trim()+"_"+regressorParams[2].trim()+featureTransformation+featureSelection+trainingDataInfo;
					regressorModelPath= regressionModelFolder+"/"+regressorName+"_"+regressorParams[0].trim()+"_"+regressorParams[2].trim()+featureTransformation+featureSelection+trainingDataInfo+samplingAlgo+".model";
				}
			}
			else if(regressorName.contains("DSR")){
				regressorDetails= regressorName+"_"+regressorParams[0].trim()+featureTransformation+featureSelection+trainingDataInfo;
				regressorModelPath= regressionModelFolder+"/"+regressorName+"_"+regressorParams[0].trim()+featureTransformation+featureSelection+trainingDataInfo+samplingAlgo+".model";
			}
			else if(regressorName.contains("DTR")){
				regressorDetails= regressorName+"_"+regressorParams[0].trim()+featureTransformation+featureSelection+trainingDataInfo;
				regressorModelPath= regressionModelFolder+"/"+regressorName+"_"+regressorParams[0].trim()+featureTransformation+featureSelection+trainingDataInfo+samplingAlgo+".model";
			}
			else if(regressorName.contains("BAG")){
				regressorDetails= regressorName+"_"+regressorParams[0]+"_"+regressorParams[1]+"_"+regressorParams[2]+featureTransformation+featureSelection+trainingDataInfo;
				regressorModelPath= regressionModelFolder+"/"+regressorName+"_"+regressorParams[0]+"_"+regressorParams[1]+"_"+regressorParams[2]+featureTransformation+featureSelection+trainingDataInfo+samplingAlgo+".model";
			}
			else if(regressorName.contains("BOOST")){
				regressorDetails= regressorName+"_"+regressorParams[0]+"_"+regressorParams[1]+"_"+regressorParams[2]+featureTransformation+featureSelection+trainingDataInfo;
				regressorModelPath= regressionModelFolder+"/"+regressorName+"_"+regressorParams[0]+"_"+regressorParams[1]+"_"+regressorParams[2]+featureTransformation+featureSelection+trainingDataInfo+samplingAlgo+".model";
			}
			else if(regressorName.contains("KNN")){
				regressorDetails= regressorName+"_"+regressorParams[0]+"_"+regressorParams[1]+featureTransformation+featureSelection+trainingDataInfo;
				regressorModelPath= regressionModelFolder+"/"+regressorName+"_"+regressorParams[0]+"_"+regressorParams[1]+featureTransformation+featureSelection+trainingDataInfo+samplingAlgo+".model";
			}
			else if(regressorName.contains("SLDAVB")){
				String baseModelName= Utilities.capitalizeFirstLetter(response)+dimensionReduction;
				if(combineFeatures){
					baseModelName= "";
					for(int feat=0; feat< baseFeature.length; feat++){
						baseModelName+= baseFeature[feat]+featureType[feat];
					}
					baseModelName+=(Utilities.capitalizeFirstLetter(response)+dimensionReduction);
				}
				regressionModelFolder= baseFolder+"/"+baseFeature[0]+featureType[0];
				regressorModelPath= regressionModelFolder+"/"+baseModelName+".model";
				regressorDetails= dimensionReduction;
			}
			else if(regressorName.contains("SLDA")){
				String baseModelName= Utilities.capitalizeFirstLetter(response)+regressorName+"_"+regressorParams[0];
				if(combineFeatures){
					baseModelName= "";
					for(int feat=0; feat< baseFeature.length; feat++){
						baseModelName+= baseFeature[feat]+featureType[feat];
					}
					baseModelName+=(Utilities.capitalizeFirstLetter(response)+regressorName+"_"+regressorParams[0]);
				}
				regressionModelFolder= baseFolder+"/"+baseFeature[0]+featureType[0];
				regressorModelPath= regressionModelFolder+"/"+baseModelName+"_"+regressorParams[1]+"_"+regressorParams[2]+samplingAlgo+".model";
				regressorDetails= regressorName+"_"+regressorParams[0]+"_"+regressorParams[1]+"_"+regressorParams[2];
			}
			else if(regressorName.contains("Random")){
				regressorModelPath= regressionModelFolder+"/"+regressorName;
			}
			if(!new File(regressorModelPath).exists() && !regressorName.contains("KNN") && !regressorName.contains("Random") && !regressorName.contains("SLDAVB")){ // if regression model does not exist
				System.out.println(regressorModelPath);
				System.err.println("Hey first build regression model and then get back to me");
				System.exit(1);
				return null;
			}
			else{
				String outputFolder= baseFolder+"/"+baseFeature[0]+featureType[0]+"/"+response+"/"+dimensionReduction;
				if(combineFeatures || dimensionReduction.equals("MMDGMM")){
					outputFolder= baseFolder+"/"+baseFeature[0]+featureType[0]+"/"+response+"/";
					for(int feat=0; feat< baseFeature.length; feat++){
						outputFolder+= baseFeature[feat]+featureType[feat];
					}
					outputFolder+=dimensionReduction;
				}
				if(!new File(outputFolder).exists())
					new File(outputFolder).mkdir();
				String randomFileName= "TrainFeatures"+regressorName+Utilities.generateRandomString(15);
				// This training file is only needed by KNN
				String fullTrainingFeaturesFilePath= outputFolder+"/"+randomFileName+".csv";
				if(regressorName.contains("KNN")){
					// pool all the training features and responses
					double[][] trainingFeatures= null;
					String[] docIds= null;
					Instances[] trainingInstances= null;
					if(dimensionReduction.equals("MMDGMM")){
						trainingInstances= new Instances[1];
						for(int set=0; set< trainingSets.length; set++){
							for(int trainvid=trainingSetVideos[set][0]; trainvid<=trainingSetVideos[set][1]; trainvid++){
								String trainFeaturesFileName=  baseFolder;
								for(int feat=0; feat<baseFeature.length; feat++){
									trainFeaturesFileName+= baseFeature[feat]+featureType[feat];
								}
								trainFeaturesFileName+= "/"+dimensionReduction+"/"+Utilities.capitalizeFirstLetter(trainingSets[set])+String.format("Seq%03d.csv", trainvid);
								System.out.println(trainFeaturesFileName);
								CSVLoader loader= new CSVLoader();
								loader.setFile(new File(trainFeaturesFileName));
								if(trainingInstances[0]==null){
									trainingInstances[0]= loader.getDataSet();
								}
								else{
									Instances currVidInstances= loader.getDataSet();
									for(Instance inst: currVidInstances){
										trainingInstances[0].add(inst);
									}
								}
							}
						}
					}
					else{
						trainingInstances= new Instances[baseFeature.length];
						for(int set=0; set< trainingSets.length; set++){
							for(int trainvid=trainingSetVideos[set][0]; trainvid<=trainingSetVideos[set][1]; trainvid++){
								// load training features as weka instances
								for(int feat=0; feat<baseFeature.length; feat++){
									// load training features as weka instances
									String trainFeaturesFileName=  baseFolder+"/"+baseFeature[feat]+featureType[feat]+"/"+dimensionReduction+"/"+Utilities.capitalizeFirstLetter(trainingSets[set])+String.format("Seq%03d.csv", trainvid);
									if(dimensionReduction.contains("SLDAVB")){
										trainFeaturesFileName=  baseFolder+"/"+baseFeature[feat]+featureType[feat]+"/"+Utilities.capitalizeFirstLetter(response)+dimensionReduction+"/"+Utilities.capitalizeFirstLetter(trainingSets[set])+String.format("Seq%03d.csv", trainvid);
									}
									System.out.println(trainFeaturesFileName);
									CSVLoader loader= new CSVLoader();
									loader.setFile(new File(trainFeaturesFileName));
									if(trainingInstances[feat]==null){
										trainingInstances[feat]= loader.getDataSet();
									}
									else{
										Instances currVidInstances= loader.getDataSet();
										for(Instance inst: currVidInstances){
											trainingInstances[feat].add(inst);
										}
									}
								}
							}
						}
					}
					// group all training features
					int count=0;
					docIds= new String[trainingInstances[0].numInstances()];
					for(Instance inst: trainingInstances[0]){
						docIds[count]= new String(""+inst.value(0));
						count++;
					}
					for(int feat=0; feat<trainingInstances.length; feat++){
						trainingInstances[feat].deleteAttributeAt(0);
					}
					trainingFeatures= new double[trainingInstances[0].numInstances()][];
					int totalFeats=0;
					for(int feat=0; feat<trainingInstances.length; feat++){
						totalFeats+= trainingInstances[feat].numAttributes();
					}
					for(int i=0; i< trainingInstances[0].numInstances(); i++){
						trainingFeatures[i]= new double[totalFeats];
						count=0;
						for(int type=0; type< trainingInstances.length; type++){
							for(int feat=0; feat<trainingInstances[type].numAttributes(); feat++){
								trainingFeatures[i][count]= trainingInstances[type].get(i).value(feat);
								if(featureTransformation.equalsIgnoreCase("log")){
									trainingFeatures[i][count]+= 1E-6;
									trainingFeatures[i][count]= Math.log(trainingFeatures[i][count]);
								}
								count++;
							}
						}
					}
					// group all responses
					double[] responses= new double[trainingInstances[0].numInstances()];
					count=0;
					for(int set=0; set< trainingSets.length; set++){
						for(int trainvid=trainingSetVideos[set][0]; trainvid<=trainingSetVideos[set][1]; trainvid++){
							// load training responses to an array
							String trainResponsesFileName=  baseFolder+"/"+"responses/"+Utilities.capitalizeFirstLetter(trainingSets[set])+String.format("%s%03d.csv", capitalizedResponse,trainvid);
							String[][] temp= Utilities.readCSVFile(trainResponsesFileName, false);
							for(int i=0; i<temp.length; i++){
								responses[count]= Double.parseDouble(temp[i][1]);
								count++;
							}
						}
					}
					// write all the features and responses to a csv file
					PrintWriter fullTrainingDataCSVFile= new PrintWriter(new File(fullTrainingFeaturesFilePath));
					fullTrainingDataCSVFile.print("DocId,");
					for(int ind=1; ind<=trainingFeatures[0].length; ind++ )
						fullTrainingDataCSVFile.print("Feature"+ind+",");
					fullTrainingDataCSVFile.println("Class1");
					for(int m=0; m<trainingFeatures.length; m++){
						fullTrainingDataCSVFile.print(docIds[m]+",");
						for(int ind=0; ind<trainingFeatures[0].length; ind++ )
							fullTrainingDataCSVFile.print(fmt.format(trainingFeatures[m][ind])+",");
						fullTrainingDataCSVFile.println(responses[m]);
					}
					fullTrainingDataCSVFile.close();
				}
				regressorDetails= regressorDetails.replace(',', '_');
				double[] crossCorrelations= new double[totalTestVideos];
				double[] rmsErrors= new double[totalTestVideos];
				double[] rSquaredValues = new double[totalTestVideos];
				double[] meanAbsErrors = new double[totalTestVideos];
				double[][] predictedValues= new double[totalTestVideos][];
				ArrayList<ArrayList<Integer>> ignoredIndices= new ArrayList<ArrayList<Integer>>();
				int vidCount=0;
				System.out.println(trainingDataInfo+ " "+testingDataInfo);
				for(int set=0; set<testingSets.length; set++){
					//PrintWriter resultsFile= new PrintWriter(new FileWriter(regressionModelFolder + "/Results.txt",true));
					//resultsFile.print(returnString.replace(',', '_')+","+testingSets[set]+","+totalVideos+","+capitalizedResponse+",");
					for(int setVid=testingSetVideos[set][0]; setVid<=testingSetVideos[set][1]; setVid++){
						String testFeaturesFolder= baseFolder+"/"+baseFeature[0]+featureType[0]+"/"+response+"/"+dimensionReduction;
						if(combineFeatures || dimensionReduction.equals("MMDGMM")){
							testFeaturesFolder= baseFolder+"/";
							for(int f=0; f<baseFeature.length; f++)
								testFeaturesFolder+=baseFeature[f]+featureType[f];
							testFeaturesFolder+= "/"+response+"/"+dimensionReduction;
						}
						/*if(!new File(testFeaturesFolder).exists())
							new File(testFeaturesFolder).mkdir();*/
						String testCSVFilePath= testFeaturesFolder+"/"+Utilities.capitalizeFirstLetter(testingSets[set])+setVid+"Features"+Utilities.generateRandomString(15)+".csv";
						//ArrayList<Integer> retainedSampleIndices= new ArrayList<Integer>();
						double[][] testingFeatures= null;
						String[] docIds= null;
						Instances[] testingInstances= null;
						double[] responses= null;
						ArrayList<Integer> ignoredSampleIndicesInThisVid= new ArrayList<Integer>();
						if(!new File(testCSVFilePath).exists()){
							//System.out.println("Test Video: "+vid);
							if(dimensionReduction.equals("MMDGMM")){
								testingInstances= new Instances[1];
								try{
									String testFeaturesFileName=  baseFolder+"/";
									for(int feat=0; feat<baseFeature.length; feat++){
										testFeaturesFileName+= baseFeature[feat]+featureType[feat];
									}
									testFeaturesFileName+= "/"+dimensionReduction+"/"+Utilities.capitalizeFirstLetter(testingSets[set])+String.format("Seq%03d.csv", setVid);
									System.out.println(testFeaturesFileName);
									CSVLoader loader= new CSVLoader();
									loader.setFile(new File(testFeaturesFileName));
									testingInstances[0]= loader.getDataSet();
								}
								catch(IOException e){
									System.err.println("Error in Video: "+ setVid);
									throw e;
								}
							}
							else{
								testingInstances= new Instances[baseFeature.length];
								for(int feat=0; feat<baseFeature.length; feat++){
									// load testing features as weka instances
									try{
										String testFeaturesFileName=  baseFolder+"/"+baseFeature[feat]+featureType[feat]+"/"+dimensionReduction+"/"+Utilities.capitalizeFirstLetter(testingSets[set])+String.format("Seq%03d.csv", setVid);
										if(regressorName.contains("SLDAVB") || dimensionReduction.contains("SLDAVB"))
											testFeaturesFileName=  baseFolder+"/"+baseFeature[feat]+featureType[feat]+"/"+Utilities.capitalizeFirstLetter(response)+dimensionReduction+"/"+Utilities.capitalizeFirstLetter(testingSets[set])+String.format("Seq%03d.csv", setVid);
										else
											if(regressorName.contains("SLDA") || dimensionReduction.contains("SLDA"))
												testFeaturesFileName=  baseFolder+"/"+baseFeature[feat]+featureType[feat]+"/"+response+"/"+dimensionReduction+"/"+Utilities.capitalizeFirstLetter(testingSets[set])+String.format("Seq%03d.csv", setVid);
										System.out.println(testFeaturesFileName);
										CSVLoader loader= new CSVLoader();
										loader.setFile(new File(testFeaturesFileName));
										testingInstances[feat]= loader.getDataSet();
										for(int i=testingInstances[feat].numInstances()-1; i>=0; i--){
											if(testingInstances[feat].get(i).hasMissingValue()){ // if there is a NaN remove this instance
												ignoredSampleIndicesInThisVid.add(i);
											}
										}
									}
									catch(IOException e){
										System.err.println("Error in Video: "+ setVid);
										throw e;
									}
								}
								// ensure that the indices to be ignored are removed from instances of all features
								for(int feat=0; feat<baseFeature.length; feat++){
									for(int i=testingInstances[feat].numInstances()-1; i>=0; i--){
										if(ignoredSampleIndicesInThisVid.contains(i)){ 
											testingInstances[feat].remove(i);
										}
									}
								}
							}
							int count=0;
							docIds= new String[testingInstances[0].numInstances()];
							for(Instance inst: testingInstances[0]){
								docIds[count]= new String(""+(int)inst.value(0));
								count++;
							}
							for(int feat=0; feat<testingInstances.length; feat++){
								testingInstances[feat].deleteAttributeAt(0);
							}
							testingFeatures= new double[testingInstances[0].numInstances()][];
							count=0;
							int totalFeats=0;
							for(int feat=0; feat<testingInstances.length; feat++){
								totalFeats+= testingInstances[feat].numAttributes();
							}
							for(int i=0; i< testingInstances[0].numInstances(); i++){
								testingFeatures[i]= new double[totalFeats];
								count=0;
								for(int type=0; type< testingInstances.length; type++){
									for(int feat=0; feat<testingInstances[type].numAttributes(); feat++){
										testingFeatures[i][count]= testingInstances[type].get(i).value(feat);
										if(featureTransformation.equalsIgnoreCase("log")){
											testingFeatures[i][count]+= 1E-6;
											testingFeatures[i][count]= Math.log(testingFeatures[i][count]);
										}
										count++;
									}
								}
							}

							count=0;
							responses= new double[testingInstances[0].numInstances()];
							String testResponsesFileName=  baseFolder+"/"+"responses/"+Utilities.capitalizeFirstLetter(testingSets[set])+String.format("%s%03d.csv", capitalizedResponse,setVid);
							String[][] temp= Utilities.readCSVFile(testResponsesFileName, false);
							/*String[][] temp1=null;
							if(samplingAlgo.equalsIgnoreCase("change")){
								String testChangesFileName= baseFolder+"/"+baseFeature[0]+"Responses/"+Utilities.capitalizeFirstLetter(testingSets[set])+String.format("%s%03dChangeIndicators.csv", capitalizedResponse,setVid);
								temp1= Utilities.readCSVFile(testChangesFileName, false);
							}*/
							for(int i=0; i<temp.length; i++){
								if(!ignoredSampleIndicesInThisVid.contains(i)){
									responses[count]= Double.parseDouble(temp[i][1]);
									/*if(samplingAlgo.equalsIgnoreCase("change")){
										if(Integer.parseInt(temp1[i][1])==1)
											retainedSampleIndices.add(count);
									}*/
									count++;
								}
							}
							if(featureTransformation.equalsIgnoreCase("smooth")){
								double[][] topics = testingFeatures;
								LoessInterpolator si= new LoessInterpolator();
								double[] xVals = new double[topics.length];
								for(int i=0; i < xVals.length; i++)
									xVals[i] = i+1;

								for(int topic = 1; topic < topics[0].length; topic++){
									double[] yVals = new double[xVals.length];
									for(int i=0; i < xVals.length; i++)
										yVals[i] = topics[i][topic];
									double[] smoothedTopic = new double[xVals.length];
									PolynomialSplineFunction psf= si.interpolate(xVals, yVals);
									for(int i=0; i < xVals.length; i++){
										double val= psf.value(xVals[i]);
										if(Double.isInfinite(val) || Double.isNaN(val)){
											if(i>1) val = testingFeatures[i-1][topic];
											else val = 0.0;
										}
										smoothedTopic[i] = val;
									}
									// scale back the smoothed topic values to lie between 0 and 1
									//smoothedTopic = Utilities.scaleData(smoothedTopic, 1, 10);
									for(int i=0; i < xVals.length; i++){
										testingFeatures[i][topic] = smoothedTopic[i];
									}
								}
								/*for(int i=0; i< testingFeatures.length; i++){
									double totalSum = 0;
									for(int topic = 1; topic < testingFeatures[0].length; topic++)
										totalSum += testingFeatures[i][topic];
									for(int topic = 1; topic < testingFeatures[0].length; topic++)
										testingFeatures[i][topic] /= totalSum;
								}*/
							}
							// write all the features and responses to a csv file
							PrintWriter testingDataCSVFile= new PrintWriter(new File(testCSVFilePath));
							testingDataCSVFile.print("DocId,");
							for(int ind=1; ind<=testingFeatures[0].length; ind++ )
								testingDataCSVFile.print("Feature"+ind+",");
							testingDataCSVFile.println("Class1");
							for(int m=0; m<testingFeatures.length; m++){
								//if(retainedSampleIndices.isEmpty() || retainedSampleIndices.contains(m)){
								testingDataCSVFile.print(docIds[m]+",");
								for(int ind=0; ind<testingFeatures[0].length; ind++ )
									testingDataCSVFile.print(fmt.format(testingFeatures[m][ind])+",");
								testingDataCSVFile.println(responses[m]);
								//}
							}
							testingDataCSVFile.flush();
							testingDataCSVFile.close();
						}
						// generate predictions using regression model
						double[] sampledPredictions= null;
						// if it is SLDA which by itself is regression model, load the coefficients
						if(regressorName.equals("SLDAVB")){
							sampledPredictions= SLDAVBBuilder.generateSLDAPredictions(regressorModelPath, testCSVFilePath);
						}
						else if(regressorName.equals("SLDA")){
							sampledPredictions= SLDABuilder.generateSLDAPredictions(regressorModelPath, testCSVFilePath);
						} 
						else if(regressorName.equals("SVR")){
							double cParam= Double.parseDouble(regressorParams[0].trim());
							String kernelParams= regressorParams[1].trim();
							SVMRegression svrModelBuilder= new SVMRegression(cParam,kernelParams,normalizeFeatures);
							svrModelBuilder.setFeatureTransformation(featureTransformation);
							svrModelBuilder.setFeatureSelection(featureSelection);
							sampledPredictions= svrModelBuilder.evaluateRegressionModel(regressorModelPath, testCSVFilePath);
						}
						else if(regressorName.contains("LR")){
							boolean onlineFlag= Boolean.parseBoolean(regressorParams[0].trim());
							boolean positiveCoeffs= Boolean.parseBoolean(regressorParams[2].trim());
							if(onlineFlag){
								double learningRate= Double.parseDouble(regressorParams[1].trim());
								LRegression lrModelBuilder= new LRegression(onlineFlag,learningRate,normalizeFeatures,positiveCoeffs);
								lrModelBuilder.setFeatureTransformation(featureTransformation);
								lrModelBuilder.setFeatureSelection(featureSelection);
								sampledPredictions= lrModelBuilder.evaluateRegressionModel(regressorModelPath, testCSVFilePath);
							}
							else{
								LRegression lrModelBuilder= new LRegression(onlineFlag,normalizeFeatures,positiveCoeffs);
								lrModelBuilder.setFeatureTransformation(featureTransformation);
								lrModelBuilder.setFeatureSelection(featureSelection);
								sampledPredictions= lrModelBuilder.evaluateRegressionModel(regressorModelPath, testCSVFilePath);
							}
						}
						else if(regressorName.contains("DSR")){
							boolean onlineFlag= Boolean.parseBoolean(regressorParams[0].trim());
							DecisionStumpRegression dsrModelBuilder= new DecisionStumpRegression(onlineFlag);
							dsrModelBuilder.setFeatureTransformation(featureTransformation);
							dsrModelBuilder.setFeatureSelection(featureSelection);
							sampledPredictions= dsrModelBuilder.evaluateRegressionModel(regressorModelPath, testCSVFilePath);
							resultsObject.setModelWeight(dsrModelBuilder.getRegressorTrainingWeight());
						}
						else if(regressorName.contains("DTR")){
							boolean pruneTree= Boolean.parseBoolean(regressorParams[0].trim());
							DecisionTreeRegression dtrModelBuilder= new DecisionTreeRegression(pruneTree);
							dtrModelBuilder.setFeatureTransformation(featureTransformation);
							dtrModelBuilder.setFeatureSelection(featureSelection);
							sampledPredictions= dtrModelBuilder.evaluateRegressionModel(regressorModelPath, testCSVFilePath);
							resultsObject.setModelWeight(dtrModelBuilder.getRegressorTrainingWeight());
						}
						else if(regressorName.contains("BAG")){
							int iters= Integer.parseInt(regressorParams[0].trim());
							int bagSize= Integer.parseInt(regressorParams[1].trim());
							String innerRegressorParams= regressorParams[2].trim();
							BaggedRegression baggedModelBuilder= new BaggedRegression(iters,bagSize,innerRegressorParams);
							baggedModelBuilder.setFeatureTransformation(featureTransformation);
							baggedModelBuilder.setFeatureSelection(featureSelection);
							sampledPredictions= baggedModelBuilder.evaluateRegressionModel(regressorModelPath, testCSVFilePath);
						}
						else if(regressorName.contains("BOOST")){
							int iters= Integer.parseInt(regressorParams[0].trim());
							double shrinkage= Double.parseDouble(regressorParams[1].trim());
							String innerRegressorParams= regressorParams[2].trim();
							BoostingRegression boostedModelBuilder= new BoostingRegression(iters,shrinkage,innerRegressorParams);
							boostedModelBuilder.setFeatureTransformation(featureTransformation);
							boostedModelBuilder.setFeatureSelection(featureSelection);
							sampledPredictions= boostedModelBuilder.evaluateRegressionModel(regressorModelPath, testCSVFilePath);
						}
						else if(regressorName.contains("KNN")){
							int neighbors= Integer.parseInt(regressorParams[0].trim());
							String distanceMetric= regressorParams[1].trim();
							KNNRegression knnModelBuilder= new KNNRegression(neighbors,distanceMetric,normalizeFeatures);
							knnModelBuilder.setFeatureTransformation(featureTransformation);
							knnModelBuilder.setFeatureSelection(featureSelection);
							// KNN needs the training instances while evaluation because it is lazy classifier
							//knnModelBuilder.crossValidation(fullTrainingFeaturesFilePath, testCSVFilePath);
							// Evaluate KNN
							sampledPredictions= knnModelBuilder.evaluateRegressionModel(fullTrainingFeaturesFilePath, testCSVFilePath);
						}
						else if(regressorName.contains("Random")){
							RandomRegression randomPredictor= new RandomRegression();
							sampledPredictions= randomPredictor.evaluateRegressionModel(fullTrainingFeaturesFilePath, testCSVFilePath);
						}
						//Utilities.printArray("Sampled Unscaled", sampledPredictions);
						//Utilities.printArray("Sampled Scaled", sampledPredictions);
						// load sampled ids
						String testResponsesFileName=  baseFolder+"/"+"responses/"+Utilities.capitalizeFirstLetter(testingSets[set])+String.format("%s%03d.csv", capitalizedResponse,setVid);
						String[][] temp= Utilities.readCSVFile(testResponsesFileName, false);
						double[] sampledIds;
						if(!ignoredSampleIndicesInThisVid.isEmpty())
							sampledIds= new double[responses.length];
						else
							sampledIds =new double[temp.length];
						double[] correctLabels= new double[sampledIds.length];
						int count=0;
						for(int i=0; i<temp.length; i++){
							if(dsc)
								sampledIds[i]= Double.parseDouble(temp[i][0]);
							else{
								if(ignoredSampleIndicesInThisVid.isEmpty() || !ignoredSampleIndicesInThisVid.contains(i)){
									sampledIds[count]= i+1;
									correctLabels[count]= Double.parseDouble(temp[i][1]);
									count++;
								}
							}
						}
						double[] unSampledIds= sampledIds;
						double[] unSampledPredictions= null;
						try{
							// smoothen the predictions (note that we are not saying interpolate the predictions)
							if(smoothPredictions){
								LoessInterpolator si= new LoessInterpolator();
								PolynomialSplineFunction psf= si.interpolate(sampledIds, sampledPredictions);
								unSampledPredictions= new double[unSampledIds.length];
								for(int i=0; i<unSampledIds.length;i++){
									unSampledPredictions[i]= psf.value(unSampledIds[i]);
								}
							}
							else{
								unSampledPredictions = sampledPredictions;
							}
						}
						catch(Exception e){
							System.out.println("Error for video: "+testingSets[set]+" "+setVid);
							Utilities.printArray("Original SampleIds",sampledIds);
							e.printStackTrace();
							throw e;
						}
						//Utilities.printArray("Unsampled Scaled", unSampledPredictions);
						// write predictions to file
						if(writePredictions){
							String predictionsFolder= baseFolder+"/FinalResults/"+baseFeature[0]+featureType[0]+dimensionReduction;
							if(combineFeatures || dimensionReduction.equals("MMDGMM")){
								predictionsFolder= baseFolder+"/FinalResults/";
								for(int feat=0; feat<baseFeature.length; feat++)
									predictionsFolder+= baseFeature[feat]+featureType[feat];
								predictionsFolder+= dimensionReduction;
							}
							if(!new File(predictionsFolder).exists())
								new File(predictionsFolder).mkdir();
							predictionsFolder+="/"+regressorDetails;
							if(!new File(predictionsFolder).exists())
								new File(predictionsFolder).mkdir();
							predictionsFolder+= "/"+testingSets[set];
							if(!new File(predictionsFolder).exists())
								new File(predictionsFolder).mkdir();
							predictionsFolder+= "/Instance";
							if(!new File(predictionsFolder).exists())
								new File(predictionsFolder).mkdir();
							if(dsc){
								for(int i=0; i<unSampledIds.length; i++){
									String videoNamesFilename= baseFolder+"/"+testingSets[set]+"VideoNames.txt";
									String videoName= Utilities.readCSVFile(videoNamesFilename, false)[i][0];
									String predictionsFilePath= predictionsFolder+"/"+videoName+"-"+capitalizedResponse+".csv";
									PrintWriter predictionsFile= new PrintWriter(predictionsFilePath);
									predictionsFile.print(unSampledPredictions[i]);
									predictionsFile.close();
								}
							}
							else{
								// Load the original video ids from file
								//String videoNamesFilename= baseFolder+"/"+testingSets[set]+"VideoNames.txt";
								//String videoName= Utilities.readCSVFile(videoNamesFilename, false)[setVid-1][0];
								String predictionsFilePath= predictionsFolder+"/"+(vidCount+1)+"-"+response.toUpperCase()+".csv";
								PrintWriter predictionsFile= new PrintWriter(predictionsFilePath);
								for(int i=0; i<unSampledIds.length; i++)
									predictionsFile.println(unSampledPredictions[i]+","+unSampledIds[i]);
								predictionsFile.close();
								// Write the predictions sampled at every 60 secs
								/*predictionsFilePath= predictionsFolder+"/"+Utilities.capitalizeFirstLetter(testingSets[set])+capitalizedResponse+String.format("%03d",setVid)+".csv";
								predictionsFile= new PrintWriter(predictionsFilePath);
								for(int i=0; i<unSampledIds.length; i++){
									//if(i%60==0)
									predictionsFile.println(unSampledPredictions[i]+","+unSampledIds[i]);
								}
								predictionsFile.close();
								predictionsFilePath= predictionsFolder+"/"+(vidCount+1)+"-"+response.toUpperCase()+"Sampled.csv";
								predictionsFile= new PrintWriter(predictionsFilePath);
								for(int i=0; i<sampledIds.length; i++){
									predictionsFile.println(sampledPredictions[i]+","+sampledIds[i]);
								}
								predictionsFile.close();*/
							}
						}
						crossCorrelations[vidCount]= Utilities.calculateCrossCorrelation(unSampledPredictions,correctLabels );
						rmsErrors[vidCount]= Utilities.calculateRMSError(unSampledPredictions,correctLabels );
						rSquaredValues[vidCount] = Utilities.calculateRSquared(unSampledPredictions,correctLabels );
						meanAbsErrors[vidCount] = Utilities.calculateMeanAbsoluteError(unSampledPredictions,correctLabels );
						predictedValues[vidCount]= unSampledPredictions;
						vidCount++;
						// calculate the correlation coefficients and mean squared errors
						// delete the test file
						new File(testCSVFilePath).delete();
						ignoredIndices.add(ignoredSampleIndicesInThisVid);
					}

				}
				//meanCrossCorrelation= meanCrossCorrelation/totalVideos;
				//meanRMSError= meanRMSError/totalVideos;
				//resultsString+= Math.abs(meanCrossCorrelation)+","+meanRMSError+"\n";
				//resultsFile.println(meanCrossCorrelation+","+meanRMSError);
				//resultsFile.close();
				// call the synchronized method to write the results to file
				//SynchronizedWrite.getInstance().writeToFile(regressionModelFolder + "/Results.txt", resultsString);
				resultsObject.setCrossCorrelations(crossCorrelations);
				resultsObject.setRmsErrors(rmsErrors);
				resultsObject.setrSquares(rSquaredValues);
				resultsObject.setMeanAbsoluteErrors(meanAbsErrors);
				resultsObject.setPredictedLabels(predictedValues);
				resultsObject.setRegressorDetails(regressorDetails);
				resultsObject.setTestDataDetails(testingDataInfo);
				resultsObject.setIgnoredIndices(ignoredIndices);
				if(regressorName.contains("KNN")){
					new File(fullTrainingFeaturesFilePath).delete();
				}
				if(deleteModelFiles){
					new File(regressorModelPath).delete();
				}
			}

		}
		catch(Exception e){
			System.out.println("Exception caugt in Regressor Evaluator");
			e.printStackTrace();System.exit(1);
		}
		return resultsObject;
	}

}
