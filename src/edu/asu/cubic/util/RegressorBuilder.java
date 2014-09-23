package edu.asu.cubic.util;

import java.io.File;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Properties;
import java.util.Random;
import java.util.concurrent.Callable;

import org.apache.commons.math3.analysis.interpolation.LoessInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.math.plot.utils.Array;

import com.google.common.base.Optional;

import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import edu.asu.cubic.distances.WeightedCosine;
import edu.asu.cubic.regression.BaggedRegression;
import edu.asu.cubic.regression.BoostingRegression;
import edu.asu.cubic.regression.DecisionStumpRegression;
import edu.asu.cubic.regression.DecisionTreeRegression;
import edu.asu.cubic.regression.LRegression;
import edu.asu.cubic.regression.SVMRegression;

public class RegressorBuilder implements Callable<String>{

	Properties requiredParameters;
	String regressorName;
	String[] regressorParams;
	String[] trainingSets;
	int[][] trainingSetVideos;
	String trainingDataInfo;
	String[] baseFeature;
	String[] featureType;

	public RegressorBuilder(Properties parameters, String rName, String[] params, String tSets, String bFeature, String fType){
		requiredParameters= parameters;
		regressorName= rName;
		regressorParams= params;
		baseFeature= new String[1];
		baseFeature[0]= bFeature;
		featureType= new String[1];
		featureType[0]= fType;
		// training files to be used can be passed as a combination of
		// trainingSet,startVideo,endVideo values
		String[] tokens= tSets.trim().split(";");
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
	}

	public RegressorBuilder(Properties parameters, String rName, String[] params, String tSets, String[] bFeature, String[] fType){
		requiredParameters= parameters;
		regressorName= rName;
		regressorParams= params;
		baseFeature= bFeature;
		featureType= fType;
		// training files to be used can be passed as a combination of
		// trainingSet,startVideo,endVideo values
		String[] tokens= tSets.trim().split(";");
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
	}

	public String call() {
		String returnString="";
		DecimalFormat fmt= new DecimalFormat("#.####");
		try {
			/*System.out.print("Building "+regressorName);
			if(regressorParams!=null)
				Utilities.printArray(regressorParams);*/
			// Load the properties file
			String baseFolder= requiredParameters.getProperty("baseFolder").trim();
			String response= requiredParameters.getProperty("response").trim();
			String dimensionReduction= requiredParameters.getProperty("dimensionReduction").trim();
			String capitalizedResponse= Utilities.capitalizeFirstLetter(response);
			String featureTransformation= requiredParameters.getProperty("transformation").trim();
			String featureSelection= requiredParameters.getProperty("featSelection").trim();
			String samplingAlgo= requiredParameters.getProperty("sampling").trim();
			//String[] regressors= requiredParameters.getProperty("regressors").trim().split(",");
			boolean combineFeatures= Boolean.parseBoolean(requiredParameters.getProperty("combineFeatures").trim());
			boolean normalizeFeatures= Boolean.parseBoolean(requiredParameters.getProperty("normalizeFeatures").trim());
			if(normalizeFeatures)
				trainingDataInfo+="Norm";
			else
				trainingDataInfo+="UnNorm";
			String outputFolder= baseFolder + "/"+ baseFeature[0] + featureType[0] +"/"+response;
			if(!new File(outputFolder).exists()){
				new File(outputFolder).mkdir();
			}
			outputFolder += "/"+dimensionReduction;
			if(regressorName.contains("SLDA")){
				outputFolder= baseFolder + "/"+ baseFeature[0] + featureType[0] + "/"+response+"/"+dimensionReduction;
			}
			if(dimensionReduction.equals("MMDGMM")){
				if(baseFeature.length<2){
					throw new Exception("For MMDGMM you have to supply atleast 2 different feature types. ");
				}
				outputFolder= baseFolder + "/"+ baseFeature[0] + featureType[0] +"/"+response+"/"+baseFeature[0]+featureType[0]+baseFeature[1]+featureType[1]+dimensionReduction;
			}
			if(combineFeatures){
				outputFolder= baseFolder + "/"+ baseFeature[0] + featureType[0] +"/"+response+"/";
				for(int feat=0; feat< baseFeature.length; feat++){
					outputFolder+= baseFeature[feat]+featureType[feat];
				}
				if(regressorName.contains("SLDA"))
					outputFolder+=(capitalizedResponse+dimensionReduction);
				else
					outputFolder+=dimensionReduction;
			}
			if(!new File(outputFolder).exists()){
				System.out.println("Creating folder: "+outputFolder);
				new File(outputFolder).mkdir();
			}
			// generate a random name for the training file
			String randomFileName= "TrainFeatures"+regressorName+Utilities.generateRandomString(15);
			String fullTrainingFeaturesFilePath= outputFolder+"/"+randomFileName+".csv";
			int totalVids= 0;
			for(int set=0; set< trainingSets.length; set++){
				for(int vid=trainingSetVideos[set][0]; vid<=trainingSetVideos[set][1]; vid++){
					totalVids++;
				}
			}
			// pool all the training features and responses
			double[][][] trainingFeatures= new double[totalVids][][];
			double[][] responses= new double[totalVids][];
			ArrayList<ArrayList<Integer>> retainedSampleIndices= new ArrayList<ArrayList<Integer>>();
			String[][] docIds= new String[totalVids][];
			Instances[] trainingInstances= null;
			int totalSamples=0;
			int vidIndex=0;
			for(int set=0; set< trainingSets.length; set++){
				for(int vid=trainingSetVideos[set][0]; vid<=trainingSetVideos[set][1]; vid++){
					ArrayList<Integer> ignoredSampleIndices= new ArrayList<Integer>();
					if(dimensionReduction.equals("MMDGMM")){
						trainingInstances= new Instances[1];
						String trainFeaturesFileName= baseFolder + "/"+ baseFeature[0] + featureType[0] +"/";
						for(int feat=0; feat< baseFeature.length; feat++){
							trainFeaturesFileName+= baseFeature[feat]+featureType[feat];
						}
						trainFeaturesFileName+= dimensionReduction+"/"+Utilities.capitalizeFirstLetter(trainingSets[set])+String.format("Seq%03d.csv", vid);
						CSVLoader loader= new CSVLoader();
						try{
							loader.setFile(new File(trainFeaturesFileName));
						}
						catch(Exception e){
							System.err.println("Unable to load file: "+ trainFeaturesFileName);
							throw e;
						}
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
					else{
						trainingInstances= new Instances[baseFeature.length];
						for(int feat=0; feat<baseFeature.length; feat++){
							// load training features as weka instances
							String trainFeaturesFileName=  baseFolder + "/"+ baseFeature[feat] + featureType[feat] +"/"+dimensionReduction+"/"+Utilities.capitalizeFirstLetter(trainingSets[set])+String.format("Seq%03d.csv", vid);
							if(regressorName.contains("SLDA")|| dimensionReduction.contains("SLDA"))
								trainFeaturesFileName=  baseFolder + "/"+ baseFeature[feat] + featureType[feat] +"/"+capitalizedResponse+dimensionReduction+"/"+Utilities.capitalizeFirstLetter(trainingSets[set])+String.format("Seq%03d.csv", vid);
							try{
								//System.out.println(trainFeaturesFileName);
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
							catch(Exception e){
								System.err.println("Unable to load file: "+ trainFeaturesFileName);
								throw e;
							}
						}
					}
					totalSamples+= trainingInstances[0].numInstances();
					int count=0;
					docIds[vidIndex]= new String[trainingInstances[0].numInstances()];
					for(Instance inst: trainingInstances[0]){
						docIds[vidIndex][count]= new String(""+inst.value(0));
						count++;
					}
					for(int feat=0; feat<trainingInstances.length; feat++){
						trainingInstances[feat].deleteAttributeAt(0);
					}
					trainingFeatures[vidIndex]= new double[trainingInstances[0].numInstances()][];
					int totalFeats=0;
					for(int feat=0; feat<trainingInstances.length; feat++){
						//System.out.println("Feat "+(feat+1)+ " Num features: "+trainingInstances[feat].numAttributes());
						//System.out.println("Num Instances: "+trainingInstances[feat].numInstances());
						totalFeats+= trainingInstances[feat].numAttributes();
					}

					for(int i=0; i< trainingInstances[0].numInstances(); i++){
						trainingFeatures[vidIndex][i]= new double[totalFeats];
						count=0;
						for(int type=0; type< trainingInstances.length; type++){
							// if the current instance has NaN then ignore it
							if(trainingInstances[type].get(i).hasMissingValue()){
								ignoredSampleIndices.add(i);
							}
							else{
								for(int feat=0; feat<trainingInstances[type].numAttributes(); feat++){
									trainingFeatures[vidIndex][i][count]= trainingInstances[type].get(i).value(feat);
									// if the transformation is log add a small noise so that there are no zeroes
									if(featureTransformation.equalsIgnoreCase("log")){
										trainingFeatures[vidIndex][i][count]+= 1E-6;
										trainingFeatures[vidIndex][i][count]= Math.log(trainingFeatures[vidIndex][i][count]);
									}
									count++;
								}
							}
						}
					}
					responses[vidIndex]= new double[trainingInstances[0].numInstances()];
					retainedSampleIndices.add(new ArrayList<Integer>());
					count=0;
					// load training responses to an array
					String trainResponsesFileName=  baseFolder+"/"+"responses/"+Utilities.capitalizeFirstLetter(trainingSets[set])+String.format("%s%03d.csv", capitalizedResponse,vid);
					String trainChangesFileName=null;
					String[][] temp= Utilities.readCSVFile(trainResponsesFileName, false);
					String[][] temp1= null;
					if(samplingAlgo.equals("change")){
						trainChangesFileName= baseFolder+"/"+"responses/"+Utilities.capitalizeFirstLetter(trainingSets[set])+String.format("%s%03dChangeIndicators.csv", capitalizedResponse,vid);
						temp1= Utilities.readCSVFile(trainChangesFileName, false);
					}
					for(int i=0; i<temp.length; i++){
						if(!ignoredSampleIndices.contains(i)){
							responses[vidIndex][count]= Double.parseDouble(temp[i][1]);
							if(samplingAlgo.equalsIgnoreCase("change")){
								if(Integer.parseInt(temp1[i][1])==1)
									retainedSampleIndices.get(vidIndex).add(count);
							}
							count++;
						}
					}
					//Utilities.printArray(trainingFeatures[vidIndex]);
					//Utilities.printArray("Vid "+(vidIndex+1)+" Responses: ",responses[vidIndex]);
					vidIndex++;
				}
			}
			if(samplingAlgo.contains("kmeans")){
				double percent= Double.parseDouble(samplingAlgo.split("_")[1].trim());
				int numClusters= (int)(totalSamples*percent/100);
				SamplingResults sr= corrBasedDataSampling(trainingFeatures, responses, numClusters);
				trainingFeatures= sr.observations;
				responses= sr.responses;
			}
			// if the feature transformation is probabilistic temporal smoothing then each probabilistic 
			// feature of video is smoothened and each frame is normalized so that sum of all features is 1
			if(featureTransformation.equalsIgnoreCase("smooth")){
				for(vidIndex=0; vidIndex<trainingFeatures.length; vidIndex++){
					double[][] topics = trainingFeatures[vidIndex];
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
								if(i>1) val = trainingFeatures[vidIndex][i-1][topic];
								else val = 0.0;
							}
							smoothedTopic[i] = val;
						}
						// scale back the smoothed topic values to lie between 0 and 1
						//smoothedTopic = Utilities.scaleData(smoothedTopic, 1, 10);
						for(int i=0; i < xVals.length; i++){
							trainingFeatures[vidIndex][i][topic] = smoothedTopic[i];
						}
					}
					/*for(int i=0; i< trainingFeatures[vidIndex].length; i++){
						double totalSum = 0;
						for(int topic = 1; topic < trainingFeatures[vidIndex][0].length; topic++)
							totalSum += trainingFeatures[vidIndex][i][topic];
						for(int topic = 1; topic < trainingFeatures[vidIndex][0].length; topic++)
							trainingFeatures[vidIndex][i][topic] /= totalSum;
					}*/
				}
			}
			// write all the features and responses to a csv file
			PrintWriter fullTrainingDataCSVFile= new PrintWriter(new File(fullTrainingFeaturesFilePath));
			fullTrainingDataCSVFile.print("DocId,");
			for(int ind=1; ind<=trainingFeatures[0][0].length; ind++ )
				fullTrainingDataCSVFile.print("Feature"+ind+",");
			fullTrainingDataCSVFile.println("Class1");
			if(samplingAlgo.contains("kmeans")){
				for(vidIndex=0; vidIndex<trainingFeatures.length; vidIndex++){
					for(int m=0; m<trainingFeatures[vidIndex].length; m++){
						fullTrainingDataCSVFile.print((vidIndex+1)+",");
						for(int ind=0; ind<trainingFeatures[0][0].length; ind++ )
							fullTrainingDataCSVFile.print(fmt.format(trainingFeatures[vidIndex][m][ind])+",");
						fullTrainingDataCSVFile.println(responses[vidIndex][m]);
					}
				}
			}
			else{
				for(vidIndex=0; vidIndex<trainingFeatures.length; vidIndex++){
					for(int m=0; m<trainingFeatures[vidIndex].length; m++)
						if(retainedSampleIndices.get(vidIndex).isEmpty() || retainedSampleIndices.get(vidIndex).contains(vidIndex)){
							fullTrainingDataCSVFile.print(docIds[vidIndex]+",");
							for(int ind=0; ind<trainingFeatures[0][0].length; ind++ )
								fullTrainingDataCSVFile.print(fmt.format(trainingFeatures[vidIndex][m][ind])+",");
							fullTrainingDataCSVFile.println(responses[vidIndex][m]);
						}
				}
			}
			fullTrainingDataCSVFile.close();
			// do this for each regressor
			String regressionModelFolder= baseFolder+ "/"+ baseFeature[0] + featureType[0] +"/"+response+"/"+dimensionReduction;
			if(combineFeatures || dimensionReduction.equals("MMDGMM")){
				regressionModelFolder= baseFolder+ "/"+ baseFeature[0] + featureType[0]+"/"+response+"/";
				for(int feat=0; feat< baseFeature.length; feat++){
					regressionModelFolder+= baseFeature[feat]+featureType[feat];
				}
				regressionModelFolder+= dimensionReduction;
			}
			if(!new File(regressionModelFolder).exists())
				new File(regressionModelFolder).mkdir();
			String regressionModelPath= null;
			// train a regression model
			if(regressorName.equals("SVR")){
				// unpack the svr parameters
				double cParam= Double.parseDouble(regressorParams[0].trim());
				String kernelParams= regressorParams[1].trim();
				regressionModelPath= regressionModelFolder+"/"+regressorName+"_"+regressorParams[0]+"_"+regressorParams[1]+featureTransformation+featureSelection+trainingDataInfo+samplingAlgo+".model";
				returnString= regressorName+"_"+cParam+"_"+kernelParams+featureTransformation+featureSelection+trainingDataInfo;
				if(!new File(regressionModelPath).exists()){ // if regression model does not exist
					SVMRegression svrModelBuilder= new SVMRegression(cParam,kernelParams,normalizeFeatures);
					svrModelBuilder.setFeatureTransformation(featureTransformation);
					svrModelBuilder.setFeatureSelection(featureSelection);
					String[] tokens= kernelParams.trim().split("_");
					String kernelName= tokens[0];
					Optional<String> gramMatrixFile = Optional.<String>absent();
					if(kernelName.equalsIgnoreCase("gram") && dimensionReduction.contains("LDA")){
						gramMatrixFile = Optional.of(regressionModelFolder + "/" + featureTransformation+featureSelection+trainingDataInfo+samplingAlgo+".gram");
						if(!new File(gramMatrixFile.get()).exists()){
							System.out.println("Creating Sim Matrices");
							double[][][] topicSimMatrices = new double[baseFeature.length][][];
							for(int feat=0; feat<baseFeature.length; feat++){
								String ldaBetaFile = baseFolder + "/"+ baseFeature[feat] + featureType[feat] +"/"+dimensionReduction+"Beta.csv";
								topicSimMatrices[feat] = Utilities.calculateTopicIntercorrelations(ldaBetaFile);
							}
							// create an identity matrix with each feature's similarity matrix as a diagonal element
							double[][] allFeatTopicSimMatrix = Utilities.eye(baseFeature.length*topicSimMatrices[0].length);
							int rowIndex = 0, colIndex = 0;
							for(int feat=0; feat<baseFeature.length; feat++){
								for(int r=0; r<topicSimMatrices[feat].length; r++){
									for(int c=0; c<topicSimMatrices[feat][0].length; c++){
										allFeatTopicSimMatrix[rowIndex+r][colIndex+c] = topicSimMatrices[feat][r][c];
									}
								}
								rowIndex+= topicSimMatrices[feat].length;
								colIndex+= topicSimMatrices[feat][0].length;
							}
							System.out.println("Pooling Training Features");
							// use the training data and the feature similarity matrix to generate gram matrix
							int totalRows = 0;
							for(vidIndex=0; vidIndex<trainingFeatures.length; vidIndex++){
								totalRows += trainingFeatures[vidIndex].length;
							}
							double[][] allTrainingFeatures = new double[totalRows][];
							rowIndex = 0;
							for(vidIndex=0; vidIndex<trainingFeatures.length; vidIndex++){
								for(int m=0; m<trainingFeatures[vidIndex].length; m++){
									allTrainingFeatures[rowIndex] = trainingFeatures[vidIndex][m];
									rowIndex++ ;
								}
							}
							Utilities.sizeOf(allTrainingFeatures);
							Utilities.sizeOf(allFeatTopicSimMatrix);
							double[][] gramMatrix = Utilities.matrixMultiply(allTrainingFeatures, allFeatTopicSimMatrix, false, false);
							Utilities.sizeOf(gramMatrix);
							gramMatrix = Utilities.matrixMultiply(gramMatrix, allTrainingFeatures, false, true);
							Utilities.sizeOf(gramMatrix);

							PrintWriter pw = new PrintWriter(new File(gramMatrixFile.get()));
							pw.println(gramMatrix.length+" "+gramMatrix[0].length);
							for(int r = 0; r < gramMatrix.length; r++){
								for(int c = 0; c < gramMatrix[r].length; c++){
									pw.print(String.format("%.2f",gramMatrix[r][c]));
									if(c < gramMatrix[r].length-1)
										pw.print(" ");
								}
								pw.println();
							}
							pw.close();
						}
						
					}
					svrModelBuilder.trainRegressionModel(regressionModelPath, fullTrainingFeaturesFilePath, gramMatrixFile);
				}
			}
			else if(regressorName.contains("LR")){
				boolean onlineFlag= Boolean.parseBoolean(regressorParams[0].trim());
				boolean positiveCoeffs= Boolean.parseBoolean(regressorParams[2].trim());
				if(onlineFlag){
					double learningRate= Double.parseDouble(regressorParams[1].trim());
					regressionModelPath= regressionModelFolder+"/"+regressorName+"_"+regressorParams[0]+"_"+regressorParams[1]+"_"+regressorParams[2].trim()+featureTransformation+featureSelection+trainingDataInfo+samplingAlgo+".model";
					returnString= regressorName+"_"+regressorParams[0]+"_"+regressorParams[1]+featureTransformation+featureSelection+trainingDataInfo;
					if(!new File(regressionModelPath).exists()){ // if regression model does not exist
						LRegression lrModelBuilder= new LRegression(onlineFlag,learningRate,normalizeFeatures,positiveCoeffs);
						lrModelBuilder.setFeatureTransformation(featureTransformation);
						lrModelBuilder.setFeatureSelection(featureSelection);
						lrModelBuilder.trainRegressionModel(regressionModelPath, fullTrainingFeaturesFilePath);
					}
				}
				else{
					regressionModelPath= regressionModelFolder+"/"+regressorName+"_"+regressorParams[0]+"_"+regressorParams[2].trim()+featureTransformation+featureSelection+trainingDataInfo+samplingAlgo+".model";
					returnString= regressorName+"_"+regressorParams[0]+featureTransformation+featureSelection+trainingDataInfo;
					if(!new File(regressionModelPath).exists()){ // if regression model does not exist
						LRegression lrModelBuilder= new LRegression(onlineFlag,normalizeFeatures,positiveCoeffs);
						lrModelBuilder.setFeatureTransformation(featureTransformation);
						lrModelBuilder.setFeatureSelection(featureSelection);
						lrModelBuilder.trainRegressionModel(regressionModelPath, fullTrainingFeaturesFilePath);
					}
				}

			}
			else if(regressorName.contains("DSR")){
				boolean onlineFlag= Boolean.parseBoolean(regressorParams[0].trim());
				regressionModelPath= regressionModelFolder+"/"+regressorName+"_"+regressorParams[0]+featureTransformation+featureSelection+trainingDataInfo+samplingAlgo+".model";
				returnString= regressorName+"_"+regressorParams[0]+featureTransformation+featureSelection+trainingDataInfo;
				if(!new File(regressionModelPath).exists()){ // if regression model does not exist
					DecisionStumpRegression dsrModelBuilder= new DecisionStumpRegression(onlineFlag);
					dsrModelBuilder.setFeatureTransformation(featureTransformation);
					dsrModelBuilder.setFeatureSelection(featureSelection);
					dsrModelBuilder.trainRegressionModel(regressionModelPath, fullTrainingFeaturesFilePath);
				}
			}
			else if(regressorName.contains("DTR")){
				boolean pruneTree= Boolean.parseBoolean(regressorParams[0].trim());
				regressionModelPath= regressionModelFolder+"/"+regressorName+"_"+regressorParams[0]+featureTransformation+featureSelection+trainingDataInfo+samplingAlgo+".model";
				returnString= regressorName+"_"+regressorParams[0]+featureTransformation+featureSelection+trainingDataInfo;
				if(!new File(regressionModelPath).exists()){ // if regression model does not exist
					DecisionTreeRegression dtrModelBuilder= new DecisionTreeRegression(pruneTree);
					dtrModelBuilder.setFeatureTransformation(featureTransformation);
					dtrModelBuilder.setFeatureSelection(featureSelection);
					dtrModelBuilder.trainRegressionModel(regressionModelPath, fullTrainingFeaturesFilePath);
				}
			}
			else if(regressorName.contains("BAG")){
				int iters= Integer.parseInt(regressorParams[0].trim());
				int bagSize= Integer.parseInt(regressorParams[1].trim());
				regressionModelPath= regressionModelFolder+"/"+regressorName+"_"+regressorParams[0]+"_"+regressorParams[1]+"_"+regressorParams[2]+featureTransformation+featureSelection+trainingDataInfo+samplingAlgo+".model";
				returnString= regressorName+"_"+regressorParams[0]+"_"+regressorParams[1]+"_"+regressorParams[2]+featureTransformation+featureSelection+trainingDataInfo;
				if(!new File(regressionModelPath).exists()){ // if regression model does not exist
					BaggedRegression baggedModelBuilder= new BaggedRegression(iters,bagSize,regressorParams[2]);
					baggedModelBuilder.setFeatureTransformation(featureTransformation);
					baggedModelBuilder.setFeatureSelection(featureSelection);
					baggedModelBuilder.trainRegressionModel(regressionModelPath, fullTrainingFeaturesFilePath);
				}
			}
			else if(regressorName.contains("BOOST")){
				int iters= Integer.parseInt(regressorParams[0].trim());
				double shrinkage= Double.parseDouble(regressorParams[1].trim());
				regressionModelPath= regressionModelFolder+"/"+regressorName+"_"+regressorParams[0]+"_"+regressorParams[1]+"_"+regressorParams[2]+featureTransformation+featureSelection+trainingDataInfo+samplingAlgo+".model";
				returnString= regressorName+"_"+regressorParams[0]+"_"+regressorParams[1]+"_"+regressorParams[2]+featureTransformation+featureSelection+trainingDataInfo;
				if(!new File(regressionModelPath).exists()){ // if regression model does not exist
					BoostingRegression boostedModelBuilder= new BoostingRegression(iters,shrinkage,regressorParams[2]);
					boostedModelBuilder.setFeatureTransformation(featureTransformation);
					boostedModelBuilder.setFeatureSelection(featureSelection);
					boostedModelBuilder.trainRegressionModel(regressionModelPath, fullTrainingFeaturesFilePath);
				}
			}
			// once the model is built, delete the csv file
			new File(fullTrainingFeaturesFilePath).delete();
			//System.out.println("Done building the model: "+returnString);
		}
		catch(Exception e){
			e.printStackTrace();System.exit(1);
		}
		return returnString;
	}

	/**
	 * Selects samples from a set of observations using the correlations between features and labels  
	 * @param observations
	 * video wise observations
	 * @param labels
	 * video wise labels
	 * @return
	 */
	public SamplingResults corrBasedDataSampling(double[][][] observations, double[][] labels, int numClusters)
			throws Exception
	{
		System.out.println("Sampling using KMeans: NumCluster: "+numClusters);
		int numLags= 20;
		double[] tou= new double[numLags];
		for(int i=1; i<=numLags; i++)
			tou[i-1]=i;
		int numVids= observations.length;
		int numFeatures= observations[0][0].length;
		double[][] r= new double[numFeatures][numLags];
		double[] touPs= new double[numLags];
		for(int l=0; l<numLags; l++)
			touPs[l]= 0;
		int totalSamples=0;
		for(int vid=0; vid<numVids; vid++){
			totalSamples+= observations[vid].length;
		}
		for(int feat=0; feat<numFeatures; feat++){
			for(int vid=0; vid<numVids; vid++){
				int numSamples= observations[vid].length;
				for(int l=0; l<numLags; l++){
					double[] featVals= new double[numSamples-l];
					for(int s=l; s<numSamples; s++)
						featVals[s-l]= observations[vid][s][feat];
					double[] labelVals= new double[numSamples-l];
					for(int s=0; s<numSamples-l; s++)
						labelVals[s]= labels[vid][s];
					double val= Utilities.calculateCrossCorrelation(featVals,labelVals);
					if(Double.isNaN(val)){
						val=0;
					}
					r[feat][l]+=val;
					/*if(l==3){
						Utilities.printArray("feats", featVals);
						Utilities.printArray("labels", labelVals);
						System.out.println(r[feat][l]);
					}*/
				}					
			}
			for(int l=0; l<numLags; l++){
				r[feat][l]/=numVids;
			}
			//Utilities.printArray("Feat "+(feat+1)+" r",r[feat]);
		}
		double total=0;
		for(int l=0; l<numLags; l++){
			for(int feat=0; feat<numFeatures; feat++){
				touPs[l]+=r[feat][l];
			}
			touPs[l]= Math.abs(touPs[l]);
			total+= touPs[l];
		}
		for(int l=0; l<numLags; l++){
			touPs[l]/= total;
		}
		Utilities.printArray("touPs ", touPs);
		double[] rhos= new double[numFeatures];
		for(int feat=0; feat<numFeatures; feat++){
			for(int l=0; l<numLags; l++){
				rhos[feat]+= Math.abs(r[feat][l])*touPs[l]; 
			}
		}
		Utilities.printArray("rhos ", rhos);
		// create instances from the observations
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		for(int i=0; i<numFeatures; i++){
			Attribute a= new Attribute("Feature"+(i+1));
			attributes.add(a);
		}
		Instances instances= new Instances("data", attributes, totalSamples);
		double[] allVidLabels= new double[totalSamples];
		int count=0;
		for(int vid=0; vid< numVids; vid++){
			int numSamples= observations[vid].length;
			for(int s=0; s<numSamples; s++){
				Instance inst= new DenseInstance(numFeatures);
				for(int feat=0; feat<numFeatures; feat++){
					inst.setValue(feat, observations[vid][s][feat]);
				}
				instances.add(inst);
				allVidLabels[count]= labels[vid][s];
				count++;
			}
		}
		//System.out.println(instances);
		WeightedCosine wd= new WeightedCosine(instances, -1, rhos);
		SimpleKMeans kmeansModel= new SimpleKMeans();
		kmeansModel.setNumClusters(numClusters);
		kmeansModel.buildClusterer(instances);
		kmeansModel.setDistanceFunction(wd);

		int[] clusterIds= new int[totalSamples];
		HashMap<Integer,ArrayList<Double>> clusterWiseLabels= new HashMap<Integer, ArrayList<Double>>();
		for(int i=0; i<totalSamples; i++){
			clusterIds[i]= kmeansModel.clusterInstance(instances.get(i));
			if(clusterWiseLabels.isEmpty() ){
				ArrayList<Double> temp= new ArrayList<Double>();
				temp.add(allVidLabels[i]);
				clusterWiseLabels.put(clusterIds[i],temp);
			}
			else{ 
				if(clusterWiseLabels.get(clusterIds[i])==null){
					ArrayList<Double> temp= new ArrayList<Double>();
					temp.add(allVidLabels[i]);
					clusterWiseLabels.put(clusterIds[i],temp);
				}
				else{
					ArrayList<Double> temp= clusterWiseLabels.get(clusterIds[i]);
					temp.add(allVidLabels[i]);
					clusterWiseLabels.put(clusterIds[i],temp);
				}
			}
		}
		//System.out.println(clusterWiseLabels);
		int totalSampledObs= clusterWiseLabels.size();
		double[][][] sampledObservations= new double[totalSampledObs][1][];
		double[][] sampledLabels= new double[totalSampledObs][1];
		Instances centroids= kmeansModel.getClusterCentroids();
		count=0;
		for(int i: clusterWiseLabels.keySet()){
			for(double label:clusterWiseLabels.get((i)) ){
				sampledLabels[count][0]+=label;
			}
			sampledLabels[count][0]/= clusterWiseLabels.get(i).size();
			sampledObservations[count][0]= centroids.get(i).toDoubleArray();
			//Utilities.printArray(sampledObservations[count]);
			count++;
		}
		//Utilities.printArray(clusterIds);
		//Utilities.printArray(sampledLabels);		

		SamplingResults results= new SamplingResults(sampledObservations, sampledLabels);
		return results;
	}

	private class SamplingResults{
		double[][][] observations;
		double[][] responses;

		SamplingResults(double[][][] obs, double[][] r){
			observations= obs;
			responses= r;
		}
		public double[][][] getObservations() {
			return observations;
		}
		public double[][] getResponses() {
			return responses;
		}
	}


}
