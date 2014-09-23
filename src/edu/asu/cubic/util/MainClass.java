package edu.asu.cubic.util;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import edu.asu.cubic.classification.ClassificationResults;
import edu.asu.cubic.regression.RegressionResults;

public class MainClass {

	public static void main(String[] args) throws IOException{
		String propertiesFilePath= args[0];
		Properties requiredParameters= new Properties();
		requiredParameters.load(new FileInputStream(propertiesFilePath));
		init(requiredParameters);
	}

	public static void init(Properties requiredParameters){
		try {

			String baseFolder= requiredParameters.getProperty("baseFolder").trim();
			String dataset= requiredParameters.getProperty("dataset").trim();
			String[] tokens= requiredParameters.getProperty("baseFeature").trim().split(";");
			String[] baseFeatures= new String[tokens.length];
			String[] featureTypes= new String[tokens.length];
			for(int i=0; i<tokens.length; i++){
				baseFeatures[i]= tokens[i].split(",")[0];
				if(tokens[i].split(",").length >= 2)
					featureTypes[i]= tokens[i].split(",")[1];
				else
					featureTypes[i]= "";
			}
			String dimensionReduction= requiredParameters.getProperty("dimensionReduction").trim();
			String samplingAlgo= requiredParameters.getProperty("sampling").trim();
			boolean writePredictions= Boolean.parseBoolean(requiredParameters.getProperty("writePredictions").trim());
			// flag that indicates if we have to combine predictions from different features
			boolean combinePredictions= Boolean.parseBoolean(requiredParameters.getProperty("combinePredictions").trim());
			boolean combineFeatures= Boolean.parseBoolean(requiredParameters.getProperty("combineFeatures").trim());
			boolean scaleResponses= Boolean.parseBoolean(requiredParameters.getProperty("scaleResponses").trim());
			String problem= requiredParameters.getProperty("problem").trim();
			String[] regressors= requiredParameters.getProperty("regressors").trim().split(";");
			String[] classifiers= requiredParameters.getProperty("classifiers").trim().split(";");
			String response= requiredParameters.getProperty("response").trim();
			String approach= requiredParameters.getProperty("approach").trim();
			String[] phases= requiredParameters.getProperty("phase").trim().split(";"); 
			String trainingSetsString= requiredParameters.getProperty("trainingSets").trim();
			int totalTrainingVids=0;
			tokens= trainingSetsString.split(";");
			String[] trainingSets= new String[tokens.length];
			int[][] trainingSetVideos= new int[tokens.length][2];
			for(int i=0; i<tokens.length; i++)
			{
				trainingSets[i]= tokens[i].split(",")[0];
				trainingSetVideos[i][0]= Integer.parseInt(tokens[i].split(",")[1]);// start video
				trainingSetVideos[i][1]= Integer.parseInt(tokens[i].split(",")[2]);// end video
				totalTrainingVids+= trainingSetVideos[i][1]-trainingSetVideos[i][0]+1;
			}
			String testingSetsString=  requiredParameters.getProperty("testingSets").trim();
			/*tokens= testingSetsString.split(";");
			int[]testingSetVideos= new int[2];
			testingSetVideos[0]= Integer.parseInt(tokens[0].split(",")[1]);// start video
			testingSetVideos[1]= Integer.parseInt(tokens[0].split(",")[2]);// end video */
			for(int p=0; p<phases.length; p++){
				System.out.println("Phase: "+phases[p]);
				if(problem.equalsIgnoreCase("regression") && phases[p].equalsIgnoreCase("training") && approach.equalsIgnoreCase("instance")){
					instanceBasedRegressionTraining(requiredParameters, regressors, trainingSetsString, baseFeatures, featureTypes, combineFeatures);
				}
				else if(problem.equalsIgnoreCase("regression") && phases[p].equalsIgnoreCase("training") && approach.equalsIgnoreCase("ensemble")){
					ensembleBasedRegressionTraining(requiredParameters, regressors, trainingSetsString, baseFeatures, featureTypes);
				}
				else if(problem.equalsIgnoreCase("regression") && phases[p].equalsIgnoreCase("testing") && approach.equalsIgnoreCase("instance")){
					instanceBasedRegressionTesting(requiredParameters, baseFolder, response, baseFeatures, featureTypes, dimensionReduction, regressors, trainingSetsString, testingSetsString, combinePredictions, writePredictions, scaleResponses,false, combineFeatures,dataset, samplingAlgo);
				}
				else if(problem.equalsIgnoreCase("regression") && phases[p].equalsIgnoreCase("testing") && approach.equalsIgnoreCase("ensemble")){
					ensembleBasedRegressionTesting(requiredParameters, baseFolder, response, baseFeatures, featureTypes, dimensionReduction, regressors, trainingSetsString, testingSetsString, combinePredictions, writePredictions, scaleResponses, false, dataset);
				}
				else if(problem.equalsIgnoreCase("classification") && phases[p].equalsIgnoreCase("training") && approach.equalsIgnoreCase("instance")){
					instanceBasedClassificationTraining(requiredParameters, classifiers, trainingSetsString, baseFeatures, featureTypes, combineFeatures);
				}
				else if(problem.equalsIgnoreCase("classification") && phases[p].equalsIgnoreCase("testing") && approach.equalsIgnoreCase("instance")){
					instanceBasedClassificationTesting(requiredParameters, baseFolder, response, baseFeatures, featureTypes, dimensionReduction, classifiers, trainingSetsString, testingSetsString, combinePredictions, writePredictions, scaleResponses,false, combineFeatures,dataset, samplingAlgo);
				}
				else if(phases[p].toLowerCase().contains("crossvalidation") ){
					int numFolds = Integer.parseInt(phases[p].toLowerCase().split("_")[1]);
					// for each fold run the training and testing phases
					// divide entire training data into folds
					String[] videoCategories= new String[totalTrainingVids];
					int[] videoNumbers= new int[totalTrainingVids];
					int count=0;
					for(int set=0; set<trainingSets.length; set++){
						for(int vid=trainingSetVideos[set][0]; vid<=trainingSetVideos[set][1]; vid++){
							videoCategories[count]= trainingSets[set];
							videoNumbers[count]= vid;
							count++;
						}
					}
					String[][] testVideoCategoriesPerFold= new String[numFolds][];
					int[][] testVideoNumbersPerFold= new int[numFolds][];
					String[][] trainVideoCategoriesPerFold= new String[numFolds][];
					int[][] trainVideoNumbersPerFold= new int[numFolds][];
					int currCount=0;
					double[][] foldWiseResults= new double[numFolds][2];
					// create pool of threads that will each build a regression model
					ExecutorService service = Executors.newFixedThreadPool(numFolds);
					// the result returned by each thread is stored in the following variable
					List<Future<String>> futures = new ArrayList<Future<String>>();
					for(int fold=0; fold<numFolds; fold++){
						System.out.println("Fold "+(fold+1));
						int start= currCount; 
						int end= start+(totalTrainingVids/numFolds);
						if(fold==numFolds-1)
							end= totalTrainingVids;
						testVideoCategoriesPerFold[fold]= new String[end-start];
						testVideoNumbersPerFold[fold]= new int[end-start];
						// populate test video numbers
						for(int i=start; i<end; i++){
							testVideoCategoriesPerFold[fold][i-start]= videoCategories[currCount];
							testVideoNumbersPerFold[fold][i-start]= videoNumbers[currCount];
							currCount++;
						}
						//Utilities.printArray(testVideoCategoriesPerFold[fold]);
						//Utilities.printArray(testVideoNumbersPerFold[fold]);
						trainVideoCategoriesPerFold[fold]= new String[totalTrainingVids-(end-start)];
						trainVideoNumbersPerFold[fold]= new int[totalTrainingVids-(end-start)];
						// populate train video numbers
						count= 0;
						for(int i=0; i<totalTrainingVids; i++){
							if(i<start || i>=end){
								trainVideoCategoriesPerFold[fold][count]= videoCategories[i];
								trainVideoNumbersPerFold[fold][count]= videoNumbers[i];
								count++;
							}
						}
						//Utilities.printArray(trainVideoCategoriesPerFold[fold]);
						//Utilities.printArray(trainVideoNumbersPerFold[fold]);
						// append the training and testing video sets to a single string in the format category,startVid,endVid;category,startVide,endVid; etc
						trainingSetsString="";
						String prevCat= trainVideoCategoriesPerFold[fold][0];
						int prevVid= trainVideoNumbersPerFold[fold][0];
						start=0;
						for(int i=0; i<trainVideoCategoriesPerFold[fold].length; i++){
							String currCat= trainVideoCategoriesPerFold[fold][i];
							int currVid= trainVideoNumbersPerFold[fold][i];
							if(!currCat.equalsIgnoreCase(prevCat) || currVid-prevVid>1 ){
								trainingSetsString+=prevCat+","+trainVideoNumbersPerFold[fold][start]+","+trainVideoNumbersPerFold[fold][i-1]+";";
								start= i;
							}
							if(i==trainVideoCategoriesPerFold[fold].length-1){
								trainingSetsString+=prevCat+","+trainVideoNumbersPerFold[fold][start]+","+trainVideoNumbersPerFold[fold][i]+";";
								break;
							}
							prevCat= currCat;
							prevVid= currVid;
						}
						System.out.println("Training "+trainingSetsString);
						testingSetsString="";
						prevCat= testVideoCategoriesPerFold[fold][0];
						prevVid= testVideoNumbersPerFold[fold][0];
						start=0;
						for(int i=0; i<testVideoCategoriesPerFold[fold].length; i++){
							String currCat= testVideoCategoriesPerFold[fold][i];
							int currVid= testVideoNumbersPerFold[fold][i];
							if(!currCat.equalsIgnoreCase(prevCat) || currVid-prevVid>1 ){
								testingSetsString+=prevCat+","+testVideoNumbersPerFold[fold][start]+","+testVideoNumbersPerFold[fold][i-1]+";";
								start= i;
							}
							if(i==testVideoCategoriesPerFold[fold].length-1){
								testingSetsString+=prevCat+","+testVideoNumbersPerFold[fold][start]+","+testVideoNumbersPerFold[fold][i]+";";
								break;
							}
							prevCat= currCat;
							prevVid= currVid;
						}
						System.out.println("Testing "+testingSetsString);
						// Once the training and testing data for this fold are extracted, run the training and testing phases
						Callable<String> mapper= (new MainClass()).new CVFoldHandler(requiredParameters, baseFolder, response, approach, baseFeatures, featureTypes, dimensionReduction, regressors, trainingSetsString, testingSetsString, combinePredictions, scaleResponses,writePredictions, true,combineFeatures, dataset, samplingAlgo);
						// adding the result returned by model to the list
						futures.add(service.submit(mapper));
					}
					service.shutdown();
					try {
						service.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
					} catch (InterruptedException e) {e.printStackTrace();return;}
					/*// write the mean results from all folds to the file
					String regressionModelFolder= baseFolder+"/"+Utilities.capitalizeFirstLetter(response)+"Analysis/"+baseFeatures[0]+featureTypes[0]+dimensionReduction;
					PrintWriter resultsFile= new PrintWriter(new FileWriter(regressionModelFolder + "/Results.txt",true));*/
				}
				else if(phases[p].equals("analysis"))
					ResultsAnalyser.analyseASCCrossValidationResults(requiredParameters);
				else if(phases[p].equals("dimreduction") && dimensionReduction.equalsIgnoreCase("pca"))
					PCABuilder.extractPCAFeatures(requiredParameters);
				else if(phases[p].equals("dimreduction") && dimensionReduction.contains("SLDAVB"))
					SLDAVBBuilder.extractSLDAFeatures(requiredParameters);
				else if(phases[p].equals("dimreduction") && dimensionReduction.contains("SLDA"))
					SLDABuilder.extractSLDAFeatures(requiredParameters);
				else if(phases[p].equals("dimreduction") && dimensionReduction.contains("LDAVB"))
					LDAVBBuilder.extractLDAFeatures(requiredParameters);
				else if(phases[p].equals("dimreduction") && dimensionReduction.contains("LDA"))
					LDABuilder.extractLDAFeatures(requiredParameters);
				else if(phases[p].equals("dimreduction") && dimensionReduction.contains("MMDGMM"))
					MMDGMMBuilder.extractMMDGMMFeatures(requiredParameters);
				else if(phases[p].equals("dimreduction") && dimensionReduction.contains("DGMM"))
					DGMMBuilder.extractDGMMFeatures(requiredParameters);
				else if(phases[p].equals("dimreduction") && dimensionReduction.contains("GMMVB"))
					GMMVBBuilder.extractGMMFeatures(requiredParameters);
				else if(phases[p].equals("dimreduction") && dimensionReduction.contains("GMM"))
					GMMBuilder.extractGMMFeatures(requiredParameters);

			}
		}
		catch(Exception e){e.printStackTrace();System.exit(1);}
	}

	public static void instanceBasedRegressionTraining(Properties requiredParameters, String[] regressors,String trainingSetsString,String[] baseFeatures, String[] featureTypes, boolean combineFeatures){

		// Obtain the regressor parameters and algorithms from properties file
		// and spawn threads each for different regression models
		int threads = regressors.length;//Runtime.getRuntime().availableProcessors();
		// create pool of threads that will each build a regression model
		ExecutorService service = Executors.newFixedThreadPool(threads);
		// the result returned by each thread is stored in the following variable
		List<Future<String>> futures = new ArrayList<Future<String>>();
		if(!combineFeatures){
			for(int f=0; f< baseFeatures.length; f++){
				// package the regression parameters and spawn one thread per model
				for(int m=0; m< regressors.length; m++){
					String[] tokens= regressors[m].split(",");
					String regressorName= tokens[0];
					String[] regressorParams= null;
					regressorParams= new String[tokens.length-1];
					for(int i=0; i<tokens.length-1; i++)
						regressorParams[i]= tokens[i+1];
					// spawning the thread
					Callable<String> mapper= new RegressorBuilder(requiredParameters, regressorName, regressorParams,trainingSetsString,baseFeatures[f],featureTypes[f]);
					// adding the result returned by model to the list
					futures.add(service.submit(mapper));
					/*RegressorBuilder builder= new RegressorBuilder(propertiesFilePath[0], regressorName, regressorParams,requiredParameters.getProperty("trainingSets").trim());
				builder.call();*/
				}
			}
			/*service.shutdown();
			try {
				service.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
			} catch (InterruptedException e) {e.printStackTrace();return;}*/
		}
		else{ // if the features have to be combined it is a single thread
			for(int m=0; m< regressors.length; m++){
				String[] tokens= regressors[m].split(",");
				String regressorName= tokens[0];
				String[] regressorParams= null;

				regressorParams= new String[tokens.length-1];
				for(int i=0; i<tokens.length-1; i++)
					regressorParams[i]= tokens[i+1];
				// spawning the thread
				Callable<String> mapper= new RegressorBuilder(requiredParameters, regressorName, regressorParams,trainingSetsString,baseFeatures,featureTypes);
				// adding the result returned by model to the list
				futures.add(service.submit(mapper));
				/*RegressorBuilder builder= new RegressorBuilder(propertiesFilePath[0], regressorName, regressorParams,requiredParameters.getProperty("trainingSets").trim());
				builder.call();*/
			}
		}
		service.shutdown();
		try {
			service.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		} catch (InterruptedException e) {e.printStackTrace();return;}
	}

	public static void instanceBasedClassificationTraining(Properties requiredParameters, String[] classifiers,String trainingSetsString,String[] baseFeatures, String[] featureTypes, boolean combineFeatures){

		int threads = classifiers.length;//Runtime.getRuntime().availableProcessors();
		// create pool of threads that will each build a model
		ExecutorService service = Executors.newFixedThreadPool(threads);
		// the result returned by each thread is stored in the following variable
		List<Future<String>> futures = new ArrayList<Future<String>>();
		if(!combineFeatures){
			for(int f=0; f< baseFeatures.length; f++){
				for(int m=0; m< classifiers.length; m++){
					String[] tokens= classifiers[m].split(",");
					String classifierName= tokens[0];
					String[] classifierParams= null;
					classifierParams= new String[tokens.length-1];
					for(int i=0; i<tokens.length-1; i++)
						classifierParams[i]= tokens[i+1];
					// spawning the thread
					Callable<String> mapper= new ClassifierBuilder(requiredParameters, classifierName, classifierParams,trainingSetsString,baseFeatures[f],featureTypes[f]);
					// adding the result returned by model to the list
					futures.add(service.submit(mapper));
				}
			}
		}
		else{ // if the features have to be combined it is a single thread
			for(int m=0; m< classifiers.length; m++){
				String[] tokens= classifiers[m].split(",");
				String classifierName= tokens[0];
				String[] classifierParams= null;

				classifierParams= new String[tokens.length-1];
				for(int i=0; i<tokens.length-1; i++)
					classifierParams[i]= tokens[i+1];
				// spawning the thread
				Callable<String> mapper= new ClassifierBuilder(requiredParameters, classifierName, classifierParams,trainingSetsString,baseFeatures,featureTypes);
				// adding the result returned by model to the list
				futures.add(service.submit(mapper));
			}
		}
		service.shutdown();
		try {
			service.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		} catch (InterruptedException e) {e.printStackTrace();return;}
	}


	public static void ensembleBasedRegressionTraining(Properties requiredParameters, String[] regressors,String trainingSetsString,String[] baseFeatures, String[] featureTypes){
		// Obtain the regressor parameters and algorithms from properties file
		// and spawn threads each for different regression models
		String[] tokens= trainingSetsString.split(";");
		String[] trainingSets= new String[tokens.length];
		int[][] trainingSetVideos= new int[tokens.length][2];
		for(int i=0; i<tokens.length; i++)
		{
			trainingSets[i]= tokens[i].split(",")[0];
			trainingSetVideos[i][0]= Integer.parseInt(tokens[i].split(",")[1]);// start video
			trainingSetVideos[i][1]= Integer.parseInt(tokens[i].split(",")[2]);// end video
		}
		for(int f=0; f< baseFeatures.length; f++){
			// Do it for each regressor
			for(int m=0; m< regressors.length; m++){
				// Spawn a thread that builds a regression model on each training video
				for(int set=0; set< trainingSets.length; set++){
					// create pool of threads that will each build a regression model
					ExecutorService service = Executors.newFixedThreadPool(trainingSetVideos[set][1]-trainingSetVideos[set][0]+1);
					// the result returned by each thread is stored in the following variable
					List<Future<String>> futures = new ArrayList<Future<String>>();
					for(int vid=trainingSetVideos[set][0]; vid<=trainingSetVideos[set][1]; vid++){
						tokens= regressors[m].split(",");
						String regressorName= tokens[0];
						String[] regressorParams= null;
						// package the regressor parameters into array
						regressorParams= new String[tokens.length-1];
						for(int i=0; i<tokens.length-1; i++)
							regressorParams[i]= tokens[i+1];
						String tSet= trainingSets[set]+","+vid+","+vid;
						/*// spawning the thread
						Callable<String> mapper= new RegressorBuilder(propertiesFilePath[0], regressorName, regressorParams, tSet);
						// adding the result returned by model to the list
						futures.add(service.submit(mapper));*/
						RegressorBuilder builder= new RegressorBuilder(requiredParameters, regressorName, regressorParams,tSet,baseFeatures[f],featureTypes[f]);
						builder.call();
					}
					service.shutdown();
					try {
						service.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
					} catch (InterruptedException e) {e.printStackTrace();service.shutdown();return;}

				}
			}
		}
	}

	public static double[] instanceBasedRegressionTesting(Properties requiredParameters, String baseFolder, String response, String[] baseFeatures, String[] featureTypes, String dimensionReduction, String[] regressors,String trainingSetsString,String testingSetsString, boolean combinePredictions, boolean writePredictions, 
			boolean scaleResponses,boolean crossvalidation, boolean combineFeatures, String dataset, String samplingAlgo){
		double[] finalMeanResults= new double[2];
		try{

			String[] tokens= testingSetsString.split(";");
			String[] testingSets= new String[tokens.length];
			int[][] testingSetVideos= new int[tokens.length][2];
			int totalTestVideos=0;
			String testingDataInfo="";
			for(int i=0; i<tokens.length; i++)
			{
				testingSets[i]= tokens[i].split(",")[0];
				testingSetVideos[i][0]= Integer.parseInt(tokens[i].split(",")[1]);// start video
				testingSetVideos[i][1]= Integer.parseInt(tokens[i].split(",")[2]);// end video
				totalTestVideos+= testingSetVideos[i][1]-testingSetVideos[i][0]+1;
				testingDataInfo+=testingSets[i]+"_"+testingSetVideos[i][0]+"_"+testingSetVideos[i][1];
			}
			String[] allTestVideoCategories= new String[totalTestVideos];
			int[] allTestVideoNumbers= new int[totalTestVideos];
			int count=0;
			for(int set=0; set<testingSets.length; set++){
				for(int vid=testingSetVideos[set][0]; vid<=testingSetVideos[set][1]; vid++){
					allTestVideoCategories[count]= testingSets[set];
					allTestVideoNumbers[count]= vid;
					count++;
				}
			}
			String capitalizedResponse= Utilities.capitalizeFirstLetter(response);
			String[] regressorsDetails= new String[regressors.length];
			if(!combineFeatures){
				double[][][][] predictionsFromVariousFeatures= new double[totalTestVideos][baseFeatures.length][regressors.length][];
				ArrayList<ArrayList<ArrayList<Integer>>> ignoredIndices= new ArrayList<ArrayList<ArrayList<Integer>>>();
				for(int f=0; f<baseFeatures.length; f++){
					int threads = regressors.length;//Runtime.getRuntime().availableProcessors();
					// create pool of threads that will each build a regression model
					ExecutorService service = Executors.newFixedThreadPool(threads);
					// the result returned by each thread is stored in the following variable
					List<Future<RegressionResults>> futures = new ArrayList<Future<RegressionResults>>();
					// ****** Map
					// package the regression parameters and spawn one thread per model
					for(int m=0; m< regressors.length; m++){
						tokens= regressors[m].split(",");
						String regressorName= tokens[0];
						String[] regressorParams= null;
						// package the regressor parameters into array
						regressorParams= new String[tokens.length-1];
						for(int i=0; i<tokens.length-1; i++)
							regressorParams[i]= tokens[i+1];
						// spawning the thread
						Callable<RegressionResults> mapper= new RegressorEvaluator(requiredParameters, regressorName, regressorParams,trainingSetsString,testingSetsString,baseFeatures[f],featureTypes[f],writePredictions);
						// adding the result returned by model to the list
						futures.add(service.submit(mapper));
						/*RegressorEvaluator evaluator= new RegressorEvaluator(propertiesFilePath[0], regressorName, regressorParams,requiredParameters.getProperty("trainingSets").trim());
				evaluator.call();*/
					}
					service.shutdown();
					try {
						service.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
					} catch (InterruptedException e) {e.printStackTrace();service.shutdown();System.exit(1);}

					// ****** Reduce
					// extract the results from each regressor and write them to a file
					// the format of the results is:
					// regressor_details,test_set,#oftestvideos,response,bagged(ba)/notbagged(nba),baggingfunction,crosscorrelation,rmserror
					String regressionModelFolder= baseFolder+"/"+baseFeatures[f]+featureTypes[f]+"/"+response+"/"+dimensionReduction;
					String resultsFileName= "Results.txt";
					if(crossvalidation)
						resultsFileName= "CrossValidationResults.txt";
					PrintWriter resultsFile= new PrintWriter(new FileWriter(regressionModelFolder + "/"+resultsFileName,true));
					int m=0;
					for(Future<RegressionResults> result: futures){
						String regressorDetails= result.get().getRegressorDetails();
						regressorsDetails[m]= regressorDetails;
						double[][] predictedValues= result.get().getPredictedLabels();
						double[] crossCorrelations= result.get().getCrossCorrelations();
						double[] rmsErrors= result.get().getRmsErrors();
						double[] rSquareVals = result.get().getrSquares();
						double[] meanAbsErrors = result.get().getMeanAbsoluteErrors();
						ignoredIndices.add(result.get().getIgnoredIndices());
						if(combinePredictions){
							for(int vid=0; vid<totalTestVideos; vid++ ){
								predictionsFromVariousFeatures[vid][f][m]= predictedValues[vid];
							}
						}
						Utilities.printArray(" ", crossCorrelations);
						Utilities.printArray("", rmsErrors);
						String resultsString= "";
						resultsString+=regressorDetails+",";
						resultsString+= testingDataInfo+","+totalTestVideos+","+capitalizedResponse+",inst,"+samplingAlgo+",";
						double meanCorrelation=0, meanRMSError=0, weightedMeanCorrelation=0, meanAbsError=0, meanRSquare=0;
						int totalPredictions=0;
						for(int vid=0; vid<totalTestVideos; vid++ ){
							weightedMeanCorrelation+= crossCorrelations[vid]*predictedValues[vid].length;
							meanCorrelation+= crossCorrelations[vid];
							meanRMSError+= rmsErrors[vid];
							meanAbsError+= meanAbsErrors[vid];
							meanRSquare+= rSquareVals[vid];
							totalPredictions+= predictedValues[vid].length;
						}
						weightedMeanCorrelation= Math.abs(weightedMeanCorrelation/totalPredictions);
						meanCorrelation= Math.abs(meanCorrelation/totalTestVideos);
						meanRMSError= Math.abs(meanRMSError/totalTestVideos);
						resultsString+= String.format("%.3f,%.3f,%.3f,%.3f,%.3f,",weightedMeanCorrelation,meanCorrelation,meanRMSError,meanAbsError,meanRSquare);
						finalMeanResults[0]= meanCorrelation; finalMeanResults[1]= meanRMSError;
						for(int vid=0; vid<totalTestVideos; vid++ ){
							resultsString+= String.format("%.2f",crossCorrelations[vid]);
							if(vid!=totalTestVideos-1)
								resultsString+=",";
							else
								resultsString+="\n";
						}
						// write to file
						resultsFile.print(resultsString);
						m++;
					}
					resultsFile.close();
					// delete all the training models if it is cross validation
					if(crossvalidation){
						tokens= trainingSetsString.split(";");
						String[] trainingSets= new String[tokens.length];
						int[][] trainingSetVideos= new int[tokens.length][2];
						for(int i=0; i<tokens.length; i++)
						{
							trainingSets[i]= tokens[i].split(",")[0];
							trainingSetVideos[i][0]= Integer.parseInt(tokens[i].split(",")[1]);// start video
							trainingSetVideos[i][1]= Integer.parseInt(tokens[i].split(",")[2]);// end video
						}
						String regressorModelPath="";
						for(m=0; m<regressors.length; m++){
							regressorModelPath+= regressionModelFolder+"/"+regressors[m].replace(",", "_");
							for(int set=0; set< trainingSets.length; set++){
								regressorModelPath+= trainingSets[set]+"_"+trainingSetVideos[set][0]+"_"+trainingSetVideos[set][1];
							}
							regressorModelPath+=".model";
							System.out.println("Deleting the model file "+regressorModelPath);
							new File(regressorModelPath).delete();
						}
					}
				}
				// if you want to combine the predictions from different features then do the following
				if(combinePredictions){
					String regressionModelFolder= baseFolder+"/"+capitalizedResponse+"Analysis/";
					String resultsFileName= "CombinedResults.txt";
					if(crossvalidation)
						resultsFileName= "CrossValidationCombinedResults.txt";
					PrintWriter resultsFile= new PrintWriter(new FileWriter(regressionModelFolder + "/"+resultsFileName,true));
					for(int m=0; m<regressors.length; m++){
						double weightedMeanCorrelation=0;
						int totalPredictions=0;
						String resultsString= "";
						double[] crossCorrelations= new double[totalTestVideos];
						double[] rmsErrors= new double[totalTestVideos];
						resultsString+=regressorsDetails[m]+",";
						for(int f=0; f<baseFeatures.length; f++){
							resultsString+=baseFeatures[f]+featureTypes[f]+dimensionReduction;
							if(f!=baseFeatures.length-1)
								resultsString+="_";
							else
								resultsString+=",";
						}
						for(int vid=0; vid<totalTestVideos; vid++){
							double[] meanPredictedLabels= new double[predictionsFromVariousFeatures[vid][0][m].length] ;
							for(int t=0; t<meanPredictedLabels.length; t++){
								double[] currTimePredictions= new double[baseFeatures.length];
								for(int f=0; f<baseFeatures.length; f++){
									currTimePredictions[f]= predictionsFromVariousFeatures[vid][f][m][t];
								}
								meanPredictedLabels[t]= Utilities.mean(currTimePredictions);
							}
							//Utilities.printArray(" ", meanPredictedLabels);
							// find correlation with the correct labels
							String labelsFolder=  baseFolder+"/labels";
							String correctLabelsFilePath= labelsFolder+"/"+allTestVideoCategories[vid]+"/"+allTestVideoCategories[vid]+String.format("%03d",allTestVideoNumbers[vid])+"_"+response+".csv";
							if(dataset.equalsIgnoreCase("avec2012")){
								labelsFolder=  baseFolder+"/"+"responses";
								correctLabelsFilePath= labelsFolder+"/"+Utilities.capitalizeFirstLetter(allTestVideoCategories[vid])+String.format("%s%03d.csv", capitalizedResponse,allTestVideoNumbers[vid]);
							}
							String[][] temp= Utilities.readCSVFile(correctLabelsFilePath, false);
							double[] correctLabels= new double[temp.length-ignoredIndices.get(vid).size()];
							String[] docIds= new String[temp.length-ignoredIndices.get(vid).size()];
							int i=0;count=0;
							for(; i<temp.length; i++){
								if(!ignoredIndices.get(vid).contains(i)){
									correctLabels[count]= Double.parseDouble(temp[i][0]);
									if(temp[i].length==2){
										if(dataset.equalsIgnoreCase("avec2012"))
											docIds[count]= temp[i][0];
										else
											docIds[count]= temp[i][1];
									}
									else
										docIds[count]= ""+(i+1);
									count++;
								}
							}
							/*if(scaleResponses)
							correctLabels= Utilities.scaleData(correctLabels, 1, 100);*/
							crossCorrelations[vid]= Utilities.calculateCrossCorrelation(correctLabels, meanPredictedLabels);;
							rmsErrors[vid]= Utilities.calculateRMSError(correctLabels, meanPredictedLabels);;
							// write the predictions to file
							if(writePredictions){
								String tempStr="";
								for(int f=0; f<baseFeatures.length; f++){
									tempStr+=baseFeatures[f]+featureTypes[f];
									if(f!=baseFeatures.length-1)
										tempStr+="_";
								}
								String predictionsFolder= baseFolder+"/FinalResults/"+tempStr;
								if(!new File(predictionsFolder).exists())
									new File(predictionsFolder).mkdir();
								predictionsFolder+= "/"+regressorsDetails[m];
								if(!new File(predictionsFolder).exists())
									new File(predictionsFolder).mkdir();
								predictionsFolder+= "/"+allTestVideoCategories[vid];
								if(!new File(predictionsFolder).exists())
									new File(predictionsFolder).mkdir();
								predictionsFolder+= "/Instance";
								if(!new File(predictionsFolder).exists())
									new File(predictionsFolder).mkdir();
								// Load the original video ids from file
								String videoNamesFilename= baseFolder+"/"+allTestVideoCategories[vid]+"VideoNames.txt";
								String videoName= Utilities.readCSVFile(videoNamesFilename, false)[allTestVideoNumbers[vid]-1][0];
								String predictionsFilePath= predictionsFolder+"/"+videoName+"-"+response.toUpperCase()+".csv";
								//String predictionsFilePath= predictionsFolder+"/"+allTestVideoCategories[vid]+String.format("%03d",allTestVideoNumbers[vid])+"_"+response+".csv";
								PrintWriter predictionsFile= new PrintWriter(predictionsFilePath);
								for(i=0; i<docIds.length; i++)
									predictionsFile.println(meanPredictedLabels[i]+","+docIds[i]);
								predictionsFile.close();
								// Write the predictions sampled at every 60 secs
								predictionsFilePath= predictionsFolder+"/"+Utilities.capitalizeFirstLetter(allTestVideoCategories[vid])+capitalizedResponse+String.format("%03d",allTestVideoNumbers[vid])+".csv";
								predictionsFile= new PrintWriter(predictionsFilePath);
								for(i=0; i<docIds.length; i++){
									if(i%60==0)
										predictionsFile.println(meanPredictedLabels[i]+","+docIds[i]);
								}
								predictionsFile.close();
								// if there is a NaN in any predictions then abort the method
								boolean containsNaN= false;
								for(i=0; i<meanPredictedLabels.length; i++){
									if(Double.isNaN(meanPredictedLabels[i])|| Double.isInfinite(meanPredictedLabels[i])){
										containsNaN= true;
										break;
									}
								}
								if(containsNaN){
									System.err.println("Found NaN in test video predictions "+allTestVideoCategories[vid]+" "+ videoName);
									System.exit(271);
								}
							}

							weightedMeanCorrelation+= crossCorrelations[vid]*meanPredictedLabels.length;
							totalPredictions+= meanPredictedLabels.length;
						}
						Utilities.printArray(" ", crossCorrelations);
						weightedMeanCorrelation= Math.abs(weightedMeanCorrelation/totalPredictions);
						resultsString+= testingDataInfo+","+totalTestVideos+","+capitalizedResponse+",inst,"+samplingAlgo+",";
						resultsString+= weightedMeanCorrelation+","+Math.abs(Utilities.mean(crossCorrelations))+","+Math.abs(Utilities.mean(rmsErrors))+",";
						finalMeanResults[0]= Utilities.mean(crossCorrelations); finalMeanResults[1]= Utilities.mean(rmsErrors);
						for(int vid=0; vid<totalTestVideos; vid++ ){
							resultsString+= crossCorrelations[vid];
							if(vid!=totalTestVideos-1)
								resultsString+=",";
							else
								resultsString+="\n";
						}
						resultsFile.print(resultsString);
					}
					resultsFile.close();
				}
			}
			else{ // if features have to be combined
				int threads = regressors.length;//Runtime.getRuntime().availableProcessors();
				// create pool of threads that will each build a regression model
				ExecutorService service = Executors.newFixedThreadPool(threads);
				// the result returned by each thread is stored in the following variable
				List<Future<RegressionResults>> futures = new ArrayList<Future<RegressionResults>>();
				// ****** Map
				// package the regression parameters and spawn one thread per model
				for(int m=0; m< regressors.length; m++){
					tokens= regressors[m].split(",");
					String regressorName= tokens[0];
					String[] regressorParams= null;
					// package the regressor parameters into array
					regressorParams= new String[tokens.length-1];
					for(int i=0; i<tokens.length-1; i++)
						regressorParams[i]= tokens[i+1];
					// spawning the thread
					Callable<RegressionResults> mapper= new RegressorEvaluator(requiredParameters, regressorName, regressorParams,trainingSetsString,testingSetsString,baseFeatures,featureTypes,writePredictions);
					// adding the result returned by model to the list
					futures.add(service.submit(mapper));
				}
				service.shutdown();
				try {
					service.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
				} catch (InterruptedException e) {e.printStackTrace();service.shutdown();System.exit(1);}

				// ****** Reduce
				// extract the results from each regressor and write them to a file
				// the format of the results is:
				// regressor_details,test_set,#oftestvideos,response,bagged(ba)/notbagged(nba),baggingfunction,crosscorrelation,rmserror
				String regressionModelFolder= baseFolder+"/";
				for(int f=0; f<baseFeatures.length; f++)
					regressionModelFolder+= baseFeatures[f]+featureTypes[f];
				regressionModelFolder+= "/"+response+"/"+dimensionReduction;
				String resultsFileName= "Results.txt";
				if(crossvalidation)
					resultsFileName= "CrossValidationResults.txt";
				PrintWriter resultsFile= new PrintWriter(new FileWriter(regressionModelFolder + "/"+resultsFileName,true));
				int m=0;
				for(Future<RegressionResults> result: futures){
					String regressorDetails= result.get().getRegressorDetails();
					regressorsDetails[m]= regressorDetails;
					double[] crossCorrelations= result.get().getCrossCorrelations();
					double[] rmsErrors= result.get().getRmsErrors();
					Utilities.printArray(" ", crossCorrelations);
					Utilities.printArray("", rmsErrors);
					String resultsString= "";
					resultsString+=regressorDetails+",";
					resultsString+= testingDataInfo+","+totalTestVideos+","+capitalizedResponse+",inst,"+samplingAlgo+",";
					double meanCorrelation=0, meanRMSError=0,weightedMeanCorrelation=0;
					int totalPredictions=0;
					for(int vid=0; vid<totalTestVideos; vid++ ){
						weightedMeanCorrelation+= crossCorrelations[vid]*result.get().getPredictedLabels()[vid].length;
						meanCorrelation+= crossCorrelations[vid];
						meanRMSError+= rmsErrors[vid];
						totalPredictions+= result.get().getPredictedLabels()[vid].length;
					}
					weightedMeanCorrelation= Math.abs(weightedMeanCorrelation/totalPredictions);
					meanCorrelation= Math.abs(meanCorrelation/totalTestVideos);
					meanRMSError= Math.abs(meanRMSError/totalTestVideos);
					resultsString+= weightedMeanCorrelation+","+meanCorrelation+","+meanRMSError+",";
					finalMeanResults[0]= meanCorrelation; finalMeanResults[1]= meanRMSError;
					for(int vid=0; vid<totalTestVideos; vid++ ){
						resultsString+= crossCorrelations[vid];
						if(vid!=totalTestVideos-1)
							resultsString+=",";
						else
							resultsString+="\n";
					}
					// write to file
					resultsFile.print(resultsString);
					m++;
				}
				resultsFile.close();
				// delete all the training models if it is cross validation
				if(crossvalidation){
					tokens= trainingSetsString.split(";");
					String[] trainingSets= new String[tokens.length];
					int[][] trainingSetVideos= new int[tokens.length][2];
					for(int i=0; i<tokens.length; i++)
					{
						trainingSets[i]= tokens[i].split(",")[0];
						trainingSetVideos[i][0]= Integer.parseInt(tokens[i].split(",")[1]);// start video
						trainingSetVideos[i][1]= Integer.parseInt(tokens[i].split(",")[2]);// end video
					}
					String regressorModelPath="";
					for(m=0; m<regressors.length; m++){
						regressorModelPath+= regressionModelFolder+"/"+regressors[m].replace(",", "_");
						for(int set=0; set< trainingSets.length; set++){
							regressorModelPath+= trainingSets[set]+"_"+trainingSetVideos[set][0]+"_"+trainingSetVideos[set][1];
						}
						regressorModelPath+=".model";
						System.out.println("Deleting the model file "+regressorModelPath);
						new File(regressorModelPath).delete();
					}
				}

			}
		}
		catch(Exception e){e.printStackTrace(); System.exit(1);}
		return finalMeanResults;
	}

	public static void instanceBasedClassificationTesting(Properties requiredParameters, String baseFolder, String response, String[] baseFeatures, String[] featureTypes, String dimensionReduction, String[] classifiers,String trainingSetsString,String testingSetsString, boolean combinePredictions, boolean writePredictions, 
			boolean scaleResponses,boolean crossvalidation, boolean combineFeatures, String dataset, String samplingAlgo){
		try{
			String[] tokens= testingSetsString.split(";");
			String[] testingSets= new String[tokens.length];
			int[][] testingSetVideos= new int[tokens.length][2];
			int totalTestVideos=0;
			String testingDataInfo="";
			for(int i=0; i<tokens.length; i++)
			{
				testingSets[i]= tokens[i].split(",")[0];
				testingSetVideos[i][0]= Integer.parseInt(tokens[i].split(",")[1]);// start video
				testingSetVideos[i][1]= Integer.parseInt(tokens[i].split(",")[2]);// end video
				totalTestVideos+= testingSetVideos[i][1]-testingSetVideos[i][0]+1;
				testingDataInfo+=testingSets[i]+"_"+testingSetVideos[i][0]+"_"+testingSetVideos[i][1];
			}
			String[] allTestVideoCategories= new String[totalTestVideos];
			int[] allTestVideoNumbers= new int[totalTestVideos];
			int count=0;
			for(int set=0; set<testingSets.length; set++){
				for(int vid=testingSetVideos[set][0]; vid<=testingSetVideos[set][1]; vid++){
					allTestVideoCategories[count]= testingSets[set];
					allTestVideoNumbers[count]= vid;
					count++;
				}
			}
			String capitalizedResponse= Utilities.capitalizeFirstLetter(response);
			String[] classifierDetails= new String[classifiers.length];
			for(int f=0; f<baseFeatures.length; f++){
				int threads = classifiers.length;//Runtime.getRuntime().availableProcessors();
				// create pool of threads that will each build a regression model
				ExecutorService service = Executors.newFixedThreadPool(threads);
				// the result returned by each thread is stored in the following variable
				List<Future<ClassificationResults>> futures = new ArrayList<Future<ClassificationResults>>();
				// ****** Map
				// package the regression parameters and spawn one thread per model
				for(int m=0; m< classifiers.length; m++){
					tokens= classifiers[m].split(",");
					String classifierName= tokens[0];
					String[] classifierParams= null;
					// package the parameters into array
					classifierParams= new String[tokens.length-1];
					for(int i=0; i<tokens.length-1; i++)
						classifierParams[i]= tokens[i+1];
					// spawning the thread
					Callable<ClassificationResults> mapper= new ClassifierEvaluator(requiredParameters, classifierName, classifierParams,trainingSetsString,testingSetsString,baseFeatures[f],featureTypes[f],writePredictions);
					// adding the result returned by model to the list
					futures.add(service.submit(mapper));
				}
				service.shutdown();
				try {
					service.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
				} catch (InterruptedException e) {e.printStackTrace();service.shutdown();System.exit(1);}

				// ****** Reduce
				// extract the results from each classifier and write them to a file
				// the format of the results is:
				// regressor_details,test_set,#oftestvideos,response,bagged(ba)/notbagged(nba),baggingfunction,crosscorrelation,rmserror
				String regressionModelFolder= baseFolder+"/"+baseFeatures[f]+featureTypes[f]+"/"+response+"/"+dimensionReduction;
				String resultsFileName= "Results.csv";
				if(crossvalidation)
					resultsFileName= "CrossValidationResults.csv";
				PrintWriter resultsFile= new PrintWriter(new FileWriter(regressionModelFolder + "/"+resultsFileName,true));
				int m=0;
				for(Future<ClassificationResults> result: futures){
					String details= result.get().getClassifierDetails();
					classifierDetails[m]= details;
					int[][][] confMatrices = result.get().getConfusionMatrices();
					double[][] accuracies= result.get().getAccuracies();
					double[][] precisions= result.get().getPrecisions();
					double[][] recalls = result.get().getRecalls();
					String resultsString= "";
					resultsString+=details+",";
					resultsString+= testingDataInfo+","+totalTestVideos+","+capitalizedResponse+",inst,"+samplingAlgo+",";
					double meanAccuracy=0, meanPrecision=0, meanRecall=0;
					for(int vid=0; vid<totalTestVideos; vid++ ){
						Utilities.printArray(confMatrices[vid]);
						for(int c=0; c < accuracies[vid].length; c++){
							meanAccuracy += accuracies[vid][c];
							meanPrecision += precisions[vid][c];
							meanRecall += recalls[vid][c];
						}
						meanAccuracy/= accuracies[vid].length;
						meanPrecision/= accuracies[vid].length;
						meanRecall/= accuracies[vid].length;
					}
					meanAccuracy /= totalTestVideos;
					meanPrecision /= totalTestVideos;
					meanRecall /= totalTestVideos;
					resultsString+= String.format("%.3f,%.3f,%.3f,",meanAccuracy, meanPrecision, meanRecall);
					for(int vid=0; vid<totalTestVideos; vid++ ){
						for(int c=0; c < accuracies[vid].length; c++){
							resultsString+= String.format("%.2f,%.2f,%.2f",accuracies[vid][c],precisions[vid][c],recalls[vid][c]);
							if(c!=accuracies[vid].length-1)
								resultsString+=",";
						}
						if(vid!=totalTestVideos-1)
							resultsString+=",";
						else
							resultsString+="\n";
					}
					// write to file
					resultsFile.print(resultsString);
					m++;
				}
				resultsFile.close();
			}
		}
		catch(Exception e){e.printStackTrace(); System.exit(1);}

	}

	public static double[] ensembleBasedRegressionTesting(Properties requiredParameters, String baseFolder, String response, String[] baseFeatures, String[] featureTypes, String dimensionReduction, String[] regressors,String trainingSetsString,String testingSetsString, boolean combinePredictions, boolean writePredictions,boolean scaleResponses,boolean crossvalidation, String dataset){
		double[] finalMeanResults= new double[2];
		try{
			// Obtain the regressor parameters and algorithms from properties file
			// and spawn threads each for different regression models
			// Do it for each regressor
			String[] tokens= testingSetsString.split(";");
			String[] testingSets= new String[tokens.length];
			int[][] testingSetVideos= new int[tokens.length][2];
			int totalTestVideos=0;
			String testingDataInfo="";
			for(int i=0; i<tokens.length; i++)
			{
				testingSets[i]= tokens[i].split(",")[0];
				testingSetVideos[i][0]= Integer.parseInt(tokens[i].split(",")[1]);// start video
				testingSetVideos[i][1]= Integer.parseInt(tokens[i].split(",")[2]);// end video
				totalTestVideos+= testingSetVideos[i][1]-testingSetVideos[i][0]+1;
				testingDataInfo+=testingSets[i]+"_"+testingSetVideos[i][0]+"_"+testingSetVideos[i][1];
			}
			String[] allTestVideoCategories= new String[totalTestVideos];
			int[] allTestVideoNumbers= new int[totalTestVideos];
			int count=0;
			for(int set=0; set<testingSets.length; set++){
				for(int vid=testingSetVideos[set][0]; vid<=testingSetVideos[set][1]; vid++){
					allTestVideoCategories[count]= testingSets[set];
					allTestVideoNumbers[count]= vid;
					count++;
				}
			}
			tokens= trainingSetsString.split(";");
			String[] trainingSets= new String[tokens.length];
			int[][] trainingSetVideos= new int[tokens.length][2];
			for(int i=0; i<tokens.length; i++)
			{
				trainingSets[i]= tokens[i].split(",")[0];
				trainingSetVideos[i][0]= Integer.parseInt(tokens[i].split(",")[1]);// start video
				trainingSetVideos[i][1]= Integer.parseInt(tokens[i].split(",")[2]);// end video
			}
			String capitalizedResponse= Utilities.capitalizeFirstLetter(response);
			double[][][][] predictionsFromVariousFeatures= new double[totalTestVideos][baseFeatures.length][regressors.length][];
			String[] regressorsDetails= new String[regressors.length];
			for(int m=0; m< regressors.length; m++){
				String trainingDataInfo= "";
				for(int i=0; i<trainingSets.length; i++)
				{
					trainingDataInfo+=trainingSets[i]+"_"+trainingSetVideos[i][0]+"_"+trainingSetVideos[i][1];
				}
				String regressorDetails= regressors[m].replace(",", "_")+trainingDataInfo;
				regressorsDetails[m]= regressorDetails;
				for(int f=0; f<baseFeatures.length; f++){
					int totalWeakLearners=0;
					for(int set=0; set<trainingSets.length; set++)
						totalWeakLearners+= trainingSetVideos[set][1]-trainingSetVideos[set][0]+1;
					double[][][] predictedValues= new double[totalWeakLearners][][]; 
					double[] learnerWeights= new double[totalWeakLearners];
					// Spawn a thread that builds a regression model on each training video
					int weakLearnerCount=0;
					for(int set=0; set< trainingSets.length; set++){
						// create pool of threads that will each build a regression model
						ExecutorService service = Executors.newFixedThreadPool(trainingSetVideos[set][1]-trainingSetVideos[set][0]+1);
						// the result returned by each thread is stored in the following variable
						List<Future<RegressionResults>> futures = new ArrayList<Future<RegressionResults>>();
						for(int vid=trainingSetVideos[set][0]; vid<=trainingSetVideos[set][1]; vid++){
							tokens= regressors[m].split(",");
							String regressorName= tokens[0];
							String[] regressorParams= null;
							// package the regressor parameters into array
							regressorParams= new String[tokens.length-1];
							for(int i=0; i<tokens.length-1; i++)
								regressorParams[i]= tokens[i+1];
							String tSet= trainingSets[set]+","+vid+","+vid;
							// spawning the thread
							Callable<RegressionResults> mapper= new RegressorEvaluator(requiredParameters, regressorName, regressorParams, tSet, testingSetsString,baseFeatures[f],featureTypes[f],false);
							// adding the result returned by model to the list
							futures.add(service.submit(mapper));
							/*RegressorEvaluator evaluator= new RegressorEvaluator(propertiesFilePath[0], regressorName, regressorParams, tSet);
						evaluator.call();*/
						}
						service.shutdown();
						try {
							service.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
						} catch (Exception e) {e.printStackTrace();service.shutdown();System.exit(1);}
						// gather the results from all weak learner
						for(Future<RegressionResults> result: futures){
							predictedValues[weakLearnerCount]= result.get().getPredictedLabels();
							learnerWeights[weakLearnerCount]= result.get().getModelWeight();
							weakLearnerCount++;
						}
						learnerWeights= Utilities.scaleData(learnerWeights, 0, 1);
						Utilities.printArray("Learner Weights:", learnerWeights);
					}
					// once the predictions from all weak learners are gathered, aggregate the predictions
					// using min, max and median
					double[] crossCorrsUsingMean= new double[totalTestVideos];
					double[] rmsErrorsUsingMean= new double[totalTestVideos];
					double[] crossCorrsUsingWMean= new double[totalTestVideos];
					double[] rmsErrorsUsingWMean= new double[totalTestVideos];
					for(int vid=0; vid<totalTestVideos; vid++){
						// load the correct labels for this video
						String labelsFolder=  baseFolder+"/labels";
						String correctLabelsFilePath= labelsFolder+"/"+allTestVideoCategories[vid]+"/"+allTestVideoCategories[vid]+String.format("%03d",allTestVideoNumbers[vid])+"_"+response+".csv";
						if(dataset.equalsIgnoreCase("avec2012")){
							labelsFolder=  baseFolder+"/"+baseFeatures[f]+"Responses";
							correctLabelsFilePath= labelsFolder+"/"+Utilities.capitalizeFirstLetter(allTestVideoCategories[vid])+String.format("%s%03d.csv", capitalizedResponse,allTestVideoNumbers[vid]);
						}
						String[][] temp= Utilities.readCSVFile(correctLabelsFilePath, false);
						double[] correctLabels= new double[temp.length];
						String[] docIds= new String[temp.length];
						int i=0;
						for(; i<temp.length; i++){
							correctLabels[i]= Double.parseDouble(temp[i][0]);
							if(temp[i].length==2){
								if(dataset.equalsIgnoreCase("avec2012"))
									docIds[i]= temp[i][1];
								else
									docIds[i]= temp[i][1];
							}
							else
								docIds[i]= ""+(i+1);
						}
						/*if(scaleResponses)
							correctLabels= Utilities.scaleData(correctLabels, 1, 100);*/
						//double[] minPredictedLabels= new double[correctLabels.length];
						//double[] maxPredictedLabels= new double[correctLabels.length];
						double[] meanPredictedLabels= new double[correctLabels.length];
						double[] wMeanPredictedLabels= new double[correctLabels.length];
						for(int time=0; time<correctLabels.length;time++){
							double[] currTimePredictions= new double[weakLearnerCount];
							for(int learner=0; learner<weakLearnerCount; learner++){
								if(!Double.isNaN(predictedValues[learner][vid][time]))
									currTimePredictions[learner]= predictedValues[learner][vid][time];
								else
									currTimePredictions[learner]=0;
							}
							//minPredictedLabels[time]= Utilities.min(currTimePredictions);
							//maxPredictedLabels[time]= Utilities.max(currTimePredictions);
							//Utilities.printArray("Time "+(time), currTimePredictions);
							meanPredictedLabels[time]= Utilities.mean(currTimePredictions);
							wMeanPredictedLabels[time]= Utilities.weightedMean(currTimePredictions, learnerWeights);
						}
						// if we want to combine predictions from different features do the following
						if(combinePredictions){
							predictionsFromVariousFeatures[vid][f][m]= meanPredictedLabels;
						}
						// write the predictions to file
						if(writePredictions){
							String predictionsFolder= baseFolder+"/FinalResults/"+baseFeatures[f]+featureTypes[f]+dimensionReduction+"/"+regressorDetails;
							if(!new File(predictionsFolder).exists())
								new File(predictionsFolder).mkdir();
							predictionsFolder+= "/"+allTestVideoCategories[vid];
							if(!new File(predictionsFolder).exists())
								new File(predictionsFolder).mkdir();
							predictionsFolder+= "/Ensemble";
							if(!new File(predictionsFolder).exists())
								new File(predictionsFolder).mkdir();
							// Load the original video ids from file
							String videoNamesFilename= baseFolder+"/"+allTestVideoCategories[vid]+"VideoNames.txt";
							String videoName= Utilities.readCSVFile(videoNamesFilename, false)[allTestVideoNumbers[vid]-1][0];
							String predictionsFilePath= predictionsFolder+"/"+videoName+"-"+response.toUpperCase()+".csv";
							//String predictionsFilePath= predictionsFolder+"/"+allTestVideoCategories[vid]+String.format("%03d",vid)+"_"+response+"Mean.csv";
							PrintWriter predictionsFile= new PrintWriter(predictionsFilePath);
							for(i=0; i<docIds.length; i++)
								predictionsFile.println(meanPredictedLabels[i]+","+docIds[i]);
							predictionsFile.close();
							/*predictionsFilePath= predictionsFolder+"/"+allTestVideoCategories[vid]+String.format("%03d",vid)+"_"+response+"WMean.csv";
							predictionsFile= new PrintWriter(predictionsFilePath);
							for(i=0; i<docIds.length; i++)
								predictionsFile.println(wMeanPredictedLabels[i]+","+docIds[i]);
							predictionsFile.close();*/
							// Write the predictions sampled at every 60 secs
							predictionsFilePath= predictionsFolder+"/"+Utilities.capitalizeFirstLetter(allTestVideoCategories[vid])+capitalizedResponse+String.format("%03d",allTestVideoNumbers[vid])+".csv";
							predictionsFile= new PrintWriter(predictionsFilePath);
							for(i=0; i<docIds.length; i++){
								if(i%60==0)
									predictionsFile.println(meanPredictedLabels[i]+","+docIds[i]);
							}
							predictionsFile.close();
							// if there is a NaN in any predictions then abort the method
							boolean containsNaN= false;
							for(i=0; i<meanPredictedLabels.length; i++){
								if(Double.isNaN(meanPredictedLabels[i])|| Double.isInfinite(meanPredictedLabels[i])){
									containsNaN= true;
									break;
								}
							}
							if(containsNaN){
								System.err.println("Found NaN in test video predictions "+allTestVideoCategories[vid]+" "+ videoName);
								System.exit(271);
							}
						}
						crossCorrsUsingMean[vid]= Utilities.calculateCrossCorrelation(correctLabels, meanPredictedLabels);
						rmsErrorsUsingMean[vid]= Utilities.calculateRMSError(correctLabels, meanPredictedLabels);
						crossCorrsUsingWMean[vid]= Utilities.calculateCrossCorrelation(correctLabels, wMeanPredictedLabels);
						rmsErrorsUsingWMean[vid]= Utilities.calculateRMSError(correctLabels, wMeanPredictedLabels);
					}
					Utilities.printArray("", crossCorrsUsingMean);
					//Utilities.printArray("", crossCorrsUsingWMean);
					// write the results to file
					String regressionModelFolder= baseFolder+"/"+capitalizedResponse+"Analysis/"+baseFeatures[f]+featureTypes[f]+dimensionReduction;
					String resultsFileName= "Results.txt";
					if(crossvalidation)
						resultsFileName= "CrossValidationResults.txt";
					PrintWriter resultsFile= new PrintWriter(new FileWriter(regressionModelFolder + "/"+resultsFileName,true));
					String resultsString= regressors[m].replace(",", "_")+trainingDataInfo+",";
					resultsString+= testingDataInfo+","+totalTestVideos+","+capitalizedResponse+",ensem,mean,";
					resultsString+= Math.abs(Utilities.mean(crossCorrsUsingMean))+","+Math.abs(Utilities.mean(rmsErrorsUsingMean))+",";
					finalMeanResults[0]= Utilities.mean(crossCorrsUsingMean); finalMeanResults[1]= Utilities.mean(rmsErrorsUsingMean);
					for(int vid=0; vid<totalTestVideos; vid++ ){
						resultsString+= crossCorrsUsingMean[vid];
						if(vid!=totalTestVideos-1)
							resultsString+=",";
						else
							resultsString+="\n";
					}
					resultsFile.print(resultsString);
					/*resultsString= regressors[m].replace(",", "_")+trainingDataInfo+",";
					resultsString+= testingSet+","+totalTestVideos+","+capitalizedResponse+",ensem,wmean,";
					resultsString+= Math.abs(Utilities.mean(crossCorrsUsingWMean))+","+Math.abs(Utilities.mean(rmsErrorsUsingWMean))+",";
					for(int vid=testingSetVideos[0]; vid<=testingSetVideos[1]; vid++ ){
						resultsString+= crossCorrsUsingWMean[vid-testingSetVideos[0]];
						if(vid!=testingSetVideos[1])
							resultsString+=",";
						else
							resultsString+="\n";
					}
					resultsFile.print(resultsString);*/
					resultsFile.close();
					// delete all the weak learners
					for(int set=0; set< trainingSets.length; set++){
						for(int vid=trainingSetVideos[set][0];vid<=trainingSetVideos[set][1]; vid++){
							String regressorModelPath= regressionModelFolder+"/"+regressors[m].replace(",", "_")+trainingSets[set]+"_"+vid+"_"+vid+".model";
							System.out.println("Deleting the model file "+regressorModelPath);
							new File(regressorModelPath).delete();
						}
					}
				}
			}
			if(combinePredictions){
				String regressionModelFolder= baseFolder+"/"+capitalizedResponse+"Analysis/";
				String resultsFileName= "CombinedResults.txt";
				if(crossvalidation)
					resultsFileName= "CrossValidationCombinedResults.txt";
				PrintWriter combinedResultsFile= new PrintWriter(new FileWriter(regressionModelFolder + "/"+resultsFileName,true));
				for(int m=0; m<regressors.length; m++){
					String resultsString= "";
					double[] crossCorrelations= new double[totalTestVideos];
					double[] rmsErrors= new double[totalTestVideos];
					resultsString+=regressorsDetails[m]+",";
					for(int f=0; f<baseFeatures.length; f++){
						resultsString+=baseFeatures[f]+featureTypes[f]+dimensionReduction;
						if(f!=baseFeatures.length-1)
							resultsString+="_";
						else
							resultsString+=",";
					}
					for(int vid=0; vid<totalTestVideos; vid++){
						double[] meanPredictedLabels= new double[predictionsFromVariousFeatures[vid][0][m].length] ;
						for(int t=0; t<meanPredictedLabels.length; t++){
							double[] currTimePredictions= new double[baseFeatures.length];
							for(int f=0; f<baseFeatures.length; f++){
								currTimePredictions[f]= predictionsFromVariousFeatures[vid][f][m][t];
							}
							meanPredictedLabels[t]= Utilities.mean(currTimePredictions);
						}
						//Utilities.printArray(" ", meanPredictedLabels);
						// find correlation with the correct labels
						String labelsFolder=  baseFolder+"/labels";
						String correctLabelsFilePath= labelsFolder+"/"+allTestVideoCategories[vid]+"/"+allTestVideoCategories[vid]+String.format("%03d",allTestVideoNumbers[vid])+"_"+response+".csv";
						if(dataset.equalsIgnoreCase("avec2012")){
							labelsFolder=  baseFolder+"/"+baseFeatures[0]+featureTypes[1]+"Responses";
							correctLabelsFilePath= labelsFolder+"/"+Utilities.capitalizeFirstLetter(allTestVideoCategories[vid])+String.format("%s%03d.csv", capitalizedResponse,allTestVideoNumbers[vid]);
						}
						String[][] temp= Utilities.readCSVFile(correctLabelsFilePath, false);
						double[] correctLabels= new double[temp.length];
						String[] docIds= new String[temp.length];
						int i=0;
						for(; i<temp.length; i++){
							correctLabels[i]= Double.parseDouble(temp[i][0]);
							if(temp[i].length==2){
								if(dataset.equalsIgnoreCase("avec2012"))
									docIds[i]= temp[i][0];
								else
									docIds[i]= temp[i][1];
							}
							else
								docIds[i]= ""+(i+1);
						}
						/*if(scaleResponses)
							correctLabels= Utilities.scaleData(correctLabels, 1, 100);*/
						crossCorrelations[vid]= Utilities.calculateCrossCorrelation(correctLabels, meanPredictedLabels);
						rmsErrors[vid]= Utilities.calculateRMSError(correctLabels, meanPredictedLabels);
						// write the predictions to file
						if(writePredictions){
							String tempStr="";
							for(int f=0; f<baseFeatures.length; f++){
								tempStr+=baseFeatures[f]+featureTypes[f];
								if(f!=baseFeatures.length-1)
									tempStr+="_";
							}
							String predictionsFolder= baseFolder+"/FinalResults/"+tempStr;
							if(!new File(predictionsFolder).exists())
								new File(predictionsFolder).mkdir();
							predictionsFolder+= "/"+regressorsDetails[m];
							if(!new File(predictionsFolder).exists())
								new File(predictionsFolder).mkdir();
							predictionsFolder+= "/"+allTestVideoCategories[vid];
							if(!new File(predictionsFolder).exists())
								new File(predictionsFolder).mkdir();
							predictionsFolder+= "/Ensemble";
							if(!new File(predictionsFolder).exists())
								new File(predictionsFolder).mkdir();
							// Load the original video ids from file
							String videoNamesFilename= baseFolder+"/"+allTestVideoCategories[vid]+"VideoNames.txt";
							String videoName= Utilities.readCSVFile(videoNamesFilename, false)[allTestVideoNumbers[vid]-1][0];
							String predictionsFilePath= predictionsFolder+"/"+videoName+"-"+response.toUpperCase()+".csv";
							//String predictionsFilePath= predictionsFolder+"/"+allTestVideoCategories[vid]+String.format("%03d",allTestVideoNumbers[vid])+"_"+response+".csv";
							PrintWriter predictionsFile= new PrintWriter(predictionsFilePath);
							for(i=0; i<docIds.length; i++)
								predictionsFile.println(meanPredictedLabels[i]+","+docIds[i]);
							predictionsFile.close();
							predictionsFilePath= predictionsFolder+"/"+Utilities.capitalizeFirstLetter(allTestVideoCategories[vid])+capitalizedResponse+String.format("%03d",allTestVideoNumbers[vid])+".csv";
							predictionsFile= new PrintWriter(predictionsFilePath);
							for(i=0; i<docIds.length; i++){
								if(i%60==0)
									predictionsFile.println(meanPredictedLabels[i]+","+docIds[i]);
							}
							predictionsFile.close();
							// if there is a NaN in any predictions then abort the method
							boolean containsNaN= false;
							for(i=0; i<meanPredictedLabels.length; i++){
								if(Double.isNaN(meanPredictedLabels[i])|| Double.isInfinite(meanPredictedLabels[i])){
									containsNaN= true;
									break;
								}
							}
							if(containsNaN){
								System.err.println("Found NaN in test video predictions "+allTestVideoCategories[vid]+" "+ videoName);
								System.exit(271);
							}
						}
					}
					Utilities.printArray(" ", crossCorrelations);
					resultsString+= testingDataInfo+","+totalTestVideos+","+capitalizedResponse+",ensem,,";
					resultsString+= Math.abs(Utilities.mean(crossCorrelations))+","+Math.abs(Utilities.mean(rmsErrors))+",";
					finalMeanResults[0]= Utilities.mean(crossCorrelations); finalMeanResults[1]= Utilities.mean(rmsErrors);
					for(int vid=0; vid<totalTestVideos; vid++ ){
						resultsString+= crossCorrelations[vid];
						if(vid!=totalTestVideos-1)
							resultsString+=",";
						else
							resultsString+="\n";
					}
					combinedResultsFile.print(resultsString);
				}
				combinedResultsFile.close();
			}
		}catch(Exception e){e.printStackTrace(); System.exit(1);}
		return finalMeanResults;
	}

	/**
	 * This class is called during cross validation to parallelize the execution of different folds
	 * @author prasanthl
	 *
	 */
	class CVFoldHandler implements Callable<String>{

		Properties requiredParameters;
		String baseFolder;
		String response;
		String[] baseFeatures;
		String[] featureTypes;
		String dimensionReduction;
		String[] regressors;
		String trainingSetsString;
		String testingSetsString;
		boolean combinePredictions;
		boolean combineFeatures;
		boolean writePredictions;
		boolean crossvalidation;
		boolean scaleResponses;
		String approach;
		String dataset;
		String samplingAlgo;

		public CVFoldHandler(Properties requiredParameters, String baseFolder, String response, String approach, String[] baseFeatures, String[] featureTypes, String dimensionReduction, String[] regressors,String trainingSetsString,String testingSetsString, boolean combinePredictions, boolean writePredictions, boolean scaleResponses,boolean crossvalidation, boolean combineFeatures, String dataset, String samplingAlgo){
			this.requiredParameters= requiredParameters;
			this.baseFolder= baseFolder;
			this.response= response;
			this.baseFeatures= baseFeatures;
			this.featureTypes= featureTypes;
			this.dimensionReduction= dimensionReduction;
			this.regressors= regressors;
			this.trainingSetsString= trainingSetsString;
			this.testingSetsString= testingSetsString;
			this.combinePredictions= combinePredictions;
			this.writePredictions= writePredictions;
			this.scaleResponses= scaleResponses;
			this.crossvalidation= crossvalidation;
			this.approach= approach;
			this.dataset= dataset;
			this.combineFeatures= combineFeatures;
			this.samplingAlgo= samplingAlgo;
		}

		public String call(){
			if(approach.equalsIgnoreCase("instance")){
				// training
				instanceBasedRegressionTraining(requiredParameters, regressors, trainingSetsString, baseFeatures, featureTypes, combineFeatures);
				// testing
				instanceBasedRegressionTesting(requiredParameters, baseFolder, response, baseFeatures, featureTypes, dimensionReduction, regressors, trainingSetsString, testingSetsString, combinePredictions, scaleResponses,writePredictions, true,combineFeatures,dataset, samplingAlgo);
			}
			else if(approach.equalsIgnoreCase("ensemble")){
				// training
				ensembleBasedRegressionTraining(requiredParameters, regressors, trainingSetsString, baseFeatures, featureTypes);
				// testing
				ensembleBasedRegressionTesting(requiredParameters, baseFolder, response, baseFeatures, featureTypes, dimensionReduction, regressors, trainingSetsString, testingSetsString, combinePredictions, scaleResponses,writePredictions, true,dataset);
			}
			return "";
		}
	}
}
