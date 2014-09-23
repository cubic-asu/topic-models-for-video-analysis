package edu.asu.cubic.regression;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import edu.asu.cubic.distances.CosineDistance;
import edu.asu.cubic.distances.HistogramIntersection;
import edu.asu.cubic.distances.JSDivergence;
import edu.asu.cubic.distances.KLDivergence;
import edu.asu.cubic.distances.ManhattanDistance;
import edu.asu.cubic.util.RegressorBuilder;
import edu.asu.cubic.util.Utilities;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.KStar;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.neighboursearch.LinearNNSearch;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.MathExpression;
import weka.filters.unsupervised.attribute.Normalize;

public class KNNRegression {

	public Instances trainingInstances;
	int K;// # of neighbors
	String distanceMetric;// the distance metric to be used for finding neighbors
	String featureTransformation;
	String featureSelection;
	boolean normalizeFeatures;
	
	public void setFeatureTransformation(String featureTransformation) {
		this.featureTransformation = featureTransformation;
	}

	public void setFeatureSelection(String featureSelection) {
		this.featureSelection = featureSelection;
	}

	public KNNRegression(int k, String distMetric, boolean normalize){
		K= k;
		distanceMetric= distMetric;
		normalizeFeatures= normalize;
	}

	public void crossValidation(String trainingFilePath, String testingFilePath) throws Exception {
		CSVLoader loader= new CSVLoader(); 
		loader.setFile(new File(trainingFilePath));
		Instances unnormalizedInstances= loader.getDataSet();
		unnormalizedInstances.deleteAttributeAt(0);
		unnormalizedInstances.setClassIndex(unnormalizedInstances.numAttributes()-1);
		trainingInstances= unnormalizedInstances;
		Normalize normalizeFilter= new Normalize();
		if(normalizeFeatures){
			normalizeFilter.setInputFormat(unnormalizedInstances);
			Instances normalizedInstances= Filter.useFilter(unnormalizedInstances, normalizeFilter);
			trainingInstances= normalizedInstances;
		}
		loader.setFile(new File(testingFilePath));
		unnormalizedInstances= loader.getDataSet();
		unnormalizedInstances.deleteAttributeAt(0);
		unnormalizedInstances.setClassIndex(unnormalizedInstances.numAttributes()-1);
		Instances allInstances= unnormalizedInstances;
		if(normalizeFeatures){
			normalizeFilter.setInputFormat(unnormalizedInstances);
			Instances normalizedInstances= Filter.useFilter(unnormalizedInstances, normalizeFilter);
			allInstances= normalizedInstances;
		}
		for(Instance inst: trainingInstances){
			allInstances.add(inst);
		}
		IBk knnSearch= new IBk(K);
		if(distanceMetric.equalsIgnoreCase("JSD")) // Jensen Shannon Divergence
			knnSearch.getNearestNeighbourSearchAlgorithm().setDistanceFunction(new JSDivergence(allInstances,allInstances.classIndex()) );
		else if(distanceMetric.equalsIgnoreCase("CD")) // Cosine distance
			knnSearch.getNearestNeighbourSearchAlgorithm().setDistanceFunction(new CosineDistance(allInstances,allInstances.classIndex()) );
		else if(distanceMetric.equalsIgnoreCase("MD")) // Manhattan distance
			knnSearch.getNearestNeighbourSearchAlgorithm().setDistanceFunction(new ManhattanDistance(allInstances,allInstances.classIndex()) );
		else if(distanceMetric.equalsIgnoreCase("KLD")) // Symmetric KL Divergence
			knnSearch.getNearestNeighbourSearchAlgorithm().setDistanceFunction(new KLDivergence(allInstances,allInstances.classIndex()) );
		else if(distanceMetric.equalsIgnoreCase("HID")) // Symmetric KL Divergence
			knnSearch.getNearestNeighbourSearchAlgorithm().setDistanceFunction(new HistogramIntersection(allInstances,allInstances.classIndex()) );
		Evaluation evaluator= new Evaluation(allInstances);
		evaluator.crossValidateModel(knnSearch, allInstances, 5, new Random() );
		System.out.println(evaluator.errorRate());
	}
	
	public double[] evaluateRegressionModel(String trainingFilePath, String testingFilePath) {
		System.out.println("# of neighbors: "+K);
		double[] predictions= null;
		try{
			// load training instances
			Instances unnormalizedInstances=null;
			int numReads=0;
			while(unnormalizedInstances==null ){
				try{
					CSVLoader loader= new CSVLoader(); 
					loader.setFile(new File(trainingFilePath));
					unnormalizedInstances= loader.getDataSet();
					numReads++;
				}
				catch(Exception npe){
					if(numReads>10){
						npe.printStackTrace();
						System.exit(1);
					}
				}
			}
			unnormalizedInstances.deleteAttributeAt(0);
			unnormalizedInstances.setClassIndex(unnormalizedInstances.numAttributes()-1);
			trainingInstances= unnormalizedInstances;
			// apply transformation filters to data
			/*if(featureTransformation.equalsIgnoreCase("log")){
				MathExpression tranformationFilter= new MathExpression();
				tranformationFilter.setExpression("log");
				tranformationFilter.setInputFormat(unnormalizedInstances);
				unnormalizedInstances= Filter.useFilter(unnormalizedInstances, tranformationFilter);
			}*/
			if(normalizeFeatures){
				Normalize normalizeFilter= new Normalize();
				normalizeFilter.setInputFormat(unnormalizedInstances);
				Instances normalizedInstances= Filter.useFilter(unnormalizedInstances, normalizeFilter);
				trainingInstances= normalizedInstances;
			}
			// load test instances
			unnormalizedInstances=null;
			numReads=0;
			while(unnormalizedInstances==null ){
				try{
					CSVLoader loader= new CSVLoader(); 
					loader.setFile(new File(testingFilePath));
					unnormalizedInstances= loader.getDataSet();
					numReads++;
				}
				catch(Exception npe){
					if(numReads>10){
						npe.printStackTrace();
						System.exit(1);
					}
				}
			}
			unnormalizedInstances.deleteAttributeAt(0);
			unnormalizedInstances.setClassIndex(unnormalizedInstances.numAttributes()-1);
			Instances testingInstances= unnormalizedInstances;
			// apply transformation filters to data
			/*if(featureTransformation.equalsIgnoreCase("log")){
				MathExpression tranformationFilter= new MathExpression();
				tranformationFilter.setExpression("log(A)");
				tranformationFilter.setInputFormat(unnormalizedInstances);
				unnormalizedInstances= Filter.useFilter(unnormalizedInstances, tranformationFilter);
			}*/
			if(normalizeFeatures){
				Normalize normalizeFilter= new Normalize();
				normalizeFilter.setInputFormat(unnormalizedInstances);
				Instances normalizedInstances= Filter.useFilter(unnormalizedInstances, normalizeFilter);
				testingInstances= normalizedInstances;
			}
			
			// create pool of threads that will get predictions for each partition of test instances
			int threads= 1;//Math.min(Runtime.getRuntime().availableProcessors(),testingInstances.numInstances());
			ExecutorService service = Executors.newFixedThreadPool(threads);
			// the result returned by each thread is stored in the following variable
			List<Future<HashMap<Integer,Double>>> futures = new ArrayList<Future<HashMap<Integer,Double>>>();
			int partitionStart=0;
			for(int partition=0; partition<threads;partition++){
				int partitionEnd;
				if(partition==threads-1)// if this is last parition assign till the end
					partitionEnd= testingInstances.numInstances()-1;
				else
					partitionEnd= partitionStart+(testingInstances.numInstances()/threads)-1;
				//System.out.println(partitionStart+","+partitionEnd);
				// spawning the thread
				Callable<HashMap<Integer,Double>> mapper= new KNNThread(partitionStart, partitionEnd, testingInstances);
				// adding the result returned by model to the list
				futures.add(service.submit(mapper));
				partitionStart= partitionStart+(testingInstances.numInstances()/threads);
			}
			service.shutdown();
			service.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
			predictions= new double[testingInstances.numInstances()];
			for(Future<HashMap<Integer,Double>> result: futures){
				HashMap<Integer,Double> temp= result.get();
				for(int index: temp.keySet()){
					predictions[index]= temp.get(index);
				}
			}
			
			// calculate the RMS Error and MAE errors
			double[] correctLabels= testingInstances.attributeToDoubleArray(testingInstances.classIndex());
			/*System.out.println("RMS Error is: "+Utilities.rmsError(correctLabels, predictions));
			System.out.println("RMS Error is: "+Utilities.calculateCrossCorrRMSError(correctLabels, predictions)[1]);*/
			//Utilities.printArray("Correct Labels",correctLabels);
			//Utilities.printArray("Predictions",predictions);
		}
		catch(Exception e){e.printStackTrace(); System.exit(1);}
		// normalize the data
		return predictions;
	}
	
	public class KNNThread implements Callable<HashMap<Integer,Double>>{
		int fromIndex;
		int toIndex;
		Instances testInstances;
		public KNNThread(int from, int to,Instances data){
			fromIndex= from;
			toIndex= to;
			testInstances= data;
		}
		@Override
		public HashMap<Integer, Double> call() throws Exception {
			HashMap<Integer, Double> predictions= new HashMap<Integer, Double>();
			/*Instances trainingInstancesWithoutClass= new Instances(trainingInstances);
			
			//System.out.println(trainingInstancesWithoutClass);
			trainingInstancesWithoutClass.remove(trainingInstances.classIndex());
			Instances testingInstancesWithoutClass= testInstances;
			testingInstancesWithoutClass.remove(testInstances.classIndex());*/
			IBk knnClassifier= new IBk(K);
			if(distanceMetric.equalsIgnoreCase("JSD")) // Jensen Shannon Divergence
				knnClassifier.getNearestNeighbourSearchAlgorithm().setDistanceFunction(new JSDivergence(trainingInstances,trainingInstances.classIndex()) );
			else if(distanceMetric.equalsIgnoreCase("CD")) // Cosine distance
				knnClassifier.getNearestNeighbourSearchAlgorithm().setDistanceFunction(new CosineDistance(trainingInstances,trainingInstances.classIndex()) );
			else if(distanceMetric.equalsIgnoreCase("MD")) // Manhattan distance
				knnClassifier.getNearestNeighbourSearchAlgorithm().setDistanceFunction(new ManhattanDistance(trainingInstances,trainingInstances.classIndex()) );
			else if(distanceMetric.equalsIgnoreCase("KLD")) // Symmetric KL Divergence
				knnClassifier.getNearestNeighbourSearchAlgorithm().setDistanceFunction(new KLDivergence(trainingInstances,trainingInstances.classIndex()) );
			knnClassifier.buildClassifier(trainingInstances);
			/*LinearNNSearch knnSearch= new LinearNNSearch(trainingInstancesWithoutClass);
			if(distanceMetric.equalsIgnoreCase("JSD")) // Jensen Shannon Divergence
				knnSearch.setDistanceFunction(new JSDivergence(trainingInstancesWithoutClass) );
			else if(distanceMetric.equalsIgnoreCase("CD")) // Cosine distance
				knnSearch.setDistanceFunction(new CosineDistance(trainingInstancesWithoutClass) );
			else if(distanceMetric.equalsIgnoreCase("MD")) // Manhattan distance
				knnSearch.setDistanceFunction(new ManhattanDistance(trainingInstancesWithoutClass) );
			else if(distanceMetric.equalsIgnoreCase("KLD")) // Symmetric KL Divergence
				knnSearch.setDistanceFunction(new KLDivergence(trainingInstancesWithoutClass) );
			for(int i=fromIndex; i<=toIndex; i++){
				Instances nearestInstances= knnSearch.kNearestNeighbours(testingInstancesWithoutClass.get(i), K);
				System.out.println("Test Instance: "+ (i));
				double[] nearestLabels= nearestInstances.attributeToDoubleArray(testingInstancesWithoutClass.classIndex());
				Utilities.printArray("Nearest Neighbors", nearestLabels);
				double meanPrediction= Utilities.mean(nearestLabels);
				predictions.put(i, meanPrediction);
				System.out.println("Mean Prediction: "+meanPrediction);
			}*/
			for(int i=fromIndex; i<=toIndex; i++){
				//System.out.println("Test Instance: "+ (i));
				double meanPrediction= knnClassifier.classifyInstance(testInstances.get(i));
				predictions.put(i, meanPrediction);
				//System.out.println("Mean Prediction: "+meanPrediction);
			}
			return predictions;
		}
		
		
	}

}
