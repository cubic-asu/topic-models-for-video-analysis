package edu.asu.cubic.dimensionality_reduction;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.distribution.MixtureMultivariateNormalDistribution;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.distribution.fitting.MultivariateNormalMixtureExpectationMaximization;
import org.apache.commons.math3.stat.correlation.Covariance;
import org.apache.commons.math3.util.Pair;

import edu.asu.cubic.util.Utilities;
import moa.classifiers.SGD;
import moa.streams.ArffFileStream;
import weka.classifiers.functions.LinearRegression;
import weka.clusterers.EM;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.MathExpression;
import weka.filters.unsupervised.attribute.Normalize;

public class GMM implements Serializable, Cloneable{

	private static final long serialVersionUID = 328229992294253820L;

	int K;
	int maxIters;
	double[][] data;
	String mPath;
	String mName;
	double[][] mixtureCoeffs;
	//EM gmmModel;
	MixtureMultivariateNormalDistribution gmmModel;
    
	public GMM(double[][] data, int numClusters, int iters, String modelPath, String modelName ){
		K= numClusters;
		maxIters= iters;
		this.data= data;
		mPath= modelPath;
		mName= modelName;
	}

	public GMM(double[][] data, GMM trainingModel, String modelPath, String modelName){
		this.data= data;
		K= trainingModel.K;
		mPath= modelPath;
		mName= modelName;
		gmmModel= trainingModel.gmmModel;
	}

	public void runEM(){
		try{
			data= Utilities.normalizeFeatures(data);
			int V= data[0].length;
			double[] weights= new double[K];
			double[][] means= new double[K][V];
			double[][][] covariances= new double[K][V][V];
			// check if the the initial assignments file exists
			HashMap<Integer,ArrayList<ArrayList<Double>>> topicWiseAssignments= new HashMap<Integer, ArrayList<ArrayList<Double>>>();
			for(int k=0; k<K; k++){
				topicWiseAssignments.put(k,new ArrayList<ArrayList<Double>>());
			}
			//double[] phiSum= new double[K];
			for(int d=0; d<data.length; d++){
				double[] probVector= new double[K];
				// for each doc create a prob vector with equal probabilities to all topics
				for(int k=0; k<K; k++)
					probVector[k]= (double)1/K;
				// randomly sample a topic for current doc
				int topic= Utilities.sampleFromDistribution(probVector);
				// fill the topicWiseAssignments 
				if(topicWiseAssignments.isEmpty()){
					ArrayList<Double> temp= new ArrayList<Double>();
					for(double val:data[d])
						temp.add(val);
					ArrayList<ArrayList<Double>> temp1= new ArrayList<ArrayList<Double>>();
					temp1.add(temp);
					topicWiseAssignments.put(topic,temp1);
				}
				else{
					if(topicWiseAssignments.get(topic)==null){
						ArrayList<Double> temp= new ArrayList<Double>();
						for(double val:data[d])
							temp.add(val);
						ArrayList<ArrayList<Double>> temp1= new ArrayList<ArrayList<Double>>();
						temp1.add(temp);
						topicWiseAssignments.put(topic,temp1);
					}
					else{
						ArrayList<Double> temp= new ArrayList<Double>();
						for(double val:data[d])
							temp.add(val);
						ArrayList<ArrayList<Double>> temp1= topicWiseAssignments.get(topic);
						temp1.add(temp);
						topicWiseAssignments.put(topic,temp1);
					}
				}
			}
			// use the topic assignments to calculate the means and variances
			for(int k=0; k< K; k++){
				weights[k]= ((double)topicWiseAssignments.get(k).size())/(double)data.length;
				double[][] currAssignments= new double[topicWiseAssignments.get(k).size()][V];
				//System.out.println(topicWiseAssignments.get(k));
				for(int n=0; n<topicWiseAssignments.get(k).size(); n++){
					for(int v=0; v<V; v++){
						currAssignments[n][v]= topicWiseAssignments.get(k).get(n).get(v);
					}
				}
				means[k]= Utilities.mean(currAssignments, 1);
				covariances[k]= new Covariance(currAssignments).getCovarianceMatrix().getData();
				//Utilities.printArrayToFile(covariances[k], "C:\\Prasanth\\Studies\\ASU\\CUbiC_Research\\AnnotatedDatasets\\AVEC2012\\TopicModels\\M.txt");
				// check if the covariance is singular if so add some noise
				for(int j=0; j<data[0].length; j++)
					covariances[k][j][j]+=Math.random()/10000;
				/*double[][] temp= Utilities.matrixInverse(covariances[k]);
				while(Utilities.containsInfinity(temp) || Utilities.containsNaN(temp)){
					System.out.println("Infinity or Nan in Covariance Inverse");
					for(int j=0; j<data[0].length; j++)
						covariances[k][j][j]+=Math.random()/10000;
					temp= Utilities.matrixInverse(covariances[k]);
				}*/
				//System.out.println("Topic: "+k+": "+currAssignments);
				//topicPrecisions[k]= Utilities.precision(currAssignments);
			}
			MultivariateNormalMixtureExpectationMaximization gmm= new MultivariateNormalMixtureExpectationMaximization(data);
			
			gmm.fit(new MixtureMultivariateNormalDistribution(weights, means, covariances),maxIters,1E-5);
			gmmModel= gmm.getFittedModel();
			List<Pair<Double, MultivariateNormalDistribution>> distributions= gmmModel.getComponents();
			mixtureCoeffs= new double[data.length][K];
			for(int d=0; d<data.length;d++){
				double sum=0;
				for(int k=0; k<K; k++){
					mixtureCoeffs[d][k]= distributions.get(k).getFirst()*distributions.get(k).getSecond().density(data[d]);
					sum+= mixtureCoeffs[d][k];
				}
				for(int k=0; k<K; k++){
					mixtureCoeffs[d][k]/= sum;
				}
				Utilities.printArray("", mixtureCoeffs[d]);
				//System.out.println((Utilities.max(mixtureCoeffs[d], 1)[0]+1));
			}
			/*System.out.println("Building GMM Model");
			String trainingFilePath= mPath+"//"+mName+"TrainingFile.csv";
			PrintWriter trainCSVFile= new PrintWriter(new File(trainingFilePath)) ;
			// write the features to csv file
			for(int i=0; i<data[0].length; i++){
				trainCSVFile.print("Feature"+(i+1));
				if(i<data[0].length-1)
					trainCSVFile.print(",");
			}
			trainCSVFile.println();
			for(int i=0; i<data.length; i++){
				for(int j=0; j<data[0].length; j++){
					trainCSVFile.print(data[i][j]);
					if(j<data[0].length-1)
						trainCSVFile.print(",");
				}
				trainCSVFile.println();
			}
			trainCSVFile.close();
			CSVLoader loader= new CSVLoader();
			loader.setFile(new File(trainingFilePath));
			Instances unnormalizedInstances= loader.getDataSet();
			Normalize normalizeFilter= new Normalize();
			normalizeFilter.setInputFormat(unnormalizedInstances);
			for(int i=0; i< unnormalizedInstances.numInstances(); i++)
				normalizeFilter.input(unnormalizedInstances.get(i));
			normalizeFilter.batchFinished();
			Instances trainingInstances= normalizeFilter.getOutputFormat();
			Instance processed;
			while ((processed = normalizeFilter.output()) != null) {
				trainingInstances.add(processed);
			}
			Utilities.printArray("First Instance",trainingInstances.get(0).toDoubleArray());
			// train using EM clustering algorithm
			gmmModel = new EM();
			gmmModel.setMaxIterations(maxIters);
			gmmModel.setNumClusters(K);
			gmmModel.buildClusterer(trainingInstances);
			double[][][] clustStatistics= gmmModel.getClusterModelsNumericAtts();
			for(int i=0; i<K; i++){
				System.out.println("Cluster: "+(i+1)+" Mean: ");
				for(int j=0; j<trainingInstances.numAttributes(); j++){
					System.out.print(clustStatistics[i][j][0]+",");
				}
				System.out.println();
				System.out.println("Cluster: "+(i+1)+" StdDev: ");
				for(int j=0; j<trainingInstances.numAttributes(); j++){
					System.out.print(clustStatistics[i][j][1]+",");
				}
				System.out.println();
			}
			// extract mixture coefficients for each of the training instance
			mixtureCoeffs= new double[trainingInstances.numInstances()][K];
			int count=0;
			for(Instance inst: trainingInstances){
				mixtureCoeffs[count]= gmmModel.distributionForInstance(inst);
				//System.out.println((Utilities.max(mixtureCoeffs[count], 1)[0]+1));
				count++;
			}
			// delete csv file
			new File(trainingFilePath).delete();*/
		}
		catch(Exception e){
			System.err.println("Exception caugt in GMM model training");
			//System.out.println(trainingFilePath);
			//System.out.println(trainingModelFilePath);
			e.printStackTrace();
			System.exit(1);}

	}

	public void infer() throws Exception {
		
		data= Utilities.normalizeFeatures(data);
		List<Pair<Double, MultivariateNormalDistribution>> distributions= gmmModel.getComponents();
		mixtureCoeffs= new double[data.length][K];
		for(int d=0; d<data.length;d++){
			for(int k=0; k<K; k++){
				mixtureCoeffs[d][k]= distributions.get(k).getSecond().density(data[d]);
			}
			//Utilities.printArray("", mixtureCoeffs[d]);
			//System.out.println((Utilities.max(mixtureCoeffs[d], 1)[0]+1));
		}
		
		/*String testFilePath= mPath+"//"+mName+"TestingFile.csv";
		PrintWriter testCSVFile= new PrintWriter(new File(testFilePath)) ;
		// write the features to csv file
		for(int i=0; i<data[0].length; i++){
			testCSVFile.print("Feature"+(i+1));
			if(i<data[0].length-1)
				testCSVFile.print(",");
		}
		testCSVFile.println();
		for(int i=0; i<data.length; i++){
			for(int j=0; j<data[0].length; j++){
				testCSVFile.print(data[i][j]);
				if(j<data[0].length-1)
					testCSVFile.print(",");
			}
			testCSVFile.println();
		}
		testCSVFile.close();
		CSVLoader loader= new CSVLoader();
		loader.setFile(new File(testFilePath));
		Instances unnormalizedInstances= loader.getDataSet();
		Normalize normalizeFilter= new Normalize();
		normalizeFilter.setInputFormat(unnormalizedInstances);
		for(int i=0; i< unnormalizedInstances.numInstances(); i++)
			normalizeFilter.input(unnormalizedInstances.get(i));
		normalizeFilter.batchFinished();
		Instances testInstances= normalizeFilter.getOutputFormat();
		Instance processed;
		while ((processed = normalizeFilter.output()) != null) {
			testInstances.add(processed);
		}
		mixtureCoeffs= new double[data.length][K];
		int count=0;
		for(Instance inst: testInstances){
			mixtureCoeffs[count]= gmmModel.distributionForInstance(inst);
			for(int i=0; i<K; i++)
				System.out.print(mixtureCoeffs[count][i]+",");
			System.out.println();
			count++;
		}
		// delete csv file
		new File(testFilePath).delete();*/
	}

	public double[][] getMixCoeffs(){
		return mixtureCoeffs;
	}

	/*public EM getGMMModel(){
		return gmmModel;
	}*/
	
	public MixtureMultivariateNormalDistribution getGMMModel(){
		return gmmModel;
	}

	public void cleanUpVariables(){
		data= null;
		mixtureCoeffs= null;
	}

	public static void main(String[] args) throws Exception {
		String[][] tokens= Utilities.readCSVFile("C:\\Prasanth\\Studies\\ASU\\CUbiC_Research\\MyResearch\\Software\\VariationBayesForGMMMatlab\\data.csv", true);
		double[][] data= new double[tokens.length][tokens[0].length];
		for(int i=0; i<tokens.length;i++)
			for(int j=0; j<tokens[0].length;j++){
				data[i][j]= Double.parseDouble(tokens[i][j]);
			}
		GMM gmm= new GMM(data, 3, 100,"C:\\Prasanth\\Studies\\ASU\\CUbiC_Research\\MyResearch\\Software\\VariationBayesForGMMMatlab","GMMModel");
		gmm.runEM();
	}

}
