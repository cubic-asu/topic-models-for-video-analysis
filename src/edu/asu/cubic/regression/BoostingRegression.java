package edu.asu.cubic.regression;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import weka.classifiers.Classifier;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.meta.AdditiveRegression;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.REPTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.NumericCleaner;
import edu.asu.cubic.util.Utilities;

public class BoostingRegression {

	int numIterations;
	double shrinkage;
	String innerRegressorName;
	String[] innerRegressorParams;
	String featureTransformation;
	String featureSelection;
	
	public void setFeatureTransformation(String featureTransformation) {
		this.featureTransformation = featureTransformation;
	}

	public void setFeatureSelection(String featureSelection) {
		this.featureSelection = featureSelection;
	}
	
	public BoostingRegression(){}

	public BoostingRegression(int iters, double shrink, String params){
		numIterations= iters;
		shrinkage= shrink;
		String[] tokens= params.trim().split("_");
		innerRegressorName= tokens[0];
		innerRegressorParams= new String[tokens.length-1];
		for(int i=0;i<tokens.length-1; i++ ){
			innerRegressorParams[i]= tokens[i+1];
		}
	}

	/**
	 * Given a training csv file with a header as input it trains an ensemble of regressors using bagging 
	 * with one level model and writes it to file 
	 * @param trainingModelFilePath
	 * @param trainingFilePath
	 * @param classIndex
	 * @throws Exception
	 */
	public void trainRegressionModel(String trainingModelFilePath, String trainingFilePath) {
		try{
			// Check if the model file already exists
			if(!new File(trainingModelFilePath).exists()){
				System.out.println("Building Boosting Model");
				CSVLoader loader= new CSVLoader();
				loader.setFile(new File(trainingFilePath));
				Instances normalizedInstances= loader.getDataSet();
				normalizedInstances.deleteAttributeAt(0);
				normalizedInstances.setClassIndex(normalizedInstances.numAttributes()-1);
				if(!innerRegressorName.equals("DSR")&& !innerRegressorName.equals("DTR")){
					// normalize the data
					Normalize normalizeFilter= new Normalize();
					normalizeFilter.setInputFormat(normalizedInstances);
					for(int i=0; i< normalizedInstances.numInstances(); i++)
						normalizeFilter.input(normalizedInstances.get(i));
					normalizeFilter.batchFinished();
					normalizedInstances= null;
					normalizedInstances= normalizeFilter.getOutputFormat();
					Instance processed;
					while ((processed = normalizeFilter.output()) != null) {
						normalizedInstances.add(processed);
					}
				}
				else{
					// clean data by rounding the numeric values to 3 decimals
					NumericCleaner decimalRounder= new NumericCleaner();
					decimalRounder.setDecimals(3);
					decimalRounder.setInputFormat(normalizedInstances);
					normalizedInstances= Filter.useFilter(normalizedInstances, decimalRounder);
					// discretize the numeric attributes
					Discretize discretizeFilter= new Discretize();
					//NumericToNominal discretizeFilter= new NumericToNominal();
					discretizeFilter.setInputFormat(normalizedInstances);
					normalizedInstances= Filter.useFilter(normalizedInstances, discretizeFilter);
				}
				Instances trainingInstances= normalizedInstances;
				/*String filename= trainingFilePath.replace(".csv", ".arff");
				if(!new File(filename).exists()){
					ArffSaver saver= new ArffSaver();
					saver.setInstances(trainingInstances);
					saver.setFile(new File(filename));
					//saver.setDestination(new File(inputFolderPath+"/"+filename));
					saver.writeBatch();
				}*/
				Classifier innerRegressor= null;
				if(innerRegressorName.equals("DSR")){
					innerRegressor= new DecisionStump();
					/*((REPTree)innerRegressor).setMaxDepth(1);
					((REPTree)innerRegressor).setNoPruning(true);*/
				}
				else if(innerRegressorName.equals("DTR")){
					innerRegressor= new REPTree();
					((REPTree)innerRegressor).setNoPruning(!Boolean.parseBoolean(innerRegressorParams[0]));
				}
				else if(innerRegressorName.equals("SVR")){
					innerRegressor = new SMOreg();
					RBFKernel rbfKernel= new RBFKernel();
					rbfKernel.setGamma(1E-3);
					((SMOreg)innerRegressor).setKernel(rbfKernel);
					/*if(kernelName.equalsIgnoreCase("rbf")){
						RBFKernel rbfKernel= new RBFKernel();
						rbfKernel.setGamma(gamma);
						regressionModel.setKernel(rbfKernel);
					}
					else if(kernelName.equalsIgnoreCase("poly")){
						PolyKernel polyKernel= new PolyKernel();
						polyKernel.setExponent(degree);
						regressionModel.setKernel(polyKernel);
					}*/
				}
				AdditiveRegression regressionModel= new AdditiveRegression();
				regressionModel.setClassifier(innerRegressor);
				regressionModel.setShrinkage(shrinkage);
				regressionModel.setNumIterations(numIterations);
				
				//regressionModel.setCalcOutOfBag(true);
				regressionModel.buildClassifier(trainingInstances);
				// write regression training model to a file
				ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(trainingModelFilePath));
				oos.writeObject(regressionModel);
				oos.close();
				/*// delete arff file
				new File(filename).delete();*/
			}
		}
		catch(Exception e){
			System.out.println(trainingFilePath);
			System.out.println(trainingModelFilePath);
			e.printStackTrace();
			System.exit(1);
		}		
	}


	/**
	 * Given a trained model and test data it predicts the regression value for each test data point
	 * @param trainingModelFilePath
	 * @param testingFilePath
	 * @param classIndex
	 * @param writePredictions
	 * @param predictionsFilePath
	 * @return
	 * @throws Exception
	 */
	public double[] evaluateRegressionModel(String trainingModelFilePath, String testingFilePath) {

		double[] predictions= null;
		try{
			Instances unnormalizedInstances=null;
			int numReads=0;
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
			if(!innerRegressorName.equals("DSR")&& !innerRegressorName.equals("DTR")){
				// normalize the data
				Normalize normalizeFilter= new Normalize();
				normalizeFilter.setInputFormat(unnormalizedInstances);
				for(int i=0; i< unnormalizedInstances.numInstances(); i++)
					normalizeFilter.input(unnormalizedInstances.get(i));
				normalizeFilter.batchFinished();
				unnormalizedInstances= null;
				unnormalizedInstances= normalizeFilter.getOutputFormat();
				Instance processed;
				while ((processed = normalizeFilter.output()) != null) {
					unnormalizedInstances.add(processed);
				}
			}
			else{
				// clean data by rounding the numeric values to 3 decimals
				NumericCleaner decimalRounder= new NumericCleaner();
				decimalRounder.setDecimals(3);
				decimalRounder.setInputFormat(unnormalizedInstances);
				unnormalizedInstances= Filter.useFilter(unnormalizedInstances, decimalRounder);
				// discretize the numeric attributes
				Discretize discretizeFilter= new Discretize();
				//NumericToNominal discretizeFilter= new NumericToNominal();
				discretizeFilter.setInputFormat(unnormalizedInstances);
				unnormalizedInstances= Filter.useFilter(unnormalizedInstances, discretizeFilter);
			}
			Instances testInstances= unnormalizedInstances;
			AdditiveRegression regressionModel= new AdditiveRegression();
			regressionModel= (AdditiveRegression)new ObjectInputStream(new FileInputStream(trainingModelFilePath)).readObject();
			predictions= new double[testInstances.numInstances()];
			System.out.println(regressionModel);
			for(int index=0; index<testInstances.numInstances(); index++){
				Instance testInstance= testInstances.instance(index);
				double prediction= regressionModel.classifyInstance(testInstance);
				Utilities.printArray("", regressionModel.distributionForInstance(testInstance));
				predictions[index]=prediction;
			}
			//Utilities.printArray("", predictions);
		}
		catch(Exception e){
			System.out.println("Exception in evaluation ");
			System.out.println(testingFilePath);
			System.out.println(trainingModelFilePath);
			e.printStackTrace();
			System.exit(1);
		}
		return predictions;
	}

}
