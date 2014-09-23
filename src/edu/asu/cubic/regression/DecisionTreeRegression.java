package edu.asu.cubic.regression;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import moa.classifiers.DecisionStump;
import moa.streams.ArffFileStream;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.REPTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericCleaner;
import edu.asu.cubic.util.Utilities;

public class DecisionTreeRegression {
	boolean pruneTree;
	double regressorTrainingWeight;
	String featureTransformation;
	String featureSelection;
	
	public void setFeatureTransformation(String featureTransformation) {
		this.featureTransformation = featureTransformation;
	}

	public void setFeatureSelection(String featureSelection) {
		this.featureSelection = featureSelection;
	}
	public DecisionTreeRegression(){}

	public DecisionTreeRegression(boolean flag){
		pruneTree= flag;
	}

	public double getRegressorTrainingWeight() {
		return regressorTrainingWeight;
	}

	public void setRegressorTrainingWeight(double regressorTrainingWeight) {
		this.regressorTrainingWeight = regressorTrainingWeight;
	}

	/**
	 * Given a training csv file with a header as input it trains a Decision tree 
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
				System.out.println("Building DTR Model");
				CSVLoader loader= new CSVLoader();
				loader.setFile(new File(trainingFilePath));
				Instances unnormalizedInstances= loader.getDataSet();
				unnormalizedInstances.deleteAttributeAt(0);
				unnormalizedInstances.setClassIndex(unnormalizedInstances.numAttributes()-1);
				// normalize the data
				/*Normalize normalizeFilter= new Normalize();
				normalizeFilter.setInputFormat(unnormalizedInstances);
				for(int i=0; i< unnormalizedInstances.numInstances(); i++)
					normalizeFilter.input(unnormalizedInstances.get(i));
				normalizeFilter.batchFinished();
				Instances normalizedInstances= normalizeFilter.getOutputFormat();
				Instance processed;
				while ((processed = normalizeFilter.output()) != null) {
					normalizedInstances.add(processed);
				}*/
				// clean data by rounding the numeric values to 3 decimals
				NumericCleaner decimalRounder= new NumericCleaner();
				decimalRounder.setDecimals(3);
				decimalRounder.setInputFormat(unnormalizedInstances);
				unnormalizedInstances= Filter.useFilter(unnormalizedInstances, decimalRounder);
				// discretize the numeric attributes
				Discretize discretizeFilter= new Discretize();
				//NumericToNominal discretizeFilter= new NumericToNominal();
				discretizeFilter.setInputFormat(unnormalizedInstances);
				Instances trainingInstances= Filter.useFilter(unnormalizedInstances, discretizeFilter);
				String filename= trainingFilePath.replace(".csv", ".arff");
				if(!new File(filename).exists()){
					ArffSaver saver= new ArffSaver();
					saver.setInstances(trainingInstances);
					saver.setFile(new File(filename));
					//saver.setDestination(new File(inputFolderPath+"/"+filename));
					saver.writeBatch();
				}
				//weka.classifiers.trees.DecisionStump decisionTree= new weka.classifiers.trees.DecisionStump();
				//M5P decisionTree= new M5P();
				//System.out.println(trainingInstances.attribute(0));
				REPTree decisionTree= new REPTree();
				//decisionTree.setMaxDepth(1);
				decisionTree.setNoPruning(!pruneTree);
				decisionTree.buildClassifier(trainingInstances);
				//System.out.println(decisionTree);
				// here is a mischief! I need to store the cross validation error on the training data 
				// so that it can be used later during evaluating the regressor. I am using the MinVarianceProp
				// property to store this weight
				Evaluation evaluator= new Evaluation(trainingInstances);
				evaluator.evaluateModel(decisionTree, trainingInstances);
				//crossValidateModel("weka.classifiers.trees.REPTree", trainingInstances, 3, decisionTree.getOptions(),new Random(1));
				decisionTree.setMinVarianceProp(evaluator.correlationCoefficient());
				// write regression training model to a file
				ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(trainingModelFilePath));
				oos.writeObject(decisionTree);
				oos.close();
				// delete arff file
				new File(filename).delete();
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
			// normalize the data
			/*Normalize normalizeFilter= new Normalize();
			normalizeFilter.setInputFormat(unnormalizedInstances);
			for(int i=0; i< unnormalizedInstances.numInstances(); i++)
				normalizeFilter.input(unnormalizedInstances.get(i));
			normalizeFilter.batchFinished();
			Instances normalizedInstances= normalizeFilter.getOutputFormat();
			Instance processed;
			while ((processed = normalizeFilter.output()) != null) {
				normalizedInstances.add(processed);
			}*/
			// clean data by rounding the numeric values to 3 decimals
			NumericCleaner decimalRounder= new NumericCleaner();
			decimalRounder.setDecimals(3);
			decimalRounder.setInputFormat(unnormalizedInstances);
			unnormalizedInstances= Filter.useFilter(unnormalizedInstances, decimalRounder);
			// discretize the numeric attributes
			Discretize discretizeFilter= new Discretize();
			//NumericToNominal discretizeFilter= new NumericToNominal();
			discretizeFilter.setInputFormat(unnormalizedInstances);
			Instances testInstances= Filter.useFilter(unnormalizedInstances, discretizeFilter);
			/*weka.classifiers.trees.DecisionStump regressionModel= new weka.classifiers.trees.DecisionStump();
			regressionModel= (weka.classifiers.trees.DecisionStump)new ObjectInputStream(new FileInputStream(trainingModelFilePath)).readObject();*/
			/*M5P regressionModel= new M5P();
			regressionModel= (M5P)new ObjectInputStream(new FileInputStream(trainingModelFilePath)).readObject();*/
			REPTree regressionModel= new REPTree();
			regressionModel= (REPTree)new ObjectInputStream(new FileInputStream(trainingModelFilePath)).readObject();
			regressorTrainingWeight= regressionModel.getMinVarianceProp();
			//Utilities.printArray(regressionModel.getOptions());
			//System.out.println(regressionModel);
			predictions= new double[testInstances.numInstances()];
			for(int index=0; index<testInstances.numInstances(); index++){
				Instance testInstance= testInstances.instance(index);
				double prediction= regressionModel.classifyInstance(testInstance);
				//Utilities.printArray("", prediction);
				// find the index that has maximum value
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
