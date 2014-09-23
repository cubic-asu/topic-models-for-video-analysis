package edu.asu.cubic.regression;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.util.concurrent.Callable;

import edu.asu.cubic.util.Utilities;
import moa.classifiers.SGD;
import moa.streams.ArffFileStream;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.MathExpression;
import weka.filters.unsupervised.attribute.Normalize;

public class LRegression {

	/**
	 * The ridge value to be used in regression
	 */
	double ridgeValue;
	double learningRate;
	boolean onlineFlag;
	String featureTransformation;
	String featureSelection;
	boolean normalizeFeatures;
	boolean positiveCoffs; // flag to consider only positive coefficients
	
	public void setFeatureTransformation(String featureTransformation) {
		this.featureTransformation = featureTransformation;
	}

	public void setFeatureSelection(String featureSelection) {
		this.featureSelection = featureSelection;
	}
	
	public LRegression(boolean flag, boolean normalize,boolean positive){
		onlineFlag= flag;
		normalizeFeatures= normalize;
		positiveCoffs= positive;
	}

	public LRegression(boolean flag,double lr, boolean normalize, boolean positive){
		learningRate= lr;
		onlineFlag= flag;
		normalizeFeatures= normalize;
		positiveCoffs= positive;
	}

	/**
	 * Given a training csv file with a header as input it trains a linear regression model and writes it to file 
	 * @param trainingModelFilePath
	 * @param trainingFilePath
	 * @param classIndex
	 * @throws Exception
	 */
	public void trainRegressionModel(String trainingModelFilePath, String trainingFilePath) {
		try{
			// Check if the model file already exists
			if(!new File(trainingModelFilePath).exists()){
				//System.out.println("Building LR Model");
				CSVLoader loader= new CSVLoader();
				loader.setFile(new File(trainingFilePath));
				Instances unnormalizedInstances= loader.getDataSet();
				unnormalizedInstances.deleteAttributeAt(0);
				unnormalizedInstances.setClassIndex(unnormalizedInstances.numAttributes()-1);
				Instances trainingInstances= unnormalizedInstances;
				// apply transformation filters to data
				/*if(featureTransformation.equalsIgnoreCase("log")){
					MathExpression tranformationFilter= new MathExpression();
					tranformationFilter.setExpression("log");
					tranformationFilter.setInputFormat(unnormalizedInstances);
					unnormalizedInstances= Filter.useFilter(unnormalizedInstances, tranformationFilter);
				}*/
				// normalize the data
				if(normalizeFeatures){
					Normalize normalizeFilter= new Normalize();
					normalizeFilter.setInputFormat(unnormalizedInstances);
					for(int i=0; i< unnormalizedInstances.numInstances(); i++)
						normalizeFilter.input(unnormalizedInstances.get(i));
					normalizeFilter.batchFinished();
					trainingInstances= normalizeFilter.getOutputFormat();
					Instance processed;
					while ((processed = normalizeFilter.output()) != null) {
						trainingInstances.add(processed);
					}
				}
				// apply feature selections filters to data
				if(featureSelection.equalsIgnoreCase("cfs")){
					AttributeSelection filter = new AttributeSelection(); 
					CfsSubsetEval eval = new CfsSubsetEval(); 
					BestFirst search = new BestFirst();	
					filter.setEvaluator(eval);	
					filter.setSearch(search);	
					filter.setInputFormat(trainingInstances); 
					trainingInstances = Filter.useFilter(trainingInstances, filter);
					// print out the features selected
					String selFeatsFilePath= trainingModelFilePath.replace(".model", "SelFeats.txt");
					PrintWriter pw= new PrintWriter(new File(selFeatsFilePath));
					System.out.print("Selected features ... ");
					for(int i=0; i< trainingInstances.numAttributes(); i++){
						if(i!=trainingInstances.classIndex()){
							System.out.print(trainingInstances.attribute(i).name()+" ");
							pw.println(trainingInstances.attribute(i).name());
						}
					}
					System.out.println();
					pw.close();
				}
				// train using stochastic gradient descent using MOA
				String filename= trainingFilePath.replace(".csv", ".arff");
				if(!new File(filename).exists()){
					ArffSaver saver= new ArffSaver();
					saver.setInstances(trainingInstances);
					saver.setFile(new File(filename));
					//saver.setDestination(new File(inputFolderPath+"/"+filename));
					saver.writeBatch();
				}
				if(onlineFlag){
					ArffFileStream dataStream= new ArffFileStream(filename,trainingInstances.classIndex()+1);
					dataStream.prepareForUse();
					// Stochastic gradient descent needed to linear streaming linear regressor
					SGD regressionModel = new SGD();
					regressionModel.setModelContext(dataStream.getHeader());
					regressionModel.prepareForUse();
					regressionModel.setLossFunction(2);
					regressionModel.setLearningRate(learningRate);
					regressionModel.setLambda(1E-1);
					/*if(ridgeValue!=0.0)
						regressionModel.setLambda(ridgeValue);*/
					System.out.println(regressionModel);
					int count=1;
					while(dataStream.hasMoreInstances()){
						Instance currInstance= dataStream.nextInstance();
						regressionModel.trainOnInstanceImpl(currInstance);
						if(Double.isInfinite(regressionModel.getVotesForInstance(currInstance)[0]) || Double.isNaN(regressionModel.getVotesForInstance(currInstance)[0])){
							regressionModel.reset();
							System.err.println("**** "+ count);
						}
						count++;
					}
					// write linear regression training model to a file
					ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(trainingModelFilePath));
					oos.writeObject(regressionModel);
					oos.close();
				}
				else{
					LinearRegression regressionModel = new LinearRegression();
					regressionModel.buildClassifier(trainingInstances);
					// write linear regression training model to a file
					ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(trainingModelFilePath));
					oos.writeObject(regressionModel);
					oos.close();
				}

				// delete arff file
				new File(filename).delete();
			}		
		}
		catch(Exception e){
			System.out.println("Exception caugt in LR Regression training");
			System.out.println(trainingFilePath);
			System.out.println(trainingModelFilePath);
			e.printStackTrace();
			System.exit(1);}
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
	public double[] evaluateRegressionModel(String trainingModelFilePath, String testingFilePath){
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
				catch(Exception e){
					if(numReads>10){
						e.printStackTrace();
						System.exit(1);
					}
				}
			}
			unnormalizedInstances.deleteAttributeAt(0);
			unnormalizedInstances.setClassIndex(unnormalizedInstances.numAttributes()-1);
			Instances testInstances= unnormalizedInstances;
			// apply transformation filters to data
			/*if(featureTransformation.equalsIgnoreCase("log")){
				MathExpression tranformationFilter= new MathExpression();
				tranformationFilter.setExpression("log");
				tranformationFilter.setInputFormat(unnormalizedInstances);
				unnormalizedInstances= Filter.useFilter(unnormalizedInstances, tranformationFilter);
			}*/
			// normalize the data
			if(normalizeFeatures){
				Normalize normalizeFilter= new Normalize();
				normalizeFilter.setInputFormat(unnormalizedInstances);
				for(int i=0; i< unnormalizedInstances.numInstances(); i++)
					normalizeFilter.input(unnormalizedInstances.get(i));
				normalizeFilter.batchFinished();
				testInstances= normalizeFilter.getOutputFormat();
				Instance processed;
				while ((processed = normalizeFilter.output()) != null) {
					testInstances.add(processed);
				}
			}
			// if the feature selection is chosen then capture the features selected during
			// training and retain only those features
			if(featureSelection.equalsIgnoreCase("cfs")){
				//System.out.println(testInstances);
				String selFeatsFilePath= trainingModelFilePath.replace(".model", "SelFeats.txt");
				String[][] tokens= Utilities.readCSVFile(selFeatsFilePath, false);
				int[] featsToBeRemoved= new int[testInstances.numAttributes()-tokens.length-1];
				int count=0;
				for(int i=0; i<testInstances.numAttributes(); i++){
					if(i!=testInstances.classIndex()){
						boolean toBeRemoved= true;
						for(int j=0; j< tokens.length; j++){
							if(testInstances.attribute(i).name().equals(tokens[j][0])){
								toBeRemoved= false; 
							}
						}
						if(toBeRemoved){
							featsToBeRemoved[count]= i;
							count++;
						}
					}
				}
				System.out.print("Features not Selected are: ");
				Utilities.printArray(featsToBeRemoved);
				// remove the features
				for(int i= featsToBeRemoved.length-1;i>=0 ; i--){
					testInstances.deleteAttributeAt(featsToBeRemoved[i]);
				}
				//System.out.println(testInstances);
			}
			predictions= new double[testInstances.numInstances()];
			if(onlineFlag){
				SGD regressionModel = new SGD();
				//System.out.println("Loading LR Model");
				regressionModel= (SGD)new ObjectInputStream(new FileInputStream(trainingModelFilePath)).readObject();
				
				//System.out.println(regressionModel);
				for(int index=0; index<testInstances.numInstances(); index++){
					Instance testInstance= testInstances.instance(index);
					if(positiveCoffs){
						double coefficients[]= regressionModel.getWeights();
						double prediction=0;
						for(int i=0; i<coefficients.length-1; i++){
							if(coefficients[i]>0)
								prediction+= testInstance.value(i)*coefficients[i];
						}
						predictions[index]= prediction;
					}
					else{
						double[] prediction= regressionModel.getVotesForInstance(testInstance);
						predictions[index]= prediction[0];
					}
					
				}
			}
			else{
				LinearRegression regressionModel = new LinearRegression();
				//System.out.println("Loading LR Model");
				regressionModel= (LinearRegression)new ObjectInputStream(new FileInputStream(trainingModelFilePath)).readObject();
				double coefficients[]= regressionModel.coefficients();
				for(int index=0; index<testInstances.numInstances(); index++){
					Instance testInstance= testInstances.instance(index);
					double prediction=0;
					if(positiveCoffs){
					// consider only positive coefficients
					// System.out.println("Considering only postive coeffients");
						for(int i=0; i<coefficients.length-1; i++){
							if(coefficients[i]>0)
								prediction+= testInstance.value(i)*coefficients[i];
						}
					}
					else
						prediction= regressionModel.classifyInstance(testInstance);
					predictions[index]= prediction;
				}
			}
		}
		catch(Exception e){
			System.out.println("Exception caugt in LR Regression evaluation");
			System.out.println(testingFilePath);
			System.out.println(trainingModelFilePath);
			e.printStackTrace();
			System.exit(1);}
		return predictions;
	}

}
