package edu.asu.cubic.regression;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.io.Serializable;

import com.google.common.base.Optional;

import edu.asu.cubic.util.Utilities;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.PrecomputedKernelMatrixKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.MathExpression;
import weka.filters.unsupervised.attribute.Normalize;

public class SVMRegression {

	/**
	 * Indicates whether to use a parallelized implementation of SVR model
	 */
	boolean parallelize;
	/**
	 * Indicates whether to smoothen the prediction on test data. This is set to true
	 * if the test data is a time sequence
	 */
	boolean smoothenPredictions;
	/**
	 * Indicates whether to apply or not the RBF kernel to SVR model
	 */
	String kernelName;
	double C;
	// rbf kernel parameters
	double gamma;
	// polynomial kernel parameters
	double degree;
	boolean selectFeatures;
	String featureTransformation;
	String featureSelection;
	boolean normalizeFeatures;
	
	public void setFeatureTransformation(String featureTransformation) {
		this.featureTransformation = featureTransformation;
	}
	
	public void setFeatureSelection(String featureSelection) {
		this.featureSelection = featureSelection;
	}

	/**
	 * The @param kernelParams has to be of the form rbf,0.01 or poly,2
	 * @param cParam
	 * @param kernelParams
	 */
	public SVMRegression(double cParam, String kernelParams, boolean normalize){
		C= cParam;
		String[] tokens= kernelParams.trim().split("_");
		kernelName= tokens[0];
		if(kernelName.equalsIgnoreCase("rbf")){
			gamma= Double.parseDouble(tokens[1]);
		}
		else if(kernelName.equalsIgnoreCase("poly")){
			degree= Double.parseDouble(tokens[1]);
		}
		normalizeFeatures= normalize;
	}

	/**
	 * Given a training csv file with a header as input it trains a SVM regression model and writes it to file 
	 * @param trainingModelFilePath
	 * @param trainingFilePath
	 * @param classIndex
	 * @throws Exception
	 */
	public void trainRegressionModel(String trainingModelFilePath, String trainingFilePath, Optional<String> gramMatFilePath) {
		try{
			// Check if the model file already exists
			if(!new File(trainingModelFilePath).exists()){
				System.out.println("Building SVR Model");
				CSVLoader loader= new CSVLoader();
				loader.setFile(new File(trainingFilePath));
				Instances unnormalizedInstances= loader.getDataSet();
				unnormalizedInstances.deleteAttributeAt(0);
				unnormalizedInstances.setClassIndex(unnormalizedInstances.numAttributes()-1);
				Instances trainingInstances= unnormalizedInstances;
				//System.out.println(trainingInstances);
				// normalize the data only if you are not using histogram intersection and kldivergence kernels
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
				// apply transformation filters to data
				/*if(featureTransformation.equalsIgnoreCase("log")){
					MathExpression tranformationFilter= new MathExpression();
					tranformationFilter.setExpression("log(A)");
					tranformationFilter.setInputFormat(trainingInstances);
					trainingInstances= Filter.useFilter(trainingInstances, tranformationFilter);
				}*/
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
				SMOreg regressionModel = new SMOreg();
				regressionModel.setC(C);
				if(kernelName.equalsIgnoreCase("rbf")){
					RBFKernel rbfKernel= new RBFKernel();
					rbfKernel.setGamma(gamma);
					regressionModel.setKernel(rbfKernel);
				}
				else { 
					if(kernelName.equalsIgnoreCase("poly")){
						PolyKernel polyKernel= new PolyKernel();
						polyKernel.setExponent(degree);
						regressionModel.setKernel(polyKernel);
					}
					else if(kernelName.equalsIgnoreCase("hi")){
						HistogramIntersectionKernel hiKernel= new HistogramIntersectionKernel();
						regressionModel.setKernel(hiKernel);
					}
					else if(kernelName.equalsIgnoreCase("kld")){
						KLDivergenceKernel kldKernel= new KLDivergenceKernel();
						regressionModel.setKernel(kldKernel);
					}
					else if(kernelName.equalsIgnoreCase("gram")){
						PrecomputedKernelMatrixKernel matrixKernel = new PrecomputedKernelMatrixKernel();
						assert !gramMatFilePath.equals(Optional.<String>absent());
						matrixKernel.setKernelMatrixFile(new File(gramMatFilePath.get()));
						regressionModel.setKernel(matrixKernel);
					}
				}
				Tag[] tags= new Tag[1];
				tags[0]= new Tag(SMOreg.FILTER_NONE, "TAGS_FILTER");
				regressionModel.setFilterType(new SelectedTag(SMOreg.FILTER_NONE, SMOreg.TAGS_FILTER));
				//Utilities.printArray("In reg:", trainingInstances.get(0).toDoubleArray());
				regressionModel.buildClassifier(trainingInstances);
				// write svm training model to a file
				// for some weird reason object output stream doesn't allow : to be in file name so replace it
				//System.out.println(regressionModel);
				ObjectOutputStream oos = 
						new ObjectOutputStream(new FileOutputStream(trainingModelFilePath));
				oos.writeObject(regressionModel);
				oos.close();
				System.out.println("Just wrote the model to "+trainingModelFilePath);
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
			Instances testInstances= unnormalizedInstances;
			if(normalizeFeatures){
				// normalize the data
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
			// apply transformation filters to data
			/*if(featureTransformation.equalsIgnoreCase("log")){
				MathExpression tranformationFilter= new MathExpression();
				tranformationFilter.setExpression("log(A)");
				tranformationFilter.setInputFormat(testInstances);
				testInstances= Filter.useFilter(testInstances, tranformationFilter);
			}*/
			predictions= new double[testInstances.numInstances()];
			SMOreg regressionModel = new SMOreg();
			regressionModel= (SMOreg)new ObjectInputStream(new FileInputStream(trainingModelFilePath)).readObject();
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
			//System.out.println(regressionModel);
			for(int index=0; index<testInstances.numInstances(); index++){
				Instance testInstance= testInstances.instance(index);
				double prediction= regressionModel.classifyInstance(testInstance);
				predictions[index]= prediction;
			}
		}
		catch(Exception e){
			System.out.println(testingFilePath);
			System.out.println(trainingModelFilePath);
			e.printStackTrace();
			System.exit(1);
		}
		return predictions;
	}


}
