package edu.asu.cubic.regression;

import java.io.File;
import java.util.Random;

import weka.core.Instances;
import weka.core.converters.CSVLoader;

public class RandomRegression {

	public double[] evaluateRegressionModel(String trainingFilePath, String testingFilePath) {
		Instances unnormalizedInstances=null;
		while(unnormalizedInstances==null ){
			try{
				CSVLoader loader= new CSVLoader(); 
				loader.setFile(new File(testingFilePath));
				unnormalizedInstances= loader.getDataSet();
			}
			catch(Exception npe){
				npe.printStackTrace();
				System.exit(1);
			}
		}
		unnormalizedInstances.deleteAttributeAt(0);
		unnormalizedInstances.setClassIndex(unnormalizedInstances.numAttributes()-1);
		double predictions[]= new double[unnormalizedInstances.numInstances()];
		boolean increasing= true;
		/*if(rand>0.5)
			increasing= true;*/
		if(increasing)
			for(int i=0; i<unnormalizedInstances.numInstances(); i++){
				predictions[i]=i;
			}
		else
			for(int i=0; i<unnormalizedInstances.numInstances(); i++){
				predictions[i]=-i;
			}
		return predictions;
	}
	
}
