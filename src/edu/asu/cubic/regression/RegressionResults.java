package edu.asu.cubic.regression;

import java.util.ArrayList;

/**
 * Is a java bean and a container for the results of a regressor
 * @author prasanthl
 *
 */
public class RegressionResults {
	// predictions at each time step per sequence
	double[][] predictedLabels;
	// cross corrrelations per sequence
	double[] crossCorrelations;
	double[] rmsErrors;
	double[] rSquares;
	double[] meanAbsoluteErrors;
	ArrayList<ArrayList<Integer>> ignoredIndices;
	
	double modelWeight;
	String regressorDetails;
	String testDataDetails;
	
	public double getModelWeight() {
		return modelWeight;
	}
	public void setModelWeight(double modelWeight) {
		this.modelWeight = modelWeight;
	}
	public String getRegressorDetails() {
		return regressorDetails;
	}
	public void setRegressorDetails(String regressorDetails) {
		this.regressorDetails = regressorDetails;
	}
	public double[][] getPredictedLabels() {
		return predictedLabels;
	}
	public void setPredictedLabels(double[][] predictedLabels) {
		this.predictedLabels = predictedLabels;
	}
	public double[] getCrossCorrelations() {
		return crossCorrelations;
	}
	public void setCrossCorrelations(double[] crossCorrelations) {
		this.crossCorrelations = crossCorrelations;
	}
	public double[] getRmsErrors() {
		return rmsErrors;
	}
	public void setRmsErrors(double[] rmsErrors) {
		this.rmsErrors = rmsErrors;
	}
	public String getTestDataDetails() {
		return testDataDetails;
	}
	public void setTestDataDetails(String testDataDetails) {
		this.testDataDetails = testDataDetails;
	}
	public ArrayList<ArrayList<Integer>> getIgnoredIndices() {
		return ignoredIndices;
	}
	public void setIgnoredIndices(ArrayList<ArrayList<Integer>> ignoredIndices) {
		this.ignoredIndices = ignoredIndices;
	}
	public double[] getrSquares() {
		return rSquares;
	}
	public void setrSquares(double[] rSquares) {
		this.rSquares = rSquares;
	}
	public double[] getMeanAbsoluteErrors() {
		return meanAbsoluteErrors;
	}
	public void setMeanAbsoluteErrors(double[] meanAbsoluteErrors) {
		this.meanAbsoluteErrors = meanAbsoluteErrors;
	}
	
}
