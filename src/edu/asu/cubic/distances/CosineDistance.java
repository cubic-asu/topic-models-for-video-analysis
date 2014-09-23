package edu.asu.cubic.distances;

import java.io.File;

import edu.asu.cubic.util.Utilities;
import weka.core.DenseInstance;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.neighboursearch.PerformanceStats;

public class CosineDistance extends EuclideanDistance{

    int classIndex;
	public CosineDistance() {
	}

	public CosineDistance(Instances data, int cIndex) {
		super(data);
		classIndex= cIndex;
	}
	
	@Override 
	public double distance(Instance first, Instance second){
		return distance(first, second, Double.POSITIVE_INFINITY,null);
	}
	
	@Override 
    public double distance(Instance first, Instance second, double cutOffValue, PerformanceStats stats) { 
		double cosineDistance=0;
		Instance newFirst= new DenseInstance(first);
		Instance newSecond= new DenseInstance(second);
		newFirst.deleteAttributeAt(classIndex);
		newSecond.deleteAttributeAt(classIndex);
		double[] p= newFirst.toDoubleArray();
		double[]q= newSecond.toDoubleArray();
		cosineDistance= Utilities.calculateCosineDistance(p, q);
		//System.out.println(cosineDistance);
		return cosineDistance;
	}
	
	public static void main(String[] args) throws Exception{
		String filePath= "C:\\Prasanth\\Studies\\ASU\\CUbiC_Research\\AnnotatedDatasets\\AVEC2013\\ArousalAnalysis\\VideoLPQ\\TrainFeaturesLR.csv";
		CSVLoader loader= new CSVLoader(); 
		loader.setFile(new File(filePath));
		Instances instances= loader.getDataSet();
		System.out.println(new CosineDistance(instances,8).distance(instances.get(0), instances.get(5)));
	}
}
