package edu.asu.cubic.distances;

import java.io.File;

import edu.asu.cubic.util.Utilities;
import weka.core.DenseInstance;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.neighboursearch.PerformanceStats;

public class HistogramIntersection extends EuclideanDistance{
	
	int classIndex;
	public HistogramIntersection() {
	}

	public HistogramIntersection(Instances data, int cIndex) {
		super(data);
		classIndex= cIndex;
	}
	
	@Override 
	public double distance(Instance first, Instance second){
		return distance(first, second, Double.POSITIVE_INFINITY,null);
	}
	
	@Override 
    public double distance(Instance first, Instance second, double cutOffValue, PerformanceStats stats) { 
		
		double histogramSimilarity = 0;
		// we can do a fast summation and minimum
		int n1 = first.numValues();
		int n2 = second.numValues();
		double total=0;
		for (int p1 = 0, p2 = 0; p1 < n1 && p2 < n2;) {
			int ind1 = first.index(p1);
			int ind2 = second.index(p2);
			if (ind1 == ind2) {
				if (ind1 != classIndex) {
					histogramSimilarity += (first.valueSparse(p1) <= second.valueSparse(p2)) ? first.valueSparse(p1) : second.valueSparse(p2);
					//total+=first.valueSparse(p1);
				}
				p1++;
				p2++;
			} 
			else if (ind1 > ind2) {
					p2++;
				} 
				else {
					p1++;
				}
		}
		
		//System.out.println(cosineDistance);
		// since histogram intersection is a similarity measure subtract it from total to make it a distance
		return Math.abs(histogramSimilarity-total);
	}
	
	public static void main(String[] args) throws Exception{
		String filePath= "C:\\Prasanth\\Studies\\ASU\\CUbiC_Research\\AnnotatedDatasets\\AVEC2013\\ArousalAnalysis\\VideoLPQ\\TrainFeaturesLR.csv";
		CSVLoader loader= new CSVLoader(); 
		loader.setFile(new File(filePath));
		Instances instances= loader.getDataSet();
		Utilities.printArray("", instances.get(2).toDoubleArray());
		Utilities.printArray("", instances.get(3).toDoubleArray());
		System.out.println(new HistogramIntersection(instances,8).distance(instances.get(2), instances.get(3)));
	}

}
