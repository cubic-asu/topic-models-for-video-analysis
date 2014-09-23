package edu.asu.cubic.distances;

import edu.asu.cubic.util.Utilities;
import weka.core.DenseInstance;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.PerformanceStats;

public class ManhattanDistance extends EuclideanDistance {

	int classIndex;
	public ManhattanDistance() {
	}

	public ManhattanDistance(Instances data,int cIndex) {
		super(data);
		classIndex= cIndex;
	}
	
	@Override 
	public double distance(Instance first, Instance second){
		return distance(first, second, Double.POSITIVE_INFINITY,null);
	}
	
	@Override 
    public double distance(Instance first, Instance second, double cutOffValue, PerformanceStats stats) { 
		double manhattanDistance=0;
		Instance newFirst= new DenseInstance(first);
		Instance newSecond= new DenseInstance(second);
		newFirst.deleteAttributeAt(classIndex);
		newSecond.deleteAttributeAt(classIndex);
		double[] p= newFirst.toDoubleArray();
		double[]q= newSecond.toDoubleArray();
		manhattanDistance= Utilities.calculateMahattanDistance(p, q);
		return manhattanDistance;
	}
}
