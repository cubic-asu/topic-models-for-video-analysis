package edu.asu.cubic.distances;

import edu.asu.cubic.util.Utilities;
import weka.core.DenseInstance;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.PerformanceStats;

public class WeightedCosine extends EuclideanDistance {

	int classIndex;
	double[] featWeights;
	public WeightedCosine() {
	}

	public WeightedCosine(Instances data,int cIndex, double[] featWeights) {
		super(data);
		classIndex= cIndex;
		this.featWeights= featWeights;
	}

	@Override 
	public double distance(Instance first, Instance second){
		return distance(first, second, Double.POSITIVE_INFINITY,null);
	}

	@Override 
	public double distance(Instance first, Instance second, double cutOffValue, PerformanceStats stats) { 
		double weightedDistance=0;
		Instance newFirst= new DenseInstance(first);
		Instance newSecond= new DenseInstance(second);
		if(classIndex!=-1){
			newFirst.deleteAttributeAt(classIndex);
			newSecond.deleteAttributeAt(classIndex);
		}
		double[]p= newFirst.toDoubleArray();
		double[]q= newSecond.toDoubleArray();
		/*for(int i=0; i<p.length; i++){
			weightedDistance+= p[i]*featWeights[i]*q[i];
		}*/
		weightedDistance=Utilities.calculateWeightedCosineDistance(p, q, featWeights);
		/*Utilities.printArray("p ", p);
		Utilities.printArray("q ", q);
		Utilities.printArray("featWeights ", featWeights);*/
		return Math.sqrt(weightedDistance);
	}

	public double distance(double[] p, double[] q){
		double weightedDistance=Utilities.calculateWeightedCosineDistance(p, q, featWeights);
		Utilities.printArray("p ", p);
		Utilities.printArray("q ", q);
		Utilities.printArray("featWeights ", featWeights);
		System.out.println("Distance: "+weightedDistance);
		return Math.sqrt(weightedDistance);
	}
}
