package edu.asu.cubic.distances;

import java.io.File;
import java.util.ArrayList;

import edu.asu.cubic.util.Utilities;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.neighboursearch.PerformanceStats;

public class KLDivergence extends EuclideanDistance {
    int classIndex;
	public KLDivergence() {
	}

	public KLDivergence(Instances data,int cIndex) {
		super(data);
		classIndex= cIndex;
	}
	
	@Override 
	public double distance(Instance first, Instance second){
		return distance(first, second, Double.POSITIVE_INFINITY,null);
	}
	
	@Override 
    public double distance(Instance first, Instance second, double cutOffValue, PerformanceStats stats) { 
		double kldDistance=0;
		Instance newFirst= new DenseInstance(first);
		Instance newSecond= new DenseInstance(second);
		newFirst.deleteAttributeAt(classIndex);
		newSecond.deleteAttributeAt(classIndex);
		double[] p= newFirst.toDoubleArray();
		double[]q= newSecond.toDoubleArray();

		kldDistance= Utilities.calculateKLDivergence(p, q)+Utilities.calculateKLDivergence(q, p);
		return kldDistance;
	}
	
	public static void main(String[] args) throws Exception{
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		for(int i=0; i<5; i++){
			Attribute a= new Attribute("Feature"+(i+1));
			attributes.add(a);
		}
		Instance inst1= new DenseInstance(5);
		inst1.setValue(0, 0.1);inst1.setValue(1, 0.2);inst1.setValue(2, 0.1);inst1.setValue(3, 0.6);inst1.setValue(4, 0);
		Instance inst2= new DenseInstance(5);
		//inst2.setValue(0, 0.1);inst2.setValue(1, 0.2);inst2.setValue(2, 0.1);inst2.setValue(3, 0.6);inst2.setValue(4, 0);
		inst2.setValue(0, 0);inst2.setValue(1, 0);inst2.setValue(2, 0.1);inst2.setValue(3, 0);inst2.setValue(4, 0.9);
		Instances instances= new Instances("data", attributes, 2);
		instances.add(inst1);instances.add(inst2);
		Utilities.printArray("", inst1.toDoubleArray());
		Utilities.printArray("", inst2.toDoubleArray());
		System.out.println("KLDivergence: "+new KLDivergence().distance(instances.get(0), instances.get(1)));
		System.out.println("CosineDistance: "+new CosineDistance().distance(instances.get(0), instances.get(1)));
		System.out.println("JSDivergence: "+new JSDivergence().distance(instances.get(0), instances.get(1)));
		System.out.println("ManhattanDistance: "+new ManhattanDistance().distance(instances.get(0), instances.get(1)));
		System.out.println("EuclideanDistance: "+new EuclideanDistance(instances).distance(instances.get(0), instances.get(1)));
	}

}
