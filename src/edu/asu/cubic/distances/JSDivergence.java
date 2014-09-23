package edu.asu.cubic.distances;

import java.io.File;

import edu.asu.cubic.util.Utilities;
import weka.core.DenseInstance;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.neighboursearch.PerformanceStats;

/**
 * Used to calculate the Jensen Shannon Divergence between 
 * @author prasanthl
 *
 */
public class JSDivergence extends EuclideanDistance {

	int classIndex;
	public JSDivergence() {
	}

	public JSDivergence(Instances data, int cIndex) {
		super(data);
		classIndex= cIndex;
	}

	@Override 
	public double distance(Instance first, Instance second){
		return distance(first, second, Double.POSITIVE_INFINITY,null);
	}

	@Override 
	public double distance(Instance first, Instance second, double cutOffValue, PerformanceStats stats) { 

		double divergence=0;
		Instance newFirst= new DenseInstance(first);
		Instance newSecond= new DenseInstance(second);
		newFirst.deleteAttributeAt(classIndex);
		newSecond.deleteAttributeAt(classIndex);
		double[] p= newFirst.toDoubleArray();
		double[]q= newSecond.toDoubleArray();
		double[] m= new double[p.length];
		for(int i=0; i<p.length; i++){
			m[i]= (p[i]+q[i])/2;
		}
		/*Utilities.printArray("", p);
		Utilities.printArray("", q);
		Utilities.printArray("", m);
		System.out.println(Utilities.calculateKLDivergence(p, m));
		System.out.println(Utilities.calculateKLDivergence(q, m));*/
		divergence= 0.5*(Utilities.calculateKLDivergence(p, m)+Utilities.calculateKLDivergence(q, m));
		/*System.out.println(divergence);*/
		return divergence;
	}

	public static void main(String[] args) throws Exception{
		String filePath= "C:\\Prasanth\\Studies\\ASU\\CUbiC_Research\\AnnotatedDatasets\\AVEC2013\\ArousalAnalysis\\VideoLPQ\\TrainFeaturesLR.csv";
		CSVLoader loader= new CSVLoader(); 
		loader.setFile(new File(filePath));
		Instances instances= loader.getDataSet();
		System.out.println(new JSDivergence().distance(instances.get(0), instances.get(2)));
	}

}
