package edu.asu.cubic.regression;

import edu.asu.cubic.util.Utilities;
import weka.classifiers.functions.supportVector.CachedKernel;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class KLDivergenceKernel extends CachedKernel {

	 private static double a= 1;
	 private static double b= 1E-6;
	  public KLDivergenceKernel() {
	    super();
	  }
	
	@Override
	/**
	 * The formula for KLDivergence Kernel is
	 * d(p,q)= exp(-a(K(p,q)+K(q,p))+b)
	 * where a= 1, b=1E-6
	 * and K(p,q) = \sum_i=1^N{log(\frac{p(i)}{q(i)})p(i)} and vice versa
	 */
	protected double evaluate(int id1, int id2, Instance inst1)
			throws Exception {
		double result;
		result= symmetricKLD(inst1,m_data.instance(id2));
		//System.out.println(id1+ " "+ id2 + " :" +result);
		return result;
	}

	protected double symmetricKLD(Instance inst1, Instance inst2) throws Exception {

		double kldDistance=0;
		Instance newFirst= new DenseInstance(inst1);
		Instance newSecond= new DenseInstance(inst2);
		newFirst.deleteAttributeAt(m_data.classIndex());
		newSecond.deleteAttributeAt(m_data.classIndex());
		double[] p= newFirst.toDoubleArray();
		double[]q= newSecond.toDoubleArray();
		kldDistance= Utilities.calculateKLDivergence(p, q)+Utilities.calculateKLDivergence(q, p);
		/*Utilities.printArray("", newFirst.toDoubleArray());
		Utilities.printArray("", newSecond.toDoubleArray());
		System.out.println("KLDivergence: "+Math.exp(-a*kldDistance));*/
		return Math.exp(-a*kldDistance);
	}

	
	@Override
	public String globalInfo() {
		return "The formula for KL Divergence Kernel is K(p,q) = \\sum_i=1^N{log(\\frac{p(i)}{q(i)})p(i)}" ;
	}
	
	 /**
	   * returns a string representation for the Kernel
	   * 
	   * @return 		a string representaiton of the kernel
	   */
	  public String toString() {
		  return "The formula for KL Divergence Kernel is K(p,q) = \\sum_i=1^N{log(\\frac{p(i)}{q(i)})p(i)}" ;
	  }
	  

}
