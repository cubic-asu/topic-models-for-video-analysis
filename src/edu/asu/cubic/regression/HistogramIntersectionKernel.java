package edu.asu.cubic.regression;

/**
 * This class is an implementation of HistogramIntersectionKernel as defined in the paper
 * Classification Using Intersection Kernel Support Vector Machines is efficient.
   Subhransu Maji and Alexander C. Berg and Jitendra Malik.
   In Proceedings, CVPR 2008, Anchorage, Alaska.
 */
import edu.asu.cubic.util.Utilities;
import weka.classifiers.functions.supportVector.CachedKernel;
import weka.core.Instance;

public class HistogramIntersectionKernel extends CachedKernel {

	 /**
	   * default constructor - does nothing.
	   */
	  public HistogramIntersectionKernel() {
	    super();
	  }
	  
	@Override
	/**
	 * The formula for Histogram Intersection Kernel is
	 * K(a,b) = Sigma_i=1^N{min(a(i),b(i))}
	 */
	protected double evaluate(int id1, int id2, Instance inst1)
			throws Exception {
		double result;
		result= sumMin(inst1,m_data.instance(id2));
		//System.out.println(id1+ " "+ id2 + " :" +result);
		return result;
	}

	protected double sumMin(Instance inst1, Instance inst2) throws Exception {

		double result = 0;
		// we can do a fast summation and minimum
		int n1 = inst1.numValues();
		int n2 = inst2.numValues();
		int classIndex = m_data.classIndex();
		for (int p1 = 0, p2 = 0; p1 < n1 && p2 < n2;) {
			int ind1 = inst1.index(p1);
			int ind2 = inst2.index(p2);
			if (ind1 == ind2) {
				if (ind1 != classIndex) {
					result += (inst1.valueSparse(p1) <= inst2.valueSparse(p2)) ? inst1.valueSparse(p1) : inst2.valueSparse(p2);
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
		/*Utilities.printArray("",inst1.toDoubleArray());
		Utilities.printArray("",inst2.toDoubleArray());
		System.out.println(result);*/
		return (result);
	}

	@Override
	public String globalInfo() {
		return "The formula for Histogram Intersection Kernel is K(a,b) = Sigma_i=1^N{min(a(i),b(i))}" ;
	}
	
	 /**
	   * returns a string representation for the Kernel
	   * 
	   * @return 		a string representaiton of the kernel
	   */
	  public String toString() {
	    return "The formula for Histogram Intersection Kernel is K(a,b) = Sigma_i=1^N{min(a(i),b(i))}" ;
	  }
	  
}
