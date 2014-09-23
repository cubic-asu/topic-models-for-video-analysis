package edu.asu.cubic.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Properties;
import java.util.Random;

import no.uib.cipr.matrix.DenseCholesky;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrices;

import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.math.array.util.Sorting;

import Jama.Matrix;

import edu.asu.cubic.dimensionality_reduction.LDAGibbs;
import edu.asu.cubic.dimensionality_reduction.LDAVB;


public class Utilities {

	/** Constant for normal distribution. */
	public static double m_normConst = Math.log(Math.sqrt(2*Math.PI));

	public static int argmax(double[] x)
	{
		int i;
		double max = x[0];
		int argmax = 0;
		for (i = 1; i < x.length; i++)
		{
			if (x[i] > max)
			{
				max = x[i];
				argmax = i;
			}
		}
		return(argmax);
	}

	public static int assignNearestCluster(double[] centroids, double value){
		int clusterId=0;
		double minDistance=1000;
		//System.out.println(value);
		for(int i=0; i<centroids.length; i++){
			//System.out.print(centroids[i]+","+Math.abs((centroids[i]-value)));
			if(minDistance>Math.abs((centroids[i]-value))){
				clusterId=i;
				minDistance= Math.abs((centroids[i]-value));
			}
		}
		//System.out.println("\n"+(clusterId+1));
		return clusterId+1;
	}

	/**
	 * The following method generates accuracy using the confusion matrix
	 * @param confusionMatrix
	 * @return
	 */
	public static double calculateAccuracy(int[][] confusionMatrix){
		double accuracy=0;
		double totalDataSize=0;
		for(int index=0; index < confusionMatrix.length; index++)
			for(int index1=0; index1 < confusionMatrix.length; index1++){
				if(index == index1)
					accuracy= accuracy + confusionMatrix[index][index1];
				totalDataSize= totalDataSize + confusionMatrix[index][index1];;
			}

		return (accuracy/totalDataSize)*100;
	}

	/**
	 * The following method generates accuracy for a particular using the confusion matrix
	 * @param confusionMatrix
	 * @return
	 */
	public static double calculateAccuracy(int[][] confusionMatrix, int classIndex){
		double totalDataSize=0;
		for(int index=0; index < confusionMatrix.length; index++){
			totalDataSize= totalDataSize + confusionMatrix[classIndex-1][index];
		}

		return (confusionMatrix[classIndex-1][classIndex-1]/totalDataSize)*100;
	}

	/**
	 * This method calculates the precision for a particular class using the confusion matrix 
	 * of all classified classes
	 * @param confusionMatrix
	 * @param classIndex
	 * @return
	 */
	public static double calculatePrecision(int[][] confusionMatrix, int classIndex){
		double precision=0;
		double tp= confusionMatrix[classIndex-1][classIndex-1];
		double fp= 0;
		for(int index=0; index < confusionMatrix.length; index++)
			if(index!=classIndex-1){
				fp= fp + confusionMatrix[index][classIndex-1];
			}
		precision= tp/(tp+fp);
		return precision;
	}

	/**
	 * This method calculates the recall for a particular class using the confusion matrix 
	 * of all classified classes
	 * @param confusionMatrix
	 * @param classIndex
	 * @return
	 */
	public static double calculateRecall(int[][] confusionMatrix, int classIndex){
		double precision=0;
		double tp= confusionMatrix[classIndex-1][classIndex-1];
		double fn= 0;
		for(int index=0; index < confusionMatrix.length; index++)
			if(index!=classIndex-1){
				fn= fn + confusionMatrix[classIndex-1][index];
			}
		precision= tp/(tp+fn);
		return precision;
	}

	/**
	 * Given a set of ranges, find the bin to which a real number belongs to
	 * @param value
	 * @param ranges
	 * @return
	 */
	public static int calculateBin(double value, double[] ranges){
		if(ranges.length == 2){
			if(value < ranges[1] && value >= ranges[0])
				return 1;
			else
				return 0;
		}
		else{
			double[] leftRanges = new double[ranges.length/2+1];
			for(int i=0; i<leftRanges.length; i++){
				leftRanges[i] = ranges[i];
			}
			int leftBin = calculateBin(value, leftRanges);
			if(leftBin > 0){
				return leftBin;
			}
			else {
				try{
					double[] rightRanges = new double[(ranges.length-leftRanges.length)+1];
					for(int i=0; i<rightRanges.length; i++){
						rightRanges[i] = ranges[ranges.length/2+i];
					}
					int rightBin = calculateBin(value, rightRanges);
					if(rightBin > 0){
						return ranges.length/2 + rightBin ;
					}
					else{
						return 0;
					}
				}
				catch(Exception e){
					e.printStackTrace();
					return 0;
				}
			}
		}
	}

	/**
	 * This function calculates the cosine distance between 2 vectors X1 and X2 as
	 * 1 - (sum(X1(i,:).*X2(j,:))/((sum(X1(i,:).*X1(i,:)))^(1/2)*(sum(X2(j,:).*X2(j,:)))^(1/2)))
	 * @param list1
	 * @param list2
	 * @return
	 */
	public static double calculateCosineDistance(double[] vector1, double[] vector2){
		double cosineDist=0;
		double numerator=0, denominator1=0, denominator2=0;
		for(int index=0; index < vector1.length; index++){
			numerator= numerator + vector1[index]*vector2[index];
		}
		for(int index=0; index < vector1.length; index++){
			denominator1= denominator1 + vector1[index]*vector1[index];
		}
		denominator1= Math.pow(denominator1, 0.5);
		for(int index=0; index < vector2.length; index++){
			denominator2= denominator2 + vector2[index]*vector2[index];
		}
		denominator2= Math.pow(denominator2, 0.5);
		cosineDist= 1- (numerator/(denominator1*denominator2));
		return cosineDist;
	}

	/**
	 * Calculates the cross correlation between two sequences
	 * @param x
	 * @param y
	 * @return
	 */
	public static double calculateCrossCorrelation(double[] x, double[] y){
		double[][] xy= new double[x.length][2];
		for(int j=0; j<x.length; j++){
			xy[j][0]= x[j];
			xy[j][1]= y[j];
		}
		PearsonsCorrelation corrObject= new PearsonsCorrelation(xy);
		return corrObject.getCorrelationMatrix().getData()[0][1];
	}

	public static double calculateKLDivergence(double[] vector1, double[] vector2){
		int numAttributes = vector1.length;
		double klDivergence=0;
		// formula for average KL divergence between two distributions is
		// KLDivergence= sigma_i {p_i*log(p_i/*q_i)}
		double p_i,q_i;
		for(int i=0; i < numAttributes; i++){
			p_i= vector1[i];
			q_i= vector2[i];
			double firstLogTerm= 0.0;

			if(p_i==0)
				p_i=1E-10;
			if(q_i==0)
				q_i=1E-10;
			firstLogTerm= Math.log(p_i/q_i)/Math.log(2);
			klDivergence= klDivergence+ (p_i*firstLogTerm) ;
		}
		return klDivergence;
	}

	public static double calculateMahattanDistance(double[] vector1,double[] vector2){
		double distance=0;
		for(int index=0; index<vector1.length; index++)
			distance+= Math.abs(vector1[index]-vector2[index]);
		return distance;
	}

	/**
	 * This function calculates the weighted cosine distance between 2 vectors X1 and X2 as
	 * 1 - (sum(X1.*W*X2)/((sum(X1.*X1))^(1/2)*(sum(X2.*X2))^(1/2)))
	 * @param list1
	 * @param list2
	 * @return
	 */
	public static double calculateWeightedCosineDistance(double[] vector1, double[] vector2, double[] weights){
		double cosineDist=0;
		double numerator=0, denominator1=0, denominator2=0;
		for(int index=0; index < vector1.length; index++){
			numerator= numerator + vector1[index]*vector2[index]*weights[index];
		}
		for(int index=0; index < vector1.length; index++){
			denominator1= denominator1 + vector1[index]*vector1[index]*weights[index];
		}
		denominator1= Math.pow(denominator1, 0.5);
		for(int index=0; index < vector2.length; index++){
			denominator2= denominator2 + vector2[index]*vector2[index]*weights[index];
		}
		denominator2= Math.pow(denominator2, 0.5);
		cosineDist= 1- (numerator/(denominator1*denominator2));
		if(Double.isNaN(cosineDist))
			cosineDist=1;
		return cosineDist;
	}

	/**
	 * Calculates the three quartile values of a given array
	 **/
	public static double[] calculateQuartiles(double[] vector){
		DescriptiveStatistics stats = new DescriptiveStatistics();
		for( int i = 0; i < vector.length; i++) {
			stats.addValue(vector[i]);
		}
		double quartiles[] = new double[5];
		quartiles[0] = min(vector)-1;
		quartiles[1] = stats.getPercentile(25);
		quartiles[2] = stats.getPercentile(50);
		quartiles[3] = stats.getPercentile(75); 
		quartiles[4] = max(vector)+1;
		return quartiles;
	}

	public static double calculateRMSError(double[] x, double[] y){
		double rmsError=0;
		double[][] xy= new double[x.length][2];
		for(int j=0; j<x.length; j++){
			xy[j][0]= x[j];
			xy[j][1]= y[j];
			rmsError+= Math.pow(x[j]-y[j],2);
		}
		rmsError= rmsError/x.length;
		return Math.sqrt(rmsError);
	}

	public static double calculateRSquared(double[] expected, double[]  observed) {
		double ssTotal = 0; // total sum of squares
		double expectedMean = mean(expected); 
		for(int i=0; i<expected.length; i++){
			ssTotal+= Math.pow(expected[i]-expectedMean,2);
		}
		double ssRes = 0; // sum of squares of residuals
		for(int i=0; i<expected.length; i++){
			ssRes+= Math.pow(expected[i]-observed[i],2);
		}
		return 1 - (ssRes/ssTotal);
	}

	public static double calculateMeanAbsoluteError(double[] expected, double[] observed){
		double error = 0;
		for(int i=0; i<expected.length; i++){
			error += Math.abs(expected[i]-observed[i]);
		}
		return error/expected.length;
	}

	public static double[][] calculateTopicIntercorrelations(String topicTypeDistributionsFiles) throws IOException {
		String[][] tokens =  readCSVFile(topicTypeDistributionsFiles, false);
		double[][] unnormTopicTypeDistributions = new double[tokens.length][];
		for(int r = 0; r < tokens.length; r++){
			unnormTopicTypeDistributions[r] = new double[tokens[r].length];
			for(int c = 0; c < tokens[0].length; c++){
				double val = Double.parseDouble(tokens[r][c]);
				if(Double.isInfinite(val) || Double.isNaN(val)){
					unnormTopicTypeDistributions[r][c] = 0.0;
				}
				else{
					unnormTopicTypeDistributions[r][c] = Math.pow(val, 10);
				}
			}
		}
		double[][] topicTypeDistributions= new double[unnormTopicTypeDistributions.length][];
		for(int d=0; d< unnormTopicTypeDistributions.length; d++){
			topicTypeDistributions[d]= Utilities.normalize(unnormTopicTypeDistributions[d]);
		}
		//printArray(topicTypeDistributions);
		int numTopics = topicTypeDistributions.length;

		double[][] topicSimilarities = new double[numTopics][numTopics];
		for(int t1=0; t1<numTopics; t1++){
			for(int t2=0; t2<=t1; t2++){
				if(t1==t2)
					topicSimilarities[t1][t2]=0.0;
				else {
					double distance =  Utilities.calculateCosineDistance(topicTypeDistributions[t1], topicTypeDistributions[t2]);
					if(Double.isNaN(distance) || Double.isInfinite(distance)){
						distance = 1.0;
					}
					topicSimilarities[t1][t2] = 1-distance;
					topicSimilarities[t2][t1] = topicSimilarities[t1][t2];
				}

				//System.out.print(String.format("%.2f",topicSimilarities[t1][t2]) + ",");
			}
			//System.out.println();
		}
		//printArray(topicSimilarities);

		return topicSimilarities;
	}

	/**
	 * Returns the word by capitalizing the first letter e.g for train it returns Train
	 * @param word
	 * @return
	 */
	public static String capitalizeFirstLetter(String word){
		String capitalizedWord= word;
		return new String(""+capitalizedWord.charAt(0)).toUpperCase()+capitalizedWord.substring(1);
	}

	public static double[][] choleskyDecomposition(double[][] mat) throws Exception{
		DenseCholesky ds= DenseCholesky.factorize(new DenseMatrix(mat));
		double[][] chol=  Matrices.getArray(ds.getU());
		while(containsNegative(getDiagonal(chol))){
			System.out.println("Negative diagonal in Chol");
			for(int i=0; i< mat.length; i++){
				mat[i][i]*=1.2;
			}
			ds= DenseCholesky.factorize(new DenseMatrix(mat));
			chol=  Matrices.getArray(ds.getU());
		}
		/*double[][] chol= new weka.core.matrix.Matrix(mat).chol().getL().transpose().getArray();*/
		return chol;
	}

	public static double[][] concatArrays(double[][] mat1, double[][] mat2, int direction){
		if(direction == 0) // horizontal concatination
			assert mat1.length == mat2.length;
		else if(direction == 1) // vertical concatination
			assert mat1[0].length == mat2[0].length;	
		int rows, cols;
		if(direction == 0){
			rows = mat1.length;
			cols = mat1[0].length + mat2[0].length;
		}
		else{
			rows = mat1.length + mat2.length;
			cols = mat1[0].length;
		}
		double[][] result = new double[rows][cols];
		for(int r = 0; r < mat1.length; r++){
			for(int c = 0; c < mat1[0].length; c++){
				result[r][c] = mat1[r][c];
			}
		}
		for(int r = 0; r < mat2.length; r++){
			for(int c = 0; c < mat2[0].length; c++){
				if(direction == 0)
					result[r][mat1[0].length+c] = mat2[r][c];
				else
					result[mat1.length+r][c] = mat2[r][c];
			}
		}

		return result;
	}

	public static boolean containsInfinity(double[][] mat){
		boolean contains= false;
		for(int i=0; i<mat.length; i++){
			for(int j=0; j<mat[0].length; j++){
				if(Double.isInfinite(mat[i][j])){
					contains= true;
					break;
				}
			}
		}
		return contains;
	}

	public static boolean containsInfinity(double[]mat){
		boolean contains= false;
		for(int i=0; i<mat.length; i++){
			if(Double.isInfinite(mat[i])){
				contains= true;
				break;
			}
		}
		return contains;
	}

	public static boolean containsNaN(double[][]  mat){
		boolean contains= false;
		for(int i=0; i<mat.length; i++){
			for(int j=0; j<mat[0].length; j++){
				if(Double.isNaN(mat[i][j])){
					contains= true;
					break;
				}
			}
		}
		return contains;
	}

	public static boolean containsNaN(double[] vector){
		boolean contains=false;
		for(double num: vector){
			if(Double.isNaN(num)){
				contains= true;
				break;
			}
		}
		return contains;
	}

	public static boolean containsZero(double[] vector){
		boolean contains=false;
		for(double num: vector){
			if(num==0.0){
				contains= true;
				break;
			}
		}
		return contains;
	}

	public static boolean containsNegative(double[] vec){
		boolean contains=false;
		for(double num: vec){
			if(num<0.0){
				contains= true;
				break;
			}
		}
		return contains;
	}

	public static double[] getDiagonal(double[][] mat){
		double[] result= new double[mat.length];
		for(int i=0; i<mat.length; i++)
			result[i]= mat[i][i];
		return result;
	}

	/* taylor approximation of first derivative of the log gamma function */
	public static double digamma(double x)
	{
		double p;
		x=x+6;
		p=1/(x*x);
		p=(((0.004166666666667*p-0.003968253986254)*p+
				0.008333333333333)*p-0.083333333333333)*p;
		p=p+Math.log(x)-0.5/x-1/(x-1)-1/(x-2)-1/(x-3)-1/(x-4)-1/(x-5)-1/(x-6);
		return p;
	}



	/**
	 * returns an m x m identity matrix
	 * @param m
	 * @return
	 */
	public static double[][] eye(int m){
		double[][] identityMat= new double[m][m];
		for(int i=0; i<m; i++)
			for(int j=0; j<m; j++){
				if(i==j)
					identityMat[i][j]=1;
				else
					identityMat[i][j]=0;
			}
		return identityMat;
	}

	public static double Gamma(double x)    // We require x > 0
	{
		// Note that the functions Gamma and LogGamma are mutually dependent.
		// Visit http://www.johndcook.com/stand_alone_code.html for the source of this code and more like it.

		/*if (x <= 0.0)
		{
			std::stringstream os;
	        os << "Invalid input argument " << x <<  ". Argument must be positive.";
	        throw std::invalid_argument( os.str() ); 
		}*/

		// Split the function domain into three intervals:
		// (0, 0.001), [0.001, 12), and (12, infinity)

		///////////////////////////////////////////////////////////////////////////
		// First interval: (0, 0.001)
		//
		// For small x, 1/Gamma(x) has power series x + gamma x^2  - ...
		// So in this range, 1/Gamma(x) = x + gamma x^2 with error on the order of x^3.
		// The relative error over this interval is less than 6e-7.

		double gamma = 0.577215664901532860606512090; // Euler's gamma constant
		final double DBL_MAX  =1.7976931348623158e+308 ;
		double y;
		int n;
		int arg_was_less_than_one;
		// numerator coefficients for approximation over the interval (1,2)
		final double p[] =
			{
				-1.71618513886549492533811E+0,
				2.47656508055759199108314E+1,
				-3.79804256470945635097577E+2,
				6.29331155312818442661052E+2,
				8.66966202790413211295064E+2,
				-3.14512729688483675254357E+4,
				-3.61444134186911729807069E+4,
				6.64561438202405440627855E+4
			};

		// denominator coefficients for approximation over the interval (1,2)
		final double q[] =
			{
				-3.08402300119738975254353E+1,
				3.15350626979604161529144E+2,
				-1.01515636749021914166146E+3,
				-3.10777167157231109440444E+3,
				2.25381184209801510330112E+4,
				4.75584627752788110767815E+3,
				-1.34659959864969306392456E+5,
				-1.15132259675553483497211E+5
			};

		double num = 0.0;
		double den = 1.0;
		int i;
		double z ;
		double result;
		double temp;

		if (x < 0.001)
			return 1.0/(x*(1.0 + gamma*x));

		///////////////////////////////////////////////////////////////////////////
		// Second interval: [0.001, 12)

		if (x < 12.0)
		{
			// The algorithm directly approximates gamma over (1,2) and uses
			// reduction identities to reduce other arguments to this interval.

			y = x;
			n = 0;
			arg_was_less_than_one = (y < 1.0? 1:0);

			// Add or subtract integers as necessary to bring y into (1,2)
			// Will correct for this below
			if (arg_was_less_than_one==1)
			{
				y += 1.0;
			}
			else
			{
				n = (int) (Math.floor(y)) - 1;  // will use n later
				y -= n;
			}

			z = y - 1;
			for (i = 0; i < 8; i++)
			{
				num = (num + p[i])*z;
				den = den*z + q[i];
			}
			result = num/den + 1.0;

			// Apply correction if argument was not initially in (1,2)
			if (arg_was_less_than_one==1)
			{
				// Use identity gamma(z) = gamma(z+1)/z
				// The variable "result" now holds gamma of the original y + 1
				// Thus we use y-1 to get back the orginal y.
				result /= (y-1.0);
			}
			else
			{
				// Use the identity gamma(z+n) = z*(z+1)* ... *(z+n-1)*gamma(z)
				for (i = 0; i < n; i++)
					result *= y++;
			}

			return result;
		}

		///////////////////////////////////////////////////////////////////////////
		// Third interval: [12, infinity)

		if (x > 171.624)
		{
			// Correct answer too large to display. Force +infinity.
			temp = DBL_MAX;
			return temp*2.0;
		}

		return Math.exp(LogGamma(x));
	}

	public static double LogGamma(double x ) // x must be positive   
	{
		// Note that the functions Gamma and LogGamma are mutually dependent.
		// Visit http://www.johndcook.com/stand_alone_code.html for the source of this code and more like it.
		/*if (x <= 0.0)
		{
			std::stringstream os;
	        os << "Invalid input argument " << x <<  ". Argument must be positive.";
	        throw std::invalid_argument( os.str() ); 
		}*/

		final double c[]=
			{
				1.0/12.0,
				-1.0/360.0,
				1.0/1260.0,
				-1.0/1680.0,
				1.0/1188.0,
				-691.0/360360.0,
				1.0/156.0,
				-3617.0/122400.0
			};

		double z;
		double sum;
		double series;
		int i;
		final double halfLogTwoPi = 0.91893853320467274178032973640562;
		double logGamma ;
		if (x < 12.0)
		{
			return Math.log(Math.abs(Gamma(x)));
		}

		// Abramowitz and Stegun 6.1.41
		// Asymptotic series should be good to at least 11 or 12 figures
		// For error analysis, see Whittiker and Watson
		// A Course in Modern Analysis (1927), page 252

		z = 1.0/(x*x);
		sum = c[7];
		for (i=6; i >= 0; i--)
		{
			sum *= z;
			sum += c[i];
		}
		series = sum/x;
		logGamma = (x - 0.5)*Math.log(x) - x + halfLogTwoPi + series;    
		return logGamma;
	}

	public static int[][] generateConfusionMatrix(int[] actualValues, int[] predictedValues, int numClasses){

		int[][] confMatrix = new int[numClasses][numClasses];
		for(int i=0; i<numClasses; i++)
			for(int j=0; j<numClasses; j++)
				confMatrix[i][j] = 0;
		for(int i=0; i<actualValues.length; i++){
			int rowIndex = actualValues[i]-1;
			int colIndex = predictedValues[i]-1;
			confMatrix[rowIndex][colIndex]++;
		}

		return confMatrix;
	}

	/*
	 * Creates csv files with name TopicTermMatrix#.csv for each topic #
	 * The file will contain a 10 x 10 matrix of probabilities where the probability
	 * of each cell is the sum of probabilities of all words that belong to that cell
	 */
	public static void generateDataForLFTPlotsFromLBP(String modelFilePath, String csvFilePath, String modelType, int numTopics) throws Exception{
		LDAGibbs ldaModel= (LDAGibbs)new ObjectInputStream(new FileInputStream(modelFilePath)).readObject();
		int[][] nw= ldaModel.getNw();
		int V= nw.length;
		int[] nwSum= ldaModel.getNwSum();
		DecimalFormat fmt= new DecimalFormat("#.####");
		for(int k=0; k<numTopics; k++){
			System.out.println("Topic: "+k);
			double[] nwNormalized= new double[V];
			double[] nwUnnormalized= new double[V];
			for(int w=0; w<V; w++){
				nwNormalized[w]= nw[w][k];
				nwUnnormalized[w]= nw[w][k];
				nwNormalized[w]/=nwSum[k];
				nwNormalized[w]= Double.parseDouble(fmt.format(nwNormalized[w]));
			}
			/*printArray("",nwUnnormalized);
			System.out.println(nwSum[k]);
			printArray("",nwNormalized);*/
			double[][] topicTermMat= new double[10][10];
			for(int row=0; row<10; row++)
				for(int col=0; col<10; col++)
					topicTermMat[row][col]=0;
			int numRows= 10, numCols= 10;;
			int wordsPerCell= V/(numRows*numCols);
			for(int w=0; w<V; w++){
				int block= w/wordsPerCell; 
				int row= block/numRows;
				int col= block%numCols;
				/*if(w==116)
					System.out.println();*/
				topicTermMat[row][col]+=nwNormalized[w];
			}
			double min=100, max=0;
			for(int row=0; row<10; row++){
				if(min>min(topicTermMat[row]))
					min= min(topicTermMat[row]);
				if(max<max(topicTermMat[row]))
					max= max(topicTermMat[row]);
			}
			for(int row=0; row<10; row++){
				topicTermMat[row]= scaleData(topicTermMat[row], min,max, 1, 100);
			}
			//printArray(topicTermMat);
			PrintWriter pw= new PrintWriter(new File(String.format("%s%d.csv", csvFilePath+"/TopicTermMatrix",(k+1))));
			for(int row=0; row<10; row++){
				for(int col=0; col<10; col++){
					pw.print(topicTermMat[row][col]);
					if(col!=9)
						pw.print(",");
				}
				pw.println();
			}
			pw.close();
		}
	}

	public static void generateDataForLFTPlotsForLDAVB(String modelFilePath, String csvFilePath, String modelType, int numTopics) throws Exception{
		LDAVB ldaModel= (LDAVB)new ObjectInputStream(new FileInputStream(modelFilePath)).readObject();
		//int[][] nw= ldaModel.getNw();
		double[][] beta= ldaModel.getBeta();
		int V= 3550;//beta[0].length;
		//int[] nwSum= ldaModel.getNwSum();
		DecimalFormat fmt= new DecimalFormat("#.####");
		for(int k=0; k<numTopics; k++){
			System.out.println("Topic: "+k);
			double[] nwNormalized= new double[V];
			double[] nwUnnormalized= new double[V];
			double nwSum=0;
			for(int w=0; w<V; w++){
				beta[k][w]= Math.exp(beta[k][w]);
				nwSum+= beta[k][w];
			}
			for(int w=0; w<V; w++){
				//nwNormalized[w]= nw[w][k];
				//nwUnnormalized[w]= nw[w][k];
				nwNormalized[w]=beta[k][w]/nwSum;
				//nwNormalized[w]= Double.parseDouble(fmt.format(nwNormalized[w]));
			}
			printArray("",nwNormalized);
			double[][] topicTermMat= new double[1][71];
			for(int row=0; row<1; row++)
				for(int col=0; col<71; col++)
					topicTermMat[row][col]=0;
			int numRows= 1, numCols= 71;;
			int wordsPerCell= V/(numRows*numCols);
			for(int w=0; w<V; w++){
				int block= w/wordsPerCell; 
				//int row= block/numRows;
				int col= block%numCols;
				/*if(w==116)
					System.out.println();*/
				topicTermMat[0][col]+=nwNormalized[w];
			}
			double min=100, max=0;
			for(int row=0; row<1; row++){
				if(min>min(topicTermMat[row]))
					min= min(topicTermMat[row]);
				if(max<max(topicTermMat[row]))
					max= max(topicTermMat[row]);
			}
			for(int row=0; row<1; row++){
				topicTermMat[row]= scaleData(topicTermMat[row], min,max, 1, 100);
			}
			//printArray(topicTermMat);
			PrintWriter pw= new PrintWriter(new File(String.format("%s%d.csv", csvFilePath+"/TopicTermMatrix",(k+1))));
			for(int row=0; row<1; row++){
				for(int col=0; col<71; col++){
					pw.print(topicTermMat[row][col]);
					if(col!=70)
						pw.print(",");
				}
				pw.println();
			}
			pw.close();
		}
	}

	public static void generateDataForEvolutionOfTopicPlots(String modelFolderPath, String modelName, String csvFilePath, String outputFilePath, int numTopics, int iters, double convergence) throws Exception{
		LDAVB ldaModel= (LDAVB)new ObjectInputStream(new FileInputStream(modelFolderPath+"/"+modelName)).readObject();
		int[][] testDocuments=Utilities.loadDocuments(csvFilePath);
		String[][] tempArray= Utilities.readCSVFile(csvFilePath, false);
		String[] docIds= new String[tempArray.length];
		for(int d=0; d<tempArray.length; d++){
			docIds[d]= tempArray[d][0];
		}
		LDAVB testingModel= new LDAVB(testDocuments,ldaModel, modelFolderPath, modelName, iters,convergence);
		testingModel.infer("");
		ArrayList<HashMap<Integer,Integer>> z= testingModel.getZ();
		PrintWriter pw= new PrintWriter(new File(outputFilePath));
		for(int d=0; d<docIds.length; d++){
			int id= Integer.parseInt(docIds[d].substring(3, docIds[d].length()));
			pw.print(id+",");
			int[] termsInDoc= new int[z.get(d).size()];
			int index=0;
			for(Integer key: z.get(d).keySet()){
				termsInDoc[index]= key;
				index++;
			}
			for(int v=0; v<termsInDoc.length; v++){
				pw.print(termsInDoc[v]+","+(z.get(d).get(termsInDoc[v])+1));
				if(v<termsInDoc.length-1)
					pw.print(",");
			}
			pw.println();
		}
		pw.close();
	}

	/*
	 * Used this link as reference: http://pyevolve.sourceforge.net/wordpress/?p=1747
	 */
	public static void generateTfIdfFeatures(String featureName, int vocabulary) throws IOException{
		Properties parameters= new Properties();
		parameters.load(new FileInputStream("Parameters.properties"));
		String baseFolder= parameters.getProperty("baseFolder").trim();
		String inputFolder= baseFolder+"/"+featureName+"/"+featureName+"/docs";
		String outputFolder= baseFolder+"/"+featureName+"/"+featureName+"/tfidf";
		if(!new File(outputFolder).exists()){
			new File(outputFolder).mkdir();
		}
		for(int vidNum=1; vidNum <= 3; vidNum++){
			ArrayList<ArrayList<Double>> tfArrayList;
			// generate tfidf matrix from all training and testing files
			double[][] tfidf;
			tfArrayList= new ArrayList<ArrayList<Double>>();
			for(int vid=vidNum; vid<=vidNum; vid++){
				System.out.println("Train Vid: "+ vid);
				String[][] temp=  Utilities.readCSVFile(inputFolder+"//"+String.format("TrainSeq%03d.csv",(vid)), false);
				int numDocs= temp.length;
				for(int doc=0; doc < numDocs; doc++){
					String[] words= temp[doc][1].split(" ");
					// initialize the frequencies to 0
					ArrayList<Double> frequencies= new ArrayList<Double>();
					for(int v=0; v<vocabulary; v++)
						frequencies.add(0.0);
					// pupolate frequencies for this doc
					for(int w=0; w<words.length; w++){
						if(!words[w].isEmpty() && !words[w].contains("NaN")){
							int v= Integer.parseInt(words[w].trim());
							frequencies.set(v, frequencies.get(v)+1);
						}
					}
					// add this array to main array
					tfArrayList.add(frequencies);
					//System.out.println(frequencies);
				}
			}
			int totalDocs= tfArrayList.size();
			double[][] tf= new double[tfArrayList.size()][vocabulary];
			double[][] termOccuredOrNot= new double[vocabulary][tfArrayList.size()];
			double[] numDocsTermOccurred= new double[vocabulary];
			for(int v=0; v<vocabulary; v++){
				numDocsTermOccurred[v]= 0;
			}
			for(int doc=0; doc < totalDocs; doc++){
				for(int v=0; v<vocabulary; v++){
					tf[doc][v]= tfArrayList.get(doc).get(v);
					if(tf[doc][v]>0){
						termOccuredOrNot[v][doc]=1;
						numDocsTermOccurred[v]++;
					}
					else
						termOccuredOrNot[v][doc]=0;
				}
				//Utilities.printArray(tf[doc]);
			}
			double[][] idf= new double[vocabulary][vocabulary];
			// initialize to zeros
			for(int v1=0; v1<vocabulary; v1++){
				for(int v2=0; v2<vocabulary; v2++){
					idf[v1][v2]=0;
				}
			}
			for(int v=0; v<vocabulary; v++){
				idf[v][v]= Math.log10(totalDocs/(1+numDocsTermOccurred[v]));
				//Utilities.printArray(idf[v]);
			}
			Matrix tfMat= new Matrix(tf);
			Matrix idfMat= new Matrix(idf);
			//System.out.println("tf");
			//Utilities.printArray(tf);
			//System.out.println("idf");
			//Utilities.printArray(idf);
			Matrix tfIdfMat= tfMat.times(idfMat);
			tfidf= tfIdfMat.getArray();

			// Generating training files
			totalDocs=0;
			for(int vid=vidNum; vid<=vidNum; vid++){
				String[][] temp= Utilities.readCSVFile(inputFolder+"//"+String.format("TrainSeq%03d.csv",(vid)), false);
				int numDocs= temp.length; 
				PrintWriter pw1= new PrintWriter(new File(outputFolder+"//"+String.format("TrainSeq%03d.csv",(vid) )));
				pw1.print("DocId,");
				for(int i=1; i <=vocabulary; i++){
					pw1.print("Feature"+i);
					if(i<vocabulary)
						pw1.print(",");
					else
						pw1.println();
				}
				for(int doc=totalDocs; doc<totalDocs+numDocs; doc++){
					pw1.print(temp[doc-totalDocs][0]+",");
					for(int v=0; v<vocabulary; v++){
						pw1.print(tfidf[doc][v]);
						if(v<vocabulary-1)
							pw1.print(",");
					}
					pw1.println();
				}
				pw1.close();
				totalDocs+= numDocs;
			}

			// generate tfidf matrix from testing files
			tfArrayList= new ArrayList<ArrayList<Double>>();
			for(int vid=vidNum; vid<=vidNum; vid++){
				System.out.println("Test Vid: "+ vid);
				String[][] temp=  Utilities.readCSVFile(inputFolder+"//"+String.format("TestSeq%03d.csv",(vid)), false);
				int numDocs= temp.length;
				for(int doc=0; doc < numDocs; doc++){
					String[] words= temp[doc][1].split(" ");
					// initialize the frequencies to 0
					ArrayList<Double> frequencies= new ArrayList<Double>();
					for(int v=0; v<vocabulary; v++)
						frequencies.add(0.0);
					// pupolate frequencies for this doc
					for(int w=0; w<words.length; w++){
						if(!words[w].isEmpty() && !words[w].contains("NaN")){
							int v= Integer.parseInt(words[w].trim());
							frequencies.set(v, frequencies.get(v)+1);
						}
					}
					// add this array to main array
					tfArrayList.add(frequencies);
				}
			}
			totalDocs= tfArrayList.size();
			tf= new double[tfArrayList.size()][vocabulary];
			termOccuredOrNot= new double[vocabulary][tfArrayList.size()];
			numDocsTermOccurred= new double[vocabulary];
			for(int v=0; v<vocabulary; v++){
				numDocsTermOccurred[v]= 0;
			}
			for(int doc=0; doc < totalDocs; doc++){
				for(int v=0; v<vocabulary; v++){
					tf[doc][v]= tfArrayList.get(doc).get(v);
				}
			}
			tfMat= new Matrix(tf);
			idfMat= new Matrix(idf);// use the idf from training data
			tfIdfMat= tfMat.times(idfMat);
			tfidf= tfIdfMat.getArray();
			totalDocs = 0;
			// Generating testing files
			for(int vid=vidNum; vid<=vidNum; vid++){
				String[][] temp= Utilities.readCSVFile(inputFolder+"//"+String.format("TestSeq%03d.csv",(vid)), false);
				int numDocs= temp.length; 
				PrintWriter pw1= new PrintWriter(new File(outputFolder+"//"+String.format("TestSeq%03d.csv",(vid) )));
				pw1.print("DocId,");
				for(int i=1; i <=vocabulary; i++){
					pw1.print("Feature"+i);
					if(i<vocabulary)
						pw1.print(",");
					else
						pw1.println();
				}
				for(int doc=totalDocs; doc<totalDocs+numDocs; doc++){
					pw1.print(temp[doc-totalDocs][0]+",");
					for(int v=0; v<vocabulary; v++){
						pw1.print(tfidf[doc][v]);
						if(v<vocabulary-1)
							pw1.print(",");
					}
					pw1.println();
				}
				pw1.close();
				totalDocs+= numDocs;
			}
		}
	}

	/**
	 * This method extracts the topic definitions \beta from a LDA model file and writes
	 * them to a .beta file. This is for LDAVB object files.
	 * @throws ClassNotFoundException 
	 * @throws IOException 
	 * @throws FileNotFoundException 
	 */
	public static void generateBetaFiles() throws FileNotFoundException, IOException, ClassNotFoundException{
		LDAVB ldaModel= (LDAVB)new ObjectInputStream(new FileInputStream("C:\\Prasanth\\Studies\\ASU\\CUbiC_Research\\Presentations_Reports\\PhDProspectus\\ComprehensiveExam\\AudioSynchedRawLDAVB_30.model"))
		.readObject();
		PrintWriter betaFile= new PrintWriter(new File("C:\\Prasanth\\Studies\\ASU\\CUbiC_Research\\Presentations_Reports\\PhDProspectus\\ComprehensiveExam\\Audio30.beta"));
		double[][] beta= ldaModel.getBeta();
		int K= 30;
		int V= 113200;
		// normalize beta
		for (int i = 0; i < K; i++)
		{   double sum=0;
		for (int j = 0; j < V ; j++)
		{
			sum+= beta[i][j];
		}
		for (int j = 0; j < V ; j++)
		{
			beta[i][j]= beta[i][j]/sum;
		}
		}
		for (int i = 0; i < K; i++)
		{
			for (int j = 0; j < V ; j++)
			{
				betaFile.print(String.format("%.3f", Math.log(beta[i][j])));
				if(j<V-1)
					betaFile.print(",");
			}
			betaFile.println();
		}
		betaFile.close();
	}

	public static String generateRandomString(int length){
		String[] randomChars= {"A","B","C","D","E","F","G","H","I","J","K","L","M",
				"N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
				"1","2","3","4","5","6","7","8","9","0",
				"!","@","#","$","&","%","@","#"};
		String randomString= "";
		Random rnd= new Random();
		for(int i=0; i<length; i++){
			randomString+= new String(""+randomChars[rnd.nextInt(randomChars.length)]);
		}
		return randomString;
	}

	public static int[][] loadDocuments(String filePath) throws IOException{
		int[][] documents;
		ArrayList<ArrayList<Integer>> documentsArrList= new ArrayList<ArrayList<Integer>>();
		BufferedReader file= new BufferedReader(new FileReader(filePath));
		String newLine= file.readLine();
		while(newLine!=null && !newLine.isEmpty()){
			String[] tokens= null;
			try{
				tokens= newLine.split(",")[1].trim().split(" ");
				//System.out.println(newLine);
			}
			catch(ArrayIndexOutOfBoundsException a){
				System.out.println(newLine);
			}
			ArrayList<Integer>  temp= new ArrayList<Integer>();
			for(int i=0; i<tokens.length; i++){
				if(!tokens[i].trim().equalsIgnoreCase("nan"))
					temp.add(Integer.parseInt(tokens[i].trim()));
				else{
					temp= new ArrayList<Integer>();
					break;
				}
			}
			documentsArrList.add(temp);
			newLine= file.readLine();
		}
		file.close();
		//System.out.println(documentsArrList);
		documents= new int[documentsArrList.size()][];//documentsArrList.size()
		for(int i=0; i<documentsArrList.size();i++){
			ArrayList<Integer>  temp= documentsArrList.get(i);
			documents[i]= new int[temp.size()];
			for(int j=0; j<temp.size(); j++)
				documents[i][j]=temp.get(j);
		}
		//printArray(documents);
		return documents;
	}

	public static int[] loadSequenceIds(String filePath, int maxNumDigits) throws IOException{
		int[] sequenceIds;
		ArrayList<Integer> documentsArrList= new ArrayList<Integer>();
		BufferedReader file= new BufferedReader(new FileReader(filePath));
		String newLine= file.readLine();
		int prevId= Integer.parseInt(newLine.split(",")[0].substring(1, maxNumDigits+1));
		int sequenceId=1;
		while(newLine!=null && !newLine.isEmpty()){
			String token= newLine.split(",")[0];
			int currId= Integer.parseInt(token.substring(1, maxNumDigits+1));
			if(currId!=prevId)
				sequenceId++;
			documentsArrList.add(sequenceId);
			prevId= currId;
			newLine= file.readLine();
		}
		file.close();
		//System.out.println(documentsArrList);
		sequenceIds= new int[documentsArrList.size()];//documentsArrList.size()
		for(int i=0; i<documentsArrList.size();i++){
			sequenceIds[i]= documentsArrList.get(i);

		}
		return sequenceIds;
	}

	public static double[] log(double[] vec){
		double[] result= new double[vec.length];
		for(int i=0; i<vec.length; i++)
			result[i]= Math.log(vec[i]);
		return result;
	}

	public static double logmvgamma(double x, double d){
		double y= (d*(d-1)/4)*Math.log(Math.PI);
		//System.out.println(y);
		for(int i=0; i<d; i++){
			y+= LogGamma(x-((double)i/2));
			//System.out.println(x-((double)i/2));
		}
		return y;
	}

	/*
	 * given log(a) and log(b), return log(a + b)
	 *
	 */
	public static double log_sum(double log_a, double log_b)
	{
		double v;

		if (log_a < log_b)
		{
			v = log_b+Math.log(1 + Math.exp(log_a-log_b));
		}
		else
		{
			v = log_a+Math.log(1 + Math.exp(log_b-log_a));
		}
		return(v);
	}

	/**
	 * Density function of normal distribution.
	 * @param x input value
	 * @param mean mean of distribution
	 * @param stdDev standard deviation of distribution
	 * @return the density
	 */
	public static double logNormalDens (double x, double mean, double stdDev) {

		double diff = x - mean;
		return - (diff * diff / (2 * stdDev * stdDev))  - m_normConst - Math.log(stdDev);
		//return Math.log(new NormalDistribution(mean,stdDev).probability(x-(stdDev/100),x+(stdDev/100)));
	}

	public static void main(String[] args) throws Exception{
		generateTfIdfFeatures("report",2429);
		/*String category = "report";
		String[][] tokens1 = readCSVFile("/Users/prasanthlade/Documents/ibm/"+category+"/responses/TrainTime001.csv", false);
		String[][] tokens2 = readCSVFile("/Users/prasanthlade/Documents/ibm/"+category+"/responses/TestTime001.csv", false);
		double[] responses  = new double[tokens1.length+tokens2.length];
		int count =0;
		for(int i =0; i<tokens1.length; i++){
			responses[count] = Double.parseDouble(tokens1[i][1]);
			count++;
		}
		for(int i =0; i<tokens2.length; i++){
			responses[count] = Double.parseDouble(tokens2[i][1]);
			count++;
		}
		double[] quartiles = calculateQuartiles(responses);
		printArray(quartiles);
		for(int i=1; i<=3; i++){
			tokens1 = readCSVFile("/Users/prasanthlade/Documents/ibm/"+category+"/responses/TrainTime00"+i+".csv", false);
			tokens2 = readCSVFile("/Users/prasanthlade/Documents/ibm/"+category+"/responses/TestTime00"+i+".csv", false);
			PrintWriter pw = new PrintWriter(new File("/Users/prasanthlade/Documents/ibm/"+category+"/responses/TrainQuanttime00"+i+".csv"));
			for(int j =0; j<tokens1.length; j++){
				int bin = calculateBin(Double.parseDouble(tokens1[j][1]), quartiles);
				pw.println(tokens1[j][0]+","+bin);
			}
			pw.close();
			pw = new PrintWriter(new File("/Users/prasanthlade/Documents/ibm/"+category+"/responses/TestQuanttime00"+i+".csv"));
			for(int j =0; j<tokens2.length; j++){
				int bin = calculateBin(Double.parseDouble(tokens2[j][1]), quartiles);
				pw.println(tokens2[j][0]+","+bin);
			}
			pw.close();
		}*/
	}

	public static double[][] matrixMultiply(double[][] A, double[][] B, boolean transposeA, boolean transposeB){
		/*Matrix m1= new Matrix(mat1);
		Matrix m2= new Matrix(mat2);
		return m1.times(m2).getArray();*/
		double[][] result= new double[A.length][B[0].length];
		if(transposeA){
			result= new double[A[0].length][B[0].length];
			result= Matrices.getArray(new DenseMatrix(A).transAmult(new DenseMatrix(B),new DenseMatrix(result)));
			//result= new weka.core.matrix.Matrix(A).transpose().times(new weka.core.matrix.Matrix(B)).getArray();
		}
		else
		{
			if(transposeB)
			{	result= new double[A.length][B.length];
			result= Matrices.getArray(new DenseMatrix(A).transBmult(new DenseMatrix(B),new DenseMatrix(result)));
			//result= new weka.core.matrix.Matrix(A).times(new weka.core.matrix.Matrix(B).transpose()).getArray();
			}
			else{
				result= Matrices.getArray(new DenseMatrix(A).mult(new DenseMatrix(B),new DenseMatrix(result)));
				//result= new weka.core.matrix.Matrix(A).times(new weka.core.matrix.Matrix(B)).getArray();
			}
		}
		return result;
	}

	public static double[][] matrixInverse(double[][] mat){

		double[][] inv= new double[mat.length][mat.length];
		/*try{
			//UTransposeInv= new Matrix(U).transpose().inverse().getArray();
			inv= new DenseDoubleAlgebra().inverse(new DenseColumnDoubleMatrix2D(mat)).toArray();
		}
		catch(RuntimeException rte){
			for(int j=0; j<mat.length; j++)
				mat[j][j]+=Math.random()/10000;
			//UTransposeInv= new Matrix(U).transpose().inverse().getArray();
			inv= new DenseDoubleAlgebra().inverse(new DenseColumnDoubleMatrix2D(mat)).toArray();
		}*/
		DenseMatrix eye= Matrices.identity(mat.length);
		inv= Matrices.getArray(new DenseMatrix(mat).solve(eye, new DenseMatrix(inv)));
		//inv= new weka.core.matrix.Matrix(mat).inverse().getArray();
		return inv;
	}

	public static double[][] matrixTranspose(double[][] mat){
		//return Matrices.getArray(new DenseMatrix(mat).transpose());
		//return new DenseDoubleAlgebra().transpose(new DenseColumnDoubleMatrix2D(mat)).toArray();
		//return new weka.core.matrix.Matrix(mat).transpose().getArray();
		double[][] trans= new double[mat[0].length][mat.length];
		for(int i=0; i<mat.length; i++)
			for(int j=0; j<mat[0].length;j++)
				trans[j][i]= mat[i][j];
		return trans;
	}

	public static double max(double[] vector){
		double maxVal= -1000000.0;
		for(double val: vector){
			if(val > maxVal)
				maxVal= val;
		}
		return maxVal;
	}

	/**
	 * Finds the highest N numbers in a vector and returns an array of indices
	 * @param vector
	 * @param N
	 * @return
	 */
	public static int[] max(double[] vector, int N){
		double[] newvec= new double[vector.length];
		for(int i=0; i<vector.length;i++)
			newvec[i]= vector[i];
		Sorting s= new Sorting(newvec, false);
		int[] allIndices= s.getIndex();
		int[] nIndices= new int[N];
		int count=0;
		for(int index=vector.length-1;index>=vector.length-N;index--){
			nIndices[count]= allIndices[index];
			count++;
		}
		return nIndices;
	}

	public static double mean(double[] vector){
		double sum= 0;
		for(double val: vector){
			sum+= val;
		}
		return sum/vector.length;
	}

	public static double[] mean(double[][] vector, int dim){
		double[] mean;
		if(dim==1){
			mean= new double[vector[0].length];
			for(int i=0; i<vector[0].length; i++){
				double sum=0;
				for(int j=0; j<vector.length; j++){
					sum+= vector[j][i];
				}
				mean[i]= sum/vector.length;
			}
		}
		else{
			mean= new double[vector.length];
			for(int i=0; i<vector.length; i++){
				double sum=0;
				for(int j=0; j<vector[0].length; j++){
					sum+= vector[i][j];
				}
				mean[i]= sum/vector[0].length;
			}
		}
		return mean;
	}

	public static double mean(ArrayList<Double> vector){
		double sum= 0;
		for(double val: vector){
			sum+= val;
		}
		return sum/vector.size();
	}

	public static double min(double[] vector){
		double minVal= 10000000.0;
		for(double val: vector){
			if(minVal > val)
				minVal= val;
		}
		return minVal;
	}

	public static int[] min(double[] vector, int N){
		double[] newvec= new double[vector.length];
		for(int i=0; i<vector.length;i++)
			newvec[i]= vector[i];
		Sorting s= new Sorting(newvec, false);
		int[] allIndices= s.getIndex();
		int[] nIndices= new int[N];
		int count=0;
		for(int index=vector.length-N;index<=vector.length-1;index++){
			nIndices[count]= allIndices[index];
			count++;
		}
		return nIndices;
	}

	public static int[] min(int[] vector, int N){
		double[] newvec= new double[vector.length];
		for(int i=0; i<vector.length;i++)
			newvec[i]= vector[i];
		Sorting s= new Sorting(newvec, false);
		int[] allIndices= s.getIndex();
		int[] nIndices= new int[N];
		int count=0;
		for(int index=vector.length-N;index<=vector.length-1;index++){
			nIndices[count]= allIndices[index];
			count++;
		}
		return nIndices;
	}

	public static double[] normalize(double[] vector){
		double sum= sum(vector);
		for(int i=0; i<vector.length; i++){
			if(sum!=0)
				vector[i]= vector[i]/sum;
			else
				vector[i]=0;
		}
		return vector;
	}

	public static double[][] normalizeFeatures(double[][] mat){
		double[][] newMat= new double[mat.length][mat[0].length];
		double[] minVals= new double[mat[0].length];
		double[] maxVals= new double[mat[0].length];
		for(int n=0; n<mat[0].length; n++){
			minVals[n]=Double.POSITIVE_INFINITY;
			maxVals[n]=Double.NEGATIVE_INFINITY;
		}
		for(int i=0; i<mat.length; i++){
			for(int n=0; n<mat[0].length; n++){
				if(mat[i][n]<minVals[n]){
					minVals[n]= mat[i][n];
				}
				if(mat[i][n]>maxVals[n]){
					maxVals[n]= mat[i][n];
				}
			}
		}
		//Utilities.printArray("maxVals ",maxVals);
		//Utilities.printArray("minVals ",minVals);
		for(int i=0; i<mat.length; i++){
			for(int n=0; n<mat[0].length; n++){
				if(maxVals[n] == minVals[n])
					newMat[i][n]= 1;
				else
					newMat[i][n] = (mat[i][n] - minVals[n]) / 
					(maxVals[n] - minVals[n]);
			}
		}
		/*
		 value = (vals[j] - m_MinArray[j]) / 
	      (m_MaxArray[j] - m_MinArray[j]) * m_Scale + m_Translation;
		 */
		return newMat;
	}

	/**
	 * Get a probability estimate for a value
	 *
	 * @param data the value to estimate the probability of
	 * @return the estimated probability of the supplied value
	 */
	public static double normalPDF(double data, double mean, double stdDev) {

		/*data = Math.rint(data / precision) * precision;
	    double zLower = (data - mean - (precision / 2)) / sigma;
	    double zUpper = (data - mean + (precision / 2)) / sigma;

	    double pLower = Statistics.normalProbability(zLower);
	    double pUpper = Statistics.normalProbability(zUpper);
	    return pUpper - pLower;*/
		return Math.exp(logNormalDens(data, mean, stdDev));
	}

	public static double precision(ArrayList<Double> vector){
		Collections.sort(vector);
		//System.out.println(vector);
		double precision= 0;
		int distinct=0;
		for(int i=1; i< vector.size(); i++){
			if(vector.get(i)!=vector.get(i-1)){
				precision+= (vector.get(i)-vector.get(i-1));
				distinct++;
			}
		}
		if(vector.size()==0 || vector.size()==1) 
			return 0; 
		else
			return precision/distinct;
	}

	public static void printArray(double[][] array){
		for(int i=0; i<array.length; i++){
			for(int j=0; j<array[i].length; j++)
				System.out.print(String.format("%.0f", array[i][j])+"\t");
			System.out.print("\n");
		}
	}

	public static void printArray(int[][] array){
		for(int i=0; i<array.length; i++){
			for(int j=0; j<array[i].length; j++)
				System.out.print(String.format("%d", array[i][j])+"\t");
			System.out.print("\n");
		}
	}

	public static void printArrayToFile(double[][] array,String filePath) {
		try{
			PrintWriter pw= new PrintWriter(new File(filePath));
			for(int i=0; i<array.length; i++){
				for(int j=0; j<array[i].length; j++)
					pw.print(String.format("%.4f", array[i][j])+"\t");
				pw.print("\n");
			}
			pw.close();
		}catch(IOException ioe){}

	}

	public static void printArray(int[] array){
		System.out.print("\n[ ");
		for(int i=0; i<array.length; i++){
			System.out.print(array[i]+" ");
		}
		System.out.print("]\n");
	}

	public static void printArray(double[] array){
		System.out.print("\n[ ");
		for(int i=0; i<array.length; i++){
			System.out.print(String.format("%.0f", array[i])+"\t");
		}
		System.out.print("]\n");
	}

	public static void printArray(String message,double[] array){
		DecimalFormat fmt= new DecimalFormat("#.####");
		System.out.print(message+": [ ");
		for(int i=0; i<array.length; i++){
			System.out.print(fmt.format(array[i])+" ");
		}
		System.out.print("]\n");
	}

	public static void printArray(String[] array){
		System.out.print("\n[ ");
		for(int i=0; i<array.length; i++){
			System.out.print(array[i]+" ");
		}
		System.out.print("]\n");
	}

	/**
	 * Reads a csv file into an array of Strings and also uses the info if there is a
	 * header or not
	 * @param fileName
	 * @param hasHeader
	 * @return
	 */
	public static String[][] readCSVFile(String fileName, boolean hasHeader) throws IOException{
		ArrayList<ArrayList<String>> dataList= new ArrayList<ArrayList<String>>();
		BufferedReader file= new BufferedReader(new FileReader(fileName));
		String newline= file.readLine();
		if(hasHeader)
			newline= file.readLine();
		while(newline!=null){
			String[] tokens= newline.split(",");
			ArrayList<String> temp= new ArrayList<String>();
			for(String token: tokens)
				temp.add(token);
			dataList.add(temp);
			newline= file.readLine();
		}
		file.close();
		String[][] data= new String[dataList.size()][dataList.get(0).size()];
		for(int row=0; row< dataList.size(); row++)
			for(int col=0; col< dataList.get(0).size(); col++)
				data[row][col]= dataList.get(row).get(col);
		return data;
	}

	public static double rmsError(double[] v1, double[] v2){
		double rms=0;
		for(int i=0; i<v1.length; i++){
			rms+= Math.pow(v1[i]-v2[i],2);
		}
		rms= rms/v1.length;
		rms= Math.sqrt(rms);
		return rms;
	}

	/**
	 * This method takes a multinomial (or bernoulli) probability distribution as an input and samples
	 * a value from the distribution 
	 * @param p
	 */
	public static int sampleFromDistribution(double[] p){
		int sample;
		// cumulate multinomial parameters
		for (int k = 1; k < p.length; k++) {
			p[k] += p[k - 1];
		}
		// scaled sample because of unnormalised p[]
		double u = Math.random()*p[p.length-1];
		for (sample = 0; sample < p.length; sample++) {
			if (u < p[sample])
				break;
		}
		return sample;
	}

	/**
	 * Scales an array of real values to the range [min,max]
	 * @param input
	 * @param min
	 * @param max
	 * @return
	 */
	public static double[] scaleData(double[] input,double min, double max){
		// find the max and min values of the array
		double originalMin= 10000,originalMax= 0;
		double[] output= new double[input.length];
		for(int i=0; i<input.length; i++){
			if(input[i]<originalMin){
				originalMin= input[i];
			}
			if(input[i]>originalMax){
				originalMax= input[i];
			}
			//System.out.println(input[i]);
		}
		//System.out.println(min+","+max+","+originalMin+","+originalMax);
		for(int i=0; i<input.length; i++){
			output[i]=(((max-min)/(originalMax-originalMin+0.00001))*(input[i]-originalMin))+min;
			//System.out.println(input[i]+","+output[i]);
		}
		return output;
	}

	public static double[] scaleData(double[] input,double originalMin, double originalMax, double targetMin, double targetMax){
		double[] output= new double[input.length];
		//System.out.println(min+","+max+","+originalMin+","+originalMax);
		for(int i=0; i<input.length; i++){
			output[i]=(((targetMax-targetMin)/(originalMax-originalMin+0.00001))*(input[i]-originalMin))+targetMin;
			//System.out.println(input[i]+","+output[i]);
		}
		return output;
	}

	public static void sizeOf(double[][] mat){
		System.out.println(mat.length+"x"+mat[0].length);
	}

	public static double[][] sqrt(double[][] mat){

		double[][] result= new double[mat.length][mat[0].length];
		for(int i=0; i<mat.length; i++)
			for(int j=0; j<mat[0].length; j++)
				result[i][j]= Math.sqrt(mat[i][j]);
		return result;
	}

	public static double[] stddeviation(double[][] v) {
		double[] var = variance(v);
		for (int i = 0; i < var.length; i++)
			var[i] = Math.sqrt(var[i]);
		return var;
	}

	public static double[] variance(double[][] v) {
		int m = v.length;
		int n = v[0].length;
		double[] var = new double[n];
		int degrees = (m - 1);
		double c;
		double s;
		for (int j = 0; j < n; j++) {
			c = 0;
			s = 0;
			for (int k = 0; k < m; k++)
				s += v[k][j];
			s = s / m;
			for (int k = 0; k < m; k++)
				c += (v[k][j] - s) * (v[k][j] - s);
			var[j] = c / degrees;
		}
		return var;
	}
	public static double sum(double[] vector){
		double sum= 0;
		for(double val: vector){
			sum+= val;
		}
		return sum;
	}

	public static double[] sum(double[][] vector, int dim){
		double[] sum;
		if(dim==1){
			sum= new double[vector[0].length];
			for(int i=0; i<vector[0].length; i++){
				sum[i]=0;
				for(int j=0; j<vector.length; j++){
					sum[i]+= vector[j][i];
				}
			}
		}
		else{
			sum= new double[vector.length];
			for(int i=0; i<vector.length; i++){
				sum[i]=0;
				for(int j=0; j<vector[0].length; j++){
					sum[i]+= vector[i][j];
				}
			}
		}
		return sum;
	}

	public static double variance(ArrayList<Double> vector){
		double var= 0;
		double mean= mean(vector);
		for(double val: vector){
			var+= (val-mean)*(val-mean);
		}
		return var/vector.size();
	}

	public static double weightedMean(double[] vector, double[] weights){
		double numer=0, denom=0;
		for(int i=0; i< vector.length; i++){
			numer+= weights[i]*vector[i];
			denom+= weights[i];
		}
		return numer/denom;
	}

	/**
	 * Given a set of ranges, this method quantizes an array of real numbers to respective bins
	 */
	 public static int[] vectorQuantizeUsingBinning(double[] vector, double[] ranges){
		 int[] quantizedVector = new int[vector.length];
		 for(int i=0; i<vector.length; i++){
			 quantizedVector[i] = calculateBin(vector[i], ranges);
		 }
		 return quantizedVector;
	 }

	 public static boolean writeCSVFile(String filePath, double[][] data, String[] header, boolean hasHeader, char delimiter){
		 try{
			 PrintWriter pw = new PrintWriter(new File(filePath));
			 if(hasHeader) {
				 assert header != null;
				 for(int i = 0; i < header.length; i++){
					 pw.print(header[i]);
					 if(i < header.length-1)
						 pw.print(delimiter);
				 }
				 pw.println();
			 }
			 else {
				 for(int row = 0; row < data.length; row++) {
					 for(int col = 0; col < data[0].length; col++){
						 pw.print(data[row][col]);
						 if(col < data[0].length-1)
							 pw.print(delimiter);
					 }
					 pw.println();
				 }
			 }
			 pw.close();
		 }
		 catch(IOException ioe){
			 ioe.printStackTrace();
			 return false;
		 }

		 return true;
	 }
}
