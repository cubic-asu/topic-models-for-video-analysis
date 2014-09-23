package edu.asu.cubic.dimensionality_reduction;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.EVD;
import no.uib.cipr.matrix.Matrices;
import no.uib.cipr.matrix.NotConvergedException;
import edu.asu.cubic.util.Utilities;
import Jama.EigenvalueDecomposition;

// static import of all array methods : linear algebra and statistics
import static org.math.array.LinearAlgebra.*;
import static org.math.array.StatisticSample.*;

/**
 * Copyright : BSD License
 * @author Yann RICHET
 */
public class PCA implements Serializable{

	private static final long serialVersionUID = 133353272589413744L;
	double[][] X; // initial datas : lines = events and columns = variables

	double[] meanX, stdevX;

	double[][] Z; // X centered reduced

	double[][] cov; // Z covariance matrix

	double[][] U; // projection matrix

	double[] info; // information matrix

	public PCA(){

	}

	public PCA(double[][] _X) throws NotConvergedException {
		X = new double[_X.length][_X[0].length];
		
		System.out.println("In PCA Constructor");
		// ignore data that contain NaN
		int count=0;
		for(int i=0; i<_X.length; i++){
			if(!Utilities.containsNaN(_X[i])){
				X[count]=_X[i];
				count++;
			}
		}
		stdevX = stddeviation(X);
		meanX = mean(X);
		//Utilities.printArray("std",stdevX);
		//Utilities.printArray("mean",meanX);
		System.out.println("Size of X is: "+X.length+"x"+X[0].length);
		Z = center_reduce(X);
		System.out.println("Done with center reduce, calculating covariance");
		
		//cov = covariance(Z);
		cov= parallelizedCovariance(Z);
		System.out.println("Calculated Covariance, Dimensions of cov matrix: "+ cov.length+"x"+cov[0].length);
		//Utilities.printArray(cov);
		// Write to file
		/*try{
			Properties requiredParameters= new Properties();
			requiredParameters.load(new FileInputStream("Parameters.properties"));
			String modelFolder= requiredParameters.getProperty("baseFolder").trim()+"/PCAModels";
			PrintWriter covarianceFile= new PrintWriter(new File(modelFolder+"/Covariance.csv"));
			for(int row=0; row<cov.length; row++){
				for(int col=0; col<cov[0].length; col++){
					covarianceFile.print(cov[row][col]+",");
				}
				covarianceFile.println();	
			}
			covarianceFile.close();
		}
		catch(IOException ioe){

		}*/
		
		System.out.println("Started Eigen Decomposition ");
		/*for(int j=0; j<cov.length; j++)
			cov[j][j]+=Math.random()/10000;
		EVD evd= new EVD(cov.length).factor(new DenseMatrix(cov));
		U= Matrices.getArray(evd.getLeftEigenvectors());
		info= evd.getRealEigenvalues();*/
		EigenvalueDecomposition e = eigen(cov);
		U = transpose(e.getV().getArray());
		info = e.getRealEigenvalues(); // covariance matrix is symetric, so only real eigenvalues...
		//Utilities.printArray("Eigen Values:", info);
		System.out.println("Finished Eigen Decomposition ");

	}

	public static double[][] covariance(double[][] v) {
		int m = v.length;
		int n = v[0].length;
		double[][] X = new double[n][n];
		int degrees = (m - 1);
		double c;
		double s1;
		double s2;
		for (int i = 0; i < n; i++) {
			double start= System.currentTimeMillis();
			for (int j = 0; j < n; j++) {
				c = 0;
				s1 = 0;
				s2 = 0;
				for (int k = 0; k < m; k++) {
					s1 += v[k][i];
					s2 += v[k][j];
				}
				s1 = s1 / m;
				s2 = s2 / m;
				for (int k = 0; k < m; k++)
					c += (v[k][i] - s1) * (v[k][j] - s2);
				X[i][j] = c / degrees;
			}
			double end= System.currentTimeMillis();
			System.out.println("Feature "+i+" Time Taken "+(end-start)/1000+" secs");
			//Utilities.printArray("", X[i]);
		}
		/*try{
			Properties requiredParameters= new Properties();
			requiredParameters.load(new FileInputStream("TopicFeaturesExtractionEvaluation.properties"));
			String modelFolder= requiredParameters.getProperty("ldaModelFolder").trim();
			PrintWriter covarianceFile= new PrintWriter(new File(modelFolder+"/Covariance.csv"));
			for(int row=0; row<X.length; row++){
				for(int col=0; col<X[0].length; col++){
					covarianceFile.print(X[row][col]+",");
				}
				covarianceFile.println();	
			}
			covarianceFile.close();
		}
		catch(IOException ioe){

		}*/

		return X;
	}

	public static double[][] parallelizedCovariance(double[][] v) {
		int m = v.length;
		int n = v[0].length;
		double[][] X = new double[n][n];
		int threads = Runtime.getRuntime().availableProcessors();
		System.out.println("# of threads "+threads);
		double start= System.currentTimeMillis();
		try
		{
			ExecutorService service = Executors.newFixedThreadPool(threads);
			List<Future<CovarianceOutput>> futures = new ArrayList<Future<CovarianceOutput>>();
			// Create a set of Mappers
			int[][] ranges= new int[threads/2][2];
			for(int i=0; i<threads/2; i++){
				ranges[i][0]= (i)*((n*2/threads))+1;
				ranges[i][1]= ranges[i][0]+((n*2/threads)-1);
				if(ranges[i][1]>n)
					ranges[i][1]= n;
			}
			if(ranges[threads/2-1][1]<n)
				ranges[threads/2-1][1]= n;
			int mapId=1;
			// Map
			for(int i=0; i<ranges.length;i++){
				for(int j=0; j<ranges.length;j++){
					Callable<CovarianceOutput> mapper= (new PCA()).new Mapper(ranges[i][0], ranges[i][1], ranges[j][0], ranges[j][1], mapId++, v);
					futures.add(service.submit(mapper));
				}
			}
			// Reduce
			for(Future<CovarianceOutput> futureoutput: futures){
				double[][] result= futureoutput.get().result;
				int fromRow= futureoutput.get().fromIndex1;
				int toRow= futureoutput.get().toIndex1;
				int fromCol= futureoutput.get().fromIndex2;
				int toCol= futureoutput.get().toIndex2;
				int rowIndex=0;
				for(int row=fromRow-1; row<toRow; row++){
					int colIndex=0;
					for(int col=fromCol-1; col<toCol; col++){
						X[row][col]= result[rowIndex][colIndex];
						colIndex++;
					}
					rowIndex++;
				}
			}
			service.shutdown();
			double end= System.currentTimeMillis();
			System.out.println("Total Time Taken "+(end-start)/1000+" secs");
			//Utilities.printArray("", X[i]);
			/*try{
				Properties requiredParameters= new Properties();
				requiredParameters.load(new FileInputStream("TopicFeaturesExtractionEvaluation.properties"));
				String modelFolder= requiredParameters.getProperty("ldaModelFolder").trim();
				PrintWriter covarianceFile= new PrintWriter(new File(modelFolder+"/Covariance.csv"));
				for(int row=0; row<X.length; row++){
					for(int col=0; col<X[0].length; col++){
						covarianceFile.print(X[row][col]+",");
					}
					covarianceFile.println();	
				}
				covarianceFile.close();
			}
			catch(IOException ioe){

			}*/
		}
		catch(ExecutionException ee){
		}
		catch(InterruptedException ee){
		}

		return X;
	}

	class CovarianceOutput {
		public int fromIndex1;
		public int toIndex1;
		public int fromIndex2;
		public int toIndex2;
		public double[][] result;
		CovarianceOutput(int f1, int t1, int f2, int t2, double[][] r){
			fromIndex1= f1;
			toIndex1= t1;
			fromIndex2= f2;
			toIndex2= t2;
			result= r;
		}
	}

	/**
	 * This Mapper class maps a subset of data to a processor (thread) and finds the 
	 * covariance of that section of features 
	 * @author prasanthl
	 *
	 */
	public class Mapper implements Callable<CovarianceOutput> {
		int fromIndex1;
		int toIndex1;
		int fromIndex2;
		int toIndex2;
		int threadIndex;
		double[][] data;
		Mapper(int f1, int t1, int f2, int t2, int tI, double[][] d){
			fromIndex1= f1;
			toIndex1= t1;
			fromIndex2= f2;
			toIndex2= t2;
			threadIndex= tI;
			data= d;
		}
		public CovarianceOutput call() throws Exception {
			double c;
			int m= data.length;
			int degrees = (m - 1);
			double s1;
			double s2;
			c = 0;
			s1 = 0;
			s2 = 0;
			System.out.println("Running Thread "+threadIndex+" From range: "+fromIndex1+" - "+toIndex1+" To range: "+fromIndex2+" - "+toIndex2);
			double[][] X= new double[toIndex1-fromIndex1+1][toIndex2-fromIndex2+1];
			double start= System.currentTimeMillis();
			int rowIndex=0;
			for (int i = fromIndex1-1; i < toIndex1; i++) {
				int colIndex=0;
				for (int j = fromIndex2-1; j < toIndex2; j++) {
					c = 0;
					s1 = 0;
					s2 = 0;
					for (int k = 0; k < m; k++) {
						s1 += data[k][i];
						s2 += data[k][j];
					}
					s1 = s1 / m;
					s2 = s2 / m;
					for (int k = 0; k < m; k++)
						c += (data[k][i] - s1) * (data[k][j] - s2);
					X[rowIndex][colIndex] = c / degrees;
					colIndex++;
				}
				rowIndex++;
				//Utilities.printArray("", X[i]);
			}
			double end= System.currentTimeMillis();
			System.out.println("Thread "+threadIndex+" Time Taken "+(end-start)/1000+" secs");
			return (new PCA()).new CovarianceOutput(fromIndex1, toIndex1, fromIndex2, toIndex2, X);
		}
	}

	// normalization of x relatively to X mean and standard deviation
	public double[][] center_reduce(double[][] x) {
		double[][] y = new double[x.length][x[0].length];
		for (int i = 0; i < y.length; i++)
			for (int j = 0; j < y[i].length; j++)
				y[i][j] = (x[i][j] - meanX[j]) / stdevX[j];
		return y;
	}

	// de-normalization of y relatively to X mean and standard deviation
	public double[] inv_center_reduce(double[] y) {
		return inv_center_reduce(new double[][] { y })[0];
	}

	// de-normalization of y relatively to X mean and standard deviation
	public double[][] inv_center_reduce(double[][] y) {
		double[][] x = new double[y.length][y[0].length];
		for (int i = 0; i < x.length; i++)
			for (int j = 0; j < x[i].length; j++)
				x[i][j] = (y[i][j] * stdevX[j]) + meanX[j];
		return x;
	}

	/**
	 * This method projects the original data to a subspace that covers a certain percent of total eigen space
	 */
	public double[][] project(double[][] originalData, double eigenPercent){

		// Find the indices of eigen values that cover this percentage
		double totalSum=0, dimensionSum=0;
		for(int i=0; i<info.length; i++){
			totalSum+=info[i];
			//System.out.println(info[i]);
		}
		int[] indices= new int[info.length];
		int dimension=0;
		for(int i=info.length-1; i>=0; i--){
			dimensionSum+=info[i];
			double percent= dimensionSum*100/totalSum; 
			if((dimensionSum*100/totalSum)>eigenPercent)
				break;
			indices[dimension]=i;
			dimension++;
		}
		System.out.println("The percent of eigen values in dimension "+dimension+" is "+ (dimensionSum*100/totalSum));
		System.out.println("U: "+ U.length+"x"+U[0].length);
		System.out.println("originalData: "+ originalData.length+"x"+originalData[0].length);
		//Utilities.printArray(indices);
		// inverse center reduce the eigen vectors
		double[][] projectedData= new double[originalData.length][dimension];
		for(int row=0; row<originalData.length; row++){
			for(int col=0; col<dimension;col++){
				projectedData[row][col]=0;
				//System.out.println("row: "+row+" col: "+col);
				for(int k=0; k<originalData[0].length; k++){
					//double[] normalizedVector= inv_center_reduce(U[indices[col]]);
					/*if(row==1 && col==0){
						System.out.println("originalData: "+ originalData.length+"x"+originalData[0].length);
						System.out.println("originalData[row][k] "+originalData[row][k]);
					}*/
					projectedData[row][col]+= originalData[row][k]*U[indices[col]][k];
				}
			}
		}
		return projectedData;
	}

	/**
	 * This method cleans up all unnecessary variables so that the object becomes much lighter
	 */
	public void cleanUpVariables(){
		Z= null;
		cov= null;
		// delete X
		X= null;
	}

}