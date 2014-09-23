package edu.asu.cubic.dimensionality_reduction;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

import Jama.Matrix;
import edu.asu.cubic.util.GMMVBBuilder;
import edu.asu.cubic.util.Utilities;

public class GMMVB implements Serializable, Cloneable{

	private static final long serialVersionUID = -147839487780687780L;

	/* Parameters */
	double prior_alpha;
	double[] alpha;
	double prior_kappa;
	double[] kappa;
	double prior_v;
	double[] v;
	int D;
	int numSamples;
	double[] prior_m;
	double[][] m;
	double[][] prior_M; // M= inv(W)
	double[][][] M;
	double tol=1e-20;
	int maxIter= 2000;
	double[][] R;
	double[][] logR;
	double[][] X;
	int K;
	ArrayList<Double> likelihoods;
	final int LAG = 5;
	String modelPath;
	String modelName;
	
	public GMMVB(double[][] docs, int K, double alpha, String mPath, String mName, double tol, int numIters ) {
		this.K= K;
		D= docs[0].length;
		prior_alpha= alpha;
		X= Utilities.matrixTranspose(Utilities.normalizeFeatures(docs));
		modelPath= mPath;
		modelName= mName;
		numSamples= X[0].length;
		this.tol= tol;
		maxIter= numIters;
	}

	public GMMVB(double[][] docs, GMMVB model,String mPath, String mName){
		this.K= model.K;
		D= docs[0].length;
		numSamples= docs.length;
		X= Utilities.matrixTranspose(Utilities.normalizeFeatures(docs));
		modelPath= mPath;
		modelName= mName;
		M= model.M;
		m= model.m;
		alpha= model.alpha;
		kappa= model.kappa;
		v= model.v;
		R= new double[numSamples][K];
	}
	
	public void initialize() throws IOException{
		likelihoods= new ArrayList<Double>();
		prior_kappa= 1;
		prior_m= Utilities.mean(X, 2);
		prior_M= new double[D][D];
		prior_M= Utilities.eye(D);
		R= new double[numSamples][K];
		prior_v= D+1;
		String[] tokens= modelName.split("_");
		// just retain the model name and number of topics
		if(new File(modelPath+"//"+tokens[0]+"_"+tokens[1]+"Init.txt").exists()){
			String[][] tokens1= Utilities.readCSVFile(modelPath+"//"+tokens[0]+"_"+tokens[1]+"Init.txt", false);
			for(int i=0; i<numSamples; i++)
				for(int j=0; j<K; j++)
					R[i][j]= Double.parseDouble(tokens1[i][j]);
			//Utilities.printArray(R);
		}
		else{
			ArrayList<Integer> indices= new ArrayList<Integer>();
			Random rnd= new Random();
			while(indices.size()!=K){
				int index= rnd.nextInt(numSamples);
				if(indices.isEmpty() || !indices.contains(index))
					indices.add(index);
			}
			//System.out.println(indices);
			double[][] m= new double[D][K];
			for(int i=0; i<K; i++){
				for(int j=0; j<D; j++){
					m[j][i]= X[j][indices.get(i)];
				}
			}
			//Utilities.printArray(m);
			double[][] temp= Utilities.matrixMultiply(m, X, true, false);
			//Utilities.printArray(temp);
			double[] temp1= new double[K];
			for(int i=0; i<K; i++){
				for(int j=0; j<D; j++){
					temp1[i]+= m[j][i]*m[j][i];
				}
				temp1[i]/= 2;
			}
			//Utilities.printArray("temp1", temp1);
			for(int i=0; i<numSamples; i++){
				for(int j=0; j<K; j++){
					R[i][j]=0;
				}
			}
			PrintWriter pw= new PrintWriter(new File(modelPath+"//"+tokens[0]+"_"+tokens[1]+"Init.txt"));
			for(int i=0; i<numSamples; i++){
				double[] temp2= new double[K];
				for(int j=0; j<K; j++){
					temp2[j]= temp[j][i]-temp1[j];
				}
				int index= Utilities.max(temp2, 1)[0];
				R[i][index]=1;
				for(int j=0; j<K; j++)
				{
					pw.print(R[i][j]);
					if(j<K-1)
						pw.print(",");
				}
				pw.println();
				/*Utilities.printArray("", temp2);
				Utilities.printArray("", R[i]);*/
			}
			pw.close();
		}
	}
	
	public void runEM() throws Exception {
		initialize();
		int t=0;
		boolean converged= false;
		likelihoods.add(Double.NEGATIVE_INFINITY);
		// check if the EM has to start from a certain point
		for(int i=1; i<maxIter;i++){
			if(new File(modelPath+"/"+modelName+"_"+(i)+".model").exists()){
				// if the model for this exists
				t= i;
				GMMVB model= (GMMVB)new ObjectInputStream(new FileInputStream(modelPath+"/"+modelName+"_"+(i)+".model")).readObject();
				m= model.m;
				if(Utilities.containsInfinity(m)|| Utilities.containsNaN(m)){
					System.err.println("Infinity or Nan in m: Quitting program");
					throw new Exception("Infinity or Nan in m: Quitting program");
				}
				M= model.M;
				for(int k=0; k<K; k++)
					if(Utilities.containsInfinity(M[k])|| Utilities.containsNaN(M[k])){
						System.err.println("Infinity or Nan in M: Quitting program");
						throw new Exception("Infinity or Nan in M: Quitting program");
					}
				R= model.R;
				if(Utilities.containsInfinity(R)|| Utilities.containsNaN(R)){
					System.err.println("Infinity or Nan in R: Quitting program");
					throw new Exception("Infinity or Nan in R: Quitting program");
				}
			}
		}
		while(t<maxIter){//!converged && 
			
			t++;
			System.out.println("Iteration: "+t);
			// M step
			System.out.println("M Step");
			mStep();
			System.out.println("E Step");
			// E step
			eStep();
			System.gc();
			try{
				// check if any of the variables have NaN or Infinity
				if(Utilities.containsInfinity(m)|| Utilities.containsNaN(m)){
					System.err.println("Infinity or Nan in m: Quitting program");
					throw new Exception("Infinity or Nan in m: Quitting program");
				}
				for(int k=0; k<K; k++)
					if(Utilities.containsInfinity(M[k])|| Utilities.containsNaN(M[k])){
						System.err.println("Infinity or Nan in M: Quitting program");
						throw new Exception("Infinity or Nan in M: Quitting program");
					}
				if(Utilities.containsInfinity(R)|| Utilities.containsNaN(R)){
					System.err.println("Infinity or Nan in R: Quitting program");
					throw new Exception("Infinity or Nan in R: Quitting program");
				}
				if(new File(modelPath+"/"+modelName+"_"+(t-1)+".model").exists()){
					// delete the model from prev iteration
					new File(modelPath+"/"+modelName+"_"+(t-1)+".model").delete();
				}
				// make a clone of this object
				GMMVB clonedModel= (GMMVB)this.clone();
				clonedModel.cleanUpVariables();
				// write the object to file
				String modelFilePath= modelPath+"/"+modelName+"_"+(t)+".model";
				ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelFilePath));
				oos.writeObject(clonedModel);
				oos.close();
			}
			catch(CloneNotSupportedException cnse)
			{cnse.printStackTrace();}
			//System.out.println("VBound Step");
			//likelihoods.add(vbound()/numSamples);
			//converged = Math.abs(likelihoods.get(t)-likelihoods.get(t-1)) < tol*Math.abs(likelihoods.get(t)) ? true: false;
			//System.out.println("Likelihood "+likelihoods.get(t)+", Converged "+converged);
		}
		if(new File(modelPath+"/"+modelName+"_"+(t)+".model").exists()){
			// delete the model from prev iteration
			new File(modelPath+"/"+modelName+"_"+(t)+".model").delete();
		}
		/*System.out.println("Total iterations: "+(t));
		int[] labels= new int[numSamples];
		for(int i=0; i<numSamples; i++){
			labels[i]= Utilities.max(R[i],1)[0];
			//System.out.println(labels[i]);
			Utilities.printArray("", R[i]);
		}*/
		
	}
	
	public void infer() {
		try{
		eStep();
		}
		catch(Exception e){
			System.err.println("Error in E Step");
			e.printStackTrace();
		}
	}
	
	public void mStep() throws Exception{
		double before, after;
		before= System.currentTimeMillis();
		double[] nk= Utilities.sum(R, 1);
		//Utilities.printArray("nk", nk);
		alpha= new double[K];
		for(int i=0; i<K; i++)
			alpha[i]= prior_alpha+nk[i];
		//Utilities.printArray(X);
		//System.out.println();
		//Utilities.printArray(R);
		//double[][] nxbar= new Matrix(X).times(new Matrix(R)).getArray();
		//double[][] nxbar= new DenseDoubleAlgebra().mult(new DenseColumnDoubleMatrix2D(X),new DenseColumnDoubleMatrix2D(R)).toArray();
		double[][] nxbar= Utilities.matrixMultiply(X, R, false, false);
		//Utilities.printArray(nxbar);
		kappa= new double[K];
		for(int i=0; i<K; i++)
			kappa[i]= prior_kappa+nk[i];
		m= new double[D][K] ;
				//(new Matrix(m,D).times(kappa)).plus(new Matrix(nxbar)).getArray();
		double[][] temp= new Matrix(prior_m,D).times(prior_kappa).getArray();
		double[][] temp1= nxbar;
		for(int i=0; i<D; i++)
			for(int j=0; j<K; j++){
				m[i][j]= temp1[i][j]+temp[i][0];
			}
		for(int i=0; i<K; i++)
			for(int j=0; j<m.length;j++)
				m[j][i]= m[j][i]/kappa[i];
		//Utilities.printArray(m);
		v= new double[K];
		for(int i=0; i<K; i++)
			v[i]= prior_v+nk[i];
		
		M= new double[K][D][D];
		double[][] sqrtR= Utilities.sqrt(R);
		//Utilities.printArray(sqrtR);
		double[][] xbar= new double[D][K];
		for(int i=0; i<K; i++)
			for(int j=0; j<D; j++){
				xbar[j][i]= nxbar[j][i]/nk[i];
				if(nk[i]==0)
					xbar[j][i]=0;
			}
		
		//Utilities.printArray(xbar);
		double[][] xbarm0= new double[D][K];//new Matrix(xbar).minus(new Matrix(m,D)).getArray();
		for(int i=0; i<D; i++)
			for(int j=0; j<K; j++){
				xbarm0[i][j]= xbar[i][j]-prior_m[i];
			}
		xbarm0= Utilities.matrixTranspose(xbarm0);
		//Utilities.printArray(xbarm0);
		double[] w= new double[K];
		for(int i=0; i<K; i++)
			w[i]= (prior_kappa*nk[i])/(prior_kappa+nk[i]);
		after= System.currentTimeMillis();
		//System.out.println("Done with Step1 in: "+(after-before)/1000+" secs");
		
		/*double[][] temp2= new double[1][D]; temp2[0]= xbarm0[0];
		before= System.currentTimeMillis();
		double[][] term3= new DenseColumnDoubleMatrix2D(temp2).zMult(new DenseColumnDoubleMatrix2D(temp2),new DenseColumnDoubleMatrix2D(D,D), 1,0,true,false).toArray();
		term3= new Matrix(term3).times(w[0]).getArray();
		after= System.currentTimeMillis();
		System.out.println("ParallelPort in: "+(after-before)/1000+" secs");
		before= System.currentTimeMillis();
		term3= (new Matrix(xbarm0[0],D).times(new Matrix(xbarm0[0],D).transpose())).times(w[0]).getArray();
		after= System.currentTimeMillis();
		System.out.println("Jama in: "+(after-before)/1000+" secs");*/
		//Utilities.printArray("w", w);
		for(int i=0; i<K; i++){
			System.out.println("\t Topic "+i);
			before= System.currentTimeMillis();
			double[][] Xs= new double[D][numSamples];
			for(int j=0; j<D; j++){
				for(int k=0; k<numSamples;k++){
					Xs[j][k]= X[j][k]-xbar[j][i];
				}
			}
			for(int j=0; j<D; j++){
				for(int k=0; k<numSamples;k++){
					Xs[j][k]= Xs[j][k]*sqrtR[k][i]; 
				}
			}
			after= System.currentTimeMillis();
			System.out.print("\tStep1 : "+(after-before)/1000+" secs");
			before= System.currentTimeMillis();
			//Utilities.printArray("Xs[0]",Xs[0]);
			//Utilities.printArray("Xs[1]",Xs[1]);
			/*double[][] term2= new Matrix(Xs).times(new Matrix(Xs).transpose()).getArray();
			after= System.currentTimeMillis();
			if(i==0)
				System.out.println("Done with Step3 Jama in: "+(after-before)/1000+" secs");*/
			before= System.currentTimeMillis();
			//double[][] term2= new DenseDoubleAlgebra().mult(new DenseColumnDoubleMatrix2D(Xs),new DenseDoubleAlgebra().transpose(new DenseColumnDoubleMatrix2D(Xs))).toArray();
			double[][] term2= Utilities.matrixMultiply(Xs, Xs, false, true);
			
			after= System.currentTimeMillis();
			System.out.println("\tStep2 : "+(after-before)/1000+" secs");
			//Utilities.printArray(term2);
			//double[][] term3= (new Matrix(xbarm0[i],D).times(new Matrix(xbarm0[i],D).transpose())).times(w[i]).getArray();
			double[][] temp2= new double[1][D]; temp2[0]= xbarm0[i];
			before= System.currentTimeMillis();
			//double[][] term3= new DenseDoubleAlgebra().mult(new DenseDoubleAlgebra().transpose(new DenseColumnDoubleMatrix2D(temp2)),new DenseColumnDoubleMatrix2D(temp2)).toArray();
			double[][] term3= Utilities.matrixMultiply(temp2, temp2, true, false);
			term3= new Matrix(term3).times(w[i]).getArray();
			after= System.currentTimeMillis();
			System.out.print("\tStep3 : "+(after-before)/1000+" secs");
			before= System.currentTimeMillis();
			//Utilities.printArray(term3);
			M[i]= (new Matrix(prior_M).plus(new Matrix(term2))).plus(new Matrix(term3)).getArray();
			if(Utilities.containsInfinity(term2)){
				throw new Exception("Infinity in M[i]: Quitting program");
			}
			else if(Utilities.containsNaN(term2)){
				throw new Exception("NaN in M[i]: Quitting program");
			}
			after= System.currentTimeMillis();
			System.out.println("\tStep4 : "+(after-before)/1000+" secs");
		}
		//System.out.println();
	}
	
	public void eStep() throws Exception{
		double[] logW= new double[K];
		double[][] EQ= new double[numSamples][K];
		double before, after;
		for(int i=0; i<K; i++){
			System.out.println("\t Topic "+i);
			 //Utilities.printArray(M[i]);
			//Utilities.printArray(new CholeskyDecomposition(new Matrix(M[i])).getL().getArray());
			before= System.currentTimeMillis();
			//double[][] U= new CholeskyDecomposition(new Matrix(M[i])).getL().transpose().getArray();
			//Utilities.printArrayToFile(M[i],"C:\\Prasanth\\Studies\\ASU\\CUbiC_Research\\AnnotatedDatasets\\AVEC2012\\TopicModels\\M.txt");
			double[][]U= Utilities.choleskyDecomposition(M[i]);
			while(Utilities.containsInfinity(U) || Utilities.containsNaN(U)){
				System.out.println("Infinity or Nan in U");
				for(int j=0; j<D; j++)
					M[i][j][j]+=Math.random()/10000;
				U= Utilities.choleskyDecomposition(M[i]);
			}
			//Utilities.printArray(U);
			//System.out.println(U.length+"x"+U[0].length);
			//double[][] U= new weka.core.matrix.CholeskyDecomposition(new weka.core.matrix.Matrix(M[i])).getL().transpose().getArray();
			after= System.currentTimeMillis();
			System.out.print("\tStep1 : "+(after-before)/1000+" secs");
			// The log|W_k| is calculated as 2*sum(log(diagvector(G))) where G is the upper triangle mat from cholesky decomposition
			// Please see the explanation here: http://blogs.sas.com/content/iml/2012/10/31/compute-the-log-determinant-of-a-matrix/
			double[] diag= new double[D];
			//Utilities.printArray(U);
			for(int j=0; j<D; j++)
				diag[j]= U[j][j]+Math.random()/10000;
			logW[i]= -2*Utilities.sum(Utilities.log(diag));
			//System.out.println("logW[i]:"+logW[i]);
			if(Double.isInfinite(logW[i])||Double.isNaN(logW[i])){
				//Utilities.printArray("diag", diag);
				//Utilities.printArray("log(diag)", Utilities.log(diag));
				throw new Exception("Infinity or Nan in logW: Quitting program");
			}
			double[][] temp= new double[D][numSamples];
			for(int j=0; j<D; j++){
				for(int k=0; k<numSamples; k++){
					temp[j][k]= X[j][k]-m[j][i];
				}
			}
			before= System.currentTimeMillis();
			double[][] UTransposeInv= Utilities.matrixInverse(Utilities.matrixTranspose(U));
			while(Utilities.containsInfinity(UTransposeInv) || Utilities.containsNaN(UTransposeInv)){
				System.out.println("Infinity or Nan in UTransposeInv");
				for(int j=0; j<D; j++)
					U[j][j]+=Math.random()/10000;
				UTransposeInv= Utilities.matrixInverse(Utilities.matrixTranspose(U));
			}
			//Utilities.printArray(UTransposeInv);
			//UTransposeInv= new Matrix(U).transpose().inverse().getArray();
			//UTransposeInv= new DenseDoubleAlgebra().inverse(new DenseDoubleAlgebra().transpose(new DenseColumnDoubleMatrix2D(U))).toArray();
			//double[][] Q= new Matrix(UTransposeInv).times(new Matrix(temp)).getArray();
			//double[][]Q= new DenseDoubleAlgebra().mult(new DenseColumnDoubleMatrix2D(UTransposeInv),new DenseColumnDoubleMatrix2D(temp)).toArray();
			double[][]Q= Utilities.matrixMultiply(UTransposeInv, temp, false, false);
			/*double[][] Q= new double[D][D];
			try{
			  Q= new weka.core.matrix.Matrix(U).transpose().inverse().times(new weka.core.matrix.Matrix(temp)).getArray();
			}
			catch(RuntimeException rte){
				Utilities.printArray(U);
			}*/
					//times(new Matrix(temp)).getArray();
			after= System.currentTimeMillis();
			System.out.println("\tStep2 : "+(after-before)/1000+" secs");
			//Utilities.printArray(Q);
			double[] temp1= new double[numSamples];
			for(int j=0; j<numSamples; j++){
				temp1[j]=0;
				for(int k=0; k<D; k++){
					temp1[j]+= Q[k][j]*Q[k][j];
				}
			}
			temp1= new Matrix(temp1,numSamples).times(v[i]).getColumnPackedCopy();
			for(int j=0; j<numSamples; j++){
				EQ[j][i]= D/kappa[i]+temp1[j];
			}
		}
		before= System.currentTimeMillis();
		if(Utilities.containsInfinity(EQ) || Utilities.containsNaN(EQ)){
			throw new Exception("Infinity or Nan in EQ: Quitting program");
		}
		if(Utilities.containsInfinity(logW) || Utilities.containsNaN(logW)){
			throw new Exception("Infinity or Nan in logW: Quitting program");
		}
		/*Utilities.printArray(EQ);
		Utilities.printArray("", logW);*/
		double[] ELogLambda= new double[K];
		for(int i=0; i<K; i++){
			for(int j=0; j<D; j++)
				ELogLambda[i]+= Utilities.digamma((v[i]-j)/2);
			ELogLambda[i]+= D*Math.log(2)+logW[i];
		}
		if(Utilities.containsInfinity(ELogLambda) || Utilities.containsNaN(ELogLambda)){
			throw new Exception("Infinity or Nan in ELogLambda: Quitting program");
		}
		//Utilities.printArray("",ELogLambda);
		double[] ELogPi= new double[K];
		for(int i=0; i<K; i++)
			ELogPi[i]= Utilities.digamma(alpha[i])-Utilities.digamma(Utilities.sum(alpha));
		//Utilities.printArray("",ELogPi);
		double[][] logRho= new double[numSamples][K];
		double[] temp= new Matrix(ELogPi,K).plus(new Matrix(ELogLambda,K).times(1/2.0)).getColumnPackedCopy();
		for(int i=0; i<numSamples; i++){
			for(int j=0; j<K; j++){
				logRho[i][j]= (temp[j])-(EQ[i][j]/2)-((D/2)*Math.log(2*Math.PI));
			}
		}
		//Utilities.printArray(logRho);
		logR= new double[numSamples][K];
		for(int i=0; i<numSamples; i++){
			temp= new double[K];
			for(int j=0; j<K; j++)
				temp[j]= logRho[i][j];
			double max= Utilities.max(temp);
			double logexpsum= 0;
			for(int j=0; j<K; j++){
				temp[j]-= max;
				logexpsum+=Math.exp(temp[j]);
			}
			//System.out.println(logexpsum);
			logexpsum= Math.log(logexpsum)+max;
			if(Double.isInfinite(logexpsum))
				logexpsum= max;
			for(int j=0; j<K; j++){
				logR[i][j]= logRho[i][j]-logexpsum;
			}
		}
		//Utilities.printArray(logR);
		if(Utilities.containsInfinity(logR) || Utilities.containsNaN(logR)){
			throw new Exception("Infinity or Nan in logR: Quitting program");
		}
		for(int i=0; i<numSamples; i++)
			for(int j=0; j<K; j++)
				R[i][j]= Math.exp(logR[i][j]);
		//Utilities.printArray(R);
		if(Utilities.containsInfinity(R) || Utilities.containsNaN(R)){
			throw new Exception("Infinity or Nan in R: Quitting program");
		}
		after= System.currentTimeMillis();
		System.out.println("\tStep3 in: "+(after-before)/1000+" secs");
	}
	
	public double vbound() throws Exception{
		double before, after;
		double totalLogLikelihood=0;
		double[] nk= Utilities.sum(R, 1);
		double[] ELogPi= new double[K];
		for(int i=0; i<K; i++)
			ELogPi[i]= Utilities.digamma(alpha[i])-Utilities.digamma(Utilities.sum(alpha));
		double Epz= 0;
		for(int i=0; i<K; i++)
			Epz+= nk[i]*ELogPi[i];
		//System.out.println(Epz);
		double Eqz= 0;
		for(int i=0; i<numSamples; i++)
			for(int j=0; j<K; j++)
				Eqz+= R[i][j]*logR[i][j];
		//System.out.println(Eqz);
		double logCalpha0= Utilities.LogGamma(K*prior_alpha)-K*Utilities.LogGamma(prior_alpha);
		//System.out.println(logCalpha0);
		double Eppi= logCalpha0+(prior_alpha-1)*Utilities.sum(ELogPi);
		//System.out.println(Eppi);
		double logCalpha= Utilities.LogGamma(Utilities.sum(alpha));
		for(int i=0; i<K; i++)
			logCalpha-=Utilities.LogGamma(alpha[i]);
		//System.out.println(logCalpha);
		double Eqpi= logCalpha;
		for(int i=0; i<K; i++)
			Eqpi+= (alpha[i]-1)*ELogPi[i];
		//System.out.println(Eqpi);
		totalLogLikelihood+= Epz-Eqz+Eppi-Eqpi;
		//System.out.println(totalLogLikelihood);
		//double[][] U0= new CholeskyDecomposition(new Matrix(prior_M)).getL().transpose().getArray();
		before= System.currentTimeMillis();
		double[][] U0= Utilities.choleskyDecomposition(prior_M);
		after= System.currentTimeMillis();
		System.out.println("Done with Step1 in: "+(after-before)/1000+" secs");
		double[][] sqrtR= Utilities.sqrt(R);
		//double[][] nxbar= Utilities.matrixMultiply(X, R);
		before= System.currentTimeMillis();
		//double[][] nxbar= new DenseDoubleAlgebra().mult(new DenseColumnDoubleMatrix2D(X),new DenseColumnDoubleMatrix2D(R)).toArray();
		double[][] nxbar= Utilities.matrixMultiply(X, R, false, false);
		after= System.currentTimeMillis();
		System.out.println("Done with Step2 in: "+(after-before)/1000+" secs");
		double[][] xbar= new double[D][K];
		for(int i=0; i<K; i++)
			for(int j=0; j<D; j++)
				xbar[j][i]= nxbar[j][i]/nk[i];
		//Utilities.printArray(xbar);
		double[] logW= new double[K];
		double[] trSW= new double[K];
		double[] trM0W= new double[K];
		double[] xbarmWxbarm= new double[K];
		double[] mm0Wmm0= new double[K];
		for(int i=0; i<K; i++){
			//double[][] U= new CholeskyDecomposition(new Matrix(M[i])).getL().transpose().getArray();
			before= System.currentTimeMillis();
			double[][] U= Utilities.choleskyDecomposition(M[i]) ;
			after= System.currentTimeMillis();
			if(i==0)
				System.out.println("Done with Step3 in: "+(after-before)/1000+" secs");
			before= System.currentTimeMillis();
			double[][] UInv= Utilities.matrixInverse(U);
			//UInv= new Matrix(U).inverse().getArray();
			after= System.currentTimeMillis();
			if(i==0)
				System.out.println("Done with Step4 in: "+(after-before)/1000+" secs");
			double[] diag= new double[D];
			for(int j=0; j<D; j++)
				diag[j]= U[j][j];
			logW[i]= -2*Utilities.sum(Utilities.log(diag));
			//System.out.print("xbar:");Utilities.printArray(xbar);
			double[][] Xs= new double[D][numSamples];
			for(int j=0; j<D; j++){
				for(int k=0; k<numSamples;k++){
					Xs[j][k]= X[j][k]-xbar[j][i];
				}
			}
			for(int j=0; j<D; j++){
				for(int k=0; k<numSamples;k++){
					Xs[j][k]= Xs[j][k]*sqrtR[k][i]; 
				}
			}
			//System.out.print("Xs:");Utilities.printArray(Xs);
			before= System.currentTimeMillis();
			//double[][] temp= new Matrix(Xs).times(new Matrix(Xs).transpose()).getArray();
			//double[][] temp= new DenseDoubleAlgebra().mult(new DenseColumnDoubleMatrix2D(Xs),new DenseDoubleAlgebra().transpose(new DenseColumnDoubleMatrix2D(Xs))).toArray();
			double[][] temp= Utilities.matrixMultiply(Xs, Xs, false, true);
			temp= new Matrix(temp).times(1/nk[i]).getArray();
			//System.out.print("temp:");Utilities.printArray(temp);
			after= System.currentTimeMillis();
			if(i==0)
				System.out.println("Done with Step5 in: "+(after-before)/1000+" secs");
			//double[][] V= new CholeskyDecomposition(new Matrix(temp)).getL().transpose().getArray();
			before= System.currentTimeMillis();
			double[][] V= Utilities.choleskyDecomposition(temp);
			/*while(Utilities.containsInfinity(V) || Utilities.containsNaN(V)){
				// add random noise to diagonal
				for(int j=0; j<D; j++)
					temp[j][j]+=0.001;
				V= new DenseDoubleAlgebra().transpose(new DenseDoubleCholeskyDecomposition(new DenseColumnDoubleMatrix2D(temp)).getL()).toArray();
			}*/
			after= System.currentTimeMillis();
			if(i==0)
				System.out.println("Done with Step6 in: "+(after-before)/1000+" secs");
			//Utilities.printArray(new Matrix(Xs).times(new Matrix(Xs).transpose()).times(1/nk[i]).getArray());
			//System.out.print("V:");Utilities.printArray(V);
			before= System.currentTimeMillis();
			//double[][] Q= new Matrix(V).times(new Matrix(UInv)).getArray();
			//double[][] Q= new DenseDoubleAlgebra().mult(new DenseColumnDoubleMatrix2D(V),new DenseColumnDoubleMatrix2D(UInv)).toArray();
			double[][] Q= Utilities.matrixMultiply(V, UInv, false, false);
			//System.out.print("Q:");Utilities.printArray(Q);
			for(int j=0; j<D; j++)
				for(int k=0; k<D; k++)
					trSW[i]+= Q[j][k]*Q[j][k];
			after= System.currentTimeMillis();
			if(i==0)
				System.out.println("Done with Step7 in: "+(after-before)/1000+" secs");
			//Q= new Matrix(U0).times(new Matrix(UInv)).getArray();
			before= System.currentTimeMillis();
			//Q= new DenseDoubleAlgebra().mult(new DenseColumnDoubleMatrix2D(U0),new DenseColumnDoubleMatrix2D(UInv)).toArray();
			Q= Utilities.matrixMultiply(U0, UInv, false, false);
			for(int j=0; j<D; j++)
				for(int k=0; k<D; k++)
					trM0W[i]+= Q[j][k]*Q[j][k];
			
			double[][] temp1= new double[D][1];
			for(int j=0; j<D; j++){
				temp1[j][0]= xbar[j][i]-m[j][i];
			}
			after= System.currentTimeMillis();
			if(i==0)
				System.out.println("Done with Step8 in: "+(after-before)/1000+" secs");
			before= System.currentTimeMillis();
			//double[][] UTransposeInv= new Matrix(U).transpose().inverse().getArray();
			double[][] UTransposeInv= Utilities.matrixInverse(Utilities.matrixTranspose(U));
			after= System.currentTimeMillis();
			if(i==0)
				System.out.println("Done with Step9 in: "+(after-before)/1000+" secs");
			before= System.currentTimeMillis();
			//double[] q= new Matrix(UTransposeInv).times(new Matrix(temp1,D)).getColumnPackedCopy();
			double[][] q= Utilities.matrixMultiply(UTransposeInv, temp1, false, false); 
			after= System.currentTimeMillis();
			if(i==0)
				System.out.println("Done with Step10 in: "+(after-before)/1000+" secs");
			//Utilities.printArray("",q);
			xbarmWxbarm[i]= 0;
			for(int j=0; j<D; j++)
				xbarmWxbarm[i]+= q[j][0]*q[j][0];
			
			temp1= new double[D][1];
			for(int j=0; j<D; j++){
				temp1[j][0]= m[j][i]-prior_m[j];
			}
			before= System.currentTimeMillis();
			//q= new Matrix(UTransposeInv).times(new Matrix(temp1,D)).getColumnPackedCopy();
			q= Utilities.matrixMultiply(UTransposeInv, temp1, false, false);
			after= System.currentTimeMillis();
			if(i==0)
				System.out.println("Done with Step11 in: "+(after-before)/1000+" secs");
			mm0Wmm0[i]= 0;
			for(int j=0; j<D; j++)
				mm0Wmm0[i]+= q[j][0]*q[j][0];
			
		}
		/*Utilities.printArray("logW",logW);
		Utilities.printArray("trSW", trSW);
		Utilities.printArray("trM0W", trM0W);
		Utilities.printArray("xbarmWxbarm", xbarmWxbarm);
		Utilities.printArray("mm0Wmm0", mm0Wmm0);*/
		double[] ELogLambda= new double[K];
		for(int i=0; i<K; i++){
			for(int j=0; j<D; j++)
				ELogLambda[i]+= Utilities.digamma((v[i]-j)/2);
			ELogLambda[i]+= D*Math.log(2)+logW[i];
		}
		//Utilities.printArray("ELogLambda", ELogLambda);
		double[] temp= new double[K];
		for(int i=0; i<K; i++){
			temp[i]= (D*Math.log(prior_kappa/(2*Math.PI))) +
					 ELogLambda[i] -
					 (D*prior_kappa/kappa[i]) -
					 (prior_kappa*v[i]*mm0Wmm0[i]);
		}
		double Epmu= Utilities.sum(temp)/2;
		//System.out.println("Epmu: "+Epmu);
		temp = new double[D];
		for(int i=0; i<D; i++)
			for(int j=0; j<D; j++)
				if(i==j)
					temp[i]=U0[i][j];
		double logB0= prior_v*Utilities.sum(Utilities.log(temp)) -
				      0.5*prior_v*D*Math.log(2) -
				      Utilities.logmvgamma(0.5*prior_v, D);
		//System.out.println("logB0: "+logB0);
		double EpLambda= K*logB0+0.5*(prior_v-D-1)*Utilities.sum(ELogLambda);
		for(int i=0; i<K; i++)
			EpLambda-= 0.5*v[i]*trM0W[i];
		//System.out.println("EpLambda: "+EpLambda);
		temp= new double[K];
		for(int i=0; i<K; i++)
			temp[i]= ELogLambda[i]+(D*Math.log(kappa[i]/(2*Math.PI)));
		double Eqmu= 0.5*Utilities.sum(temp)- (0.5*D*K);
		//System.out.println("Eqmu: "+Eqmu);
		double[] logB= new double[K];
		for(int i=0; i<K; i++)
			logB[i] = -v[i]*(logW[i]+(D*Math.log(2)))/2
			          -Utilities.logmvgamma(0.5*v[i], D);
		//Utilities.printArray("logB", logB);
		temp = new double[K];
		for(int i=0; i<K; i++)
			temp[i]= (v[i]-D-1)*ELogLambda[i]-(v[i]*D);
		double EqLambda= 0.5*Utilities.sum(temp)+Utilities.sum(logB);
		//System.out.println("EqLambda: "+EqLambda);
		double EpX = 0;
		for(int i=0; i<K; i++){
			EpX+= nk[i]*(ELogLambda[i] - (D/kappa[i])
					        -(v[i]*trSW[i]) - (v[i]*xbarmWxbarm[i]) - (D*Math.log(2*Math.PI))
					        );
		}
		EpX*= 0.5;
		//System.out.println("EpX: "+EpX);
		totalLogLikelihood+= Epmu-Eqmu+EpLambda-EqLambda+EpX;
		//System.out.println(totalLogLikelihood);
		return totalLogLikelihood;
	}
	
	public double[][] getR(){
		return R;
	}
	
	public double[][][] getInverseCov(){
		return M;
	}
	
	public double[][] getMean(){
		return m;
	}
	
	public double[] getAlpha(){
		return alpha;
	}
	
	public double[] getKappa(){
		return kappa;
	}
	
	public double[] getV(){
		return v;
	}
	
	public int getK(){
		return K;
	}
	/**
	 * This method cleans up all unnecessary variables so that the object becomes much lighter
	 */
	public void cleanUpVariables(){
		X= null;
		prior_M= null;
		logR= null;
	}
	
	public static void main(String[] args) throws Exception{
		String[][] tokens= Utilities.readCSVFile("C:\\Prasanth\\Studies\\ASU\\CUbiC_Research\\MyResearch\\Software\\VariationBayesForGMMMatlab\\data.csv", true);
		double[][] features= new double[tokens.length][tokens[0].length];
		for(int i=0; i<tokens.length; i++)
			for(int j=0; j<tokens[0].length; j++){
				features[i][j]= Double.parseDouble(tokens[i][j]);
			}
		GMMVB object= new GMMVB(features, 5, 5,"C:\\Prasanth\\Studies\\ASU\\CUbiC_Research\\MyResearch\\Software\\VariationBayesForGMMMatlab\\", "GMMVB_5",1e-20, 700);
		object.runEM();
		//GMMVBBuilder.extractGMMFeatures("Parameters.properties");
	}
	
}
