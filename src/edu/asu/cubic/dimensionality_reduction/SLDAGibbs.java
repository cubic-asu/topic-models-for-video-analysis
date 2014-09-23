package edu.asu.cubic.dimensionality_reduction;


import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.io.Serializable;

import moa.streams.ArffFileStream;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import edu.asu.cubic.util.Utilities;

public class SLDAGibbs implements Serializable, Cloneable{

	private static final long serialVersionUID = 3483833086310775381L;

	String phase;// training or unseen
	/**
	 * document data (term lists)
	 */
	int[][] documents;
	/**
	 * labels/annotations for documents
	 */
	double[] y;
	/**
	 * The normalized topic distributions per document 
	 */
	double[][] z_bar;
	/**
	 * The regression coefficients one per each topic
	 */
	double[] b;
	/**
	 * The value indicate the type of regression to be used 1 for linear regression (
	 * for real output) and 2 for logistic regression (for binary output)
	 */
	int modelType;
	/**
	 * Learning rate For Streaming Linear Regression (needed by Weka MOA) 
	 */
	double learningRate;
	/**
	 * vocabulary size
	 */
	int V;
	/**
	 * number of topics
	 */
	int K;
	/**
	 * Dirichlet parameter (document--topic associations)
	 */
	double[] alpha;
	/**
	 * Dirichlet parameter (topic--term associations)
	 */
	double[][] beta;
	/**
	 * Sum of all Dirichlet parameters (document--topic associations)
	 */
	double alphaSum=0;
	/**
	 * Sum of all Dirichlet parameters (topic--term associations)
	 */
	double[] betaSum;
	/**
	 * topic assignments for each word.
	 */
	int z[][];
	int initialZs[][];
	/**
	 * cwt[i][j] number of instances of word i (term?) assigned to topic j.
	 */
	int[][] nw;
	/**
	 * na[i][j] number of words in document i assigned to topic j.
	 */
	int[][] nd;
	/**
	 * nwsum[j] total number of words assigned to topic j.
	 */
	int[] nwSum;
	/**
	 * nasum[i] total number of words in document i.
	 */
	int[] ndSum;
	/**
	 * cumulative statistics of theta
	 */
	double[][] thetasum;
	/**
	 * cumulative statistics of phi
	 */
	double[][] phisum;
	/**
	 * Likelihood of data at the end of each Gibbs sampling iteration
	 */
	double[] likelihoods;
	/**
	 * size of statistics
	 */
	int numstats;
	/**
	 * sampling lag (?)
	 */
	int THIN_INTERVAL = 10;
	/**
	 * burn-in period
	 */
	int BURN_IN = 10;
	/**
	 * max iterations, these will be used in the E-step
	 */
	int ITERATIONS = 20;
	/**
	 * Number of iterations in M-step
	 */
	int m_iterations = 5;
	/**
	 * sample lag (if -1 only one sample taken)
	 */
	int SAMPLE_LAG;
	/**
	 * nwTrain[i][j] number of training instances of word i (term?) assigned to topic j.
	 */
	int[][] nwTrain;
	/**
	 * nwsumTrain[j] total number of words in training data assigned to topic j.
	 */
	int[] nwsumTrain;

	int dispcol = 0;
	// The name with which this model will be dumped to disk every 100 iterations
	String modelPath; 
	String baseModelName;

	public SLDAGibbs(int[][] documents, double[] annotations, int V, int K, double[] alpha, double[][] beta, String mPath, String bmName, double lRate) {
		this.documents = documents;
		this.V = V;
		this.K = K;
		this.alpha = alpha;
		this.beta = beta;
		betaSum= new double[K];
		for(int i=0; i<K; i++)
			alphaSum+=alpha[i];
		for(int i=0; i<K; i++)
			betaSum[i]=0;
		for(int i=0; i<K; i++)
			for(int j=0; j<V; j++)
				betaSum[i]+=beta[i][j];
		// init sampler statistics
		thetasum = new double[documents.length][K];
		phisum = new double[K][V];
		numstats = 0;
		phase= "training";
		modelPath= mPath;
		baseModelName= bmName;
		// slda specific assignments
		this.y = annotations;
		int M= documents.length;
		z_bar= new double[M][K];
		b= new double[K+1]; // one coeff per topic plus and intercept
		modelType= 1; // setting it to linear model (not handling logistic at this point of time)
		learningRate= lRate;
	}

	public SLDAGibbs(int[][] documents, SLDAGibbs model, int K) {
		this.documents = documents;
		this.V = model.V;
		this.K= K;
		this.alpha = model.alpha;
		this.beta = model.beta;
		this.alphaSum = model.alphaSum;
		this.betaSum = model.betaSum;
		this.nwTrain = model.nw;
		this.nwsumTrain = model.nwSum;
		phase= "unseen";
		thetasum = new double[documents.length][K];
		phisum = new double[K][V];
		numstats=0;
	}

	public void initialStateForTraining(){
		//System.out.println("In LDATrain initialState ");
		int M = documents.length;
		// initialise count variables.
		nw = new int[V][K];
		nd = new int[M][K];
		nwSum = new int[K];
		ndSum = new int[M];
		// The z_i are are initialised to values in [1,K] to determine the
		// initial state of the Markov chain.
		//System.out.println(K+","+M);
		z = new int[M][];
		//System.out.println(documents.length);
		for (int m = 0; m < M; m++) {
			//System.out.println("Document: "+(m+1));
			int N = documents[m].length;
			//System.out.println("  "+N);
			z[m] = new int[N];
			for (int n = 0; n < N; n++) {
				int topic = (int) (Math.random() * K);
				z[m][n] = topic;
				//System.out.println("m: "+m+" n: "+n);
				// number of instances of word i assigned to topic j
				nw[documents[m][n]][topic]++;
				// number of words in document i assigned to topic j.
				nd[m][topic]++;
				// total number of words assigned to topic j.
				nwSum[topic]++;
			}
			// total number of words in document i
			ndSum[m] = N;
		}
		initialZs= z;
	}

	public void initialStateForUnseenDocs() {
		//System.out.println("In LDATest initialState ");
		int M = documents.length;
		// initialise count variables.
		nw = new int[V][K];
		nd = new int[M][K];
		nwSum = new int[K];
		ndSum = new int[M];
		// The z_i are are initialised to values in [1,K] to determine the
		// initial state of the Markov chain.
		// for each term assign the topic to which this term has been assigned
		// max times in the training data
		z = new int[M][];
		for (int m = 0; m < M; m++) {
			int N = documents[m].length;
			z[m] = new int[N];
			for (int n = 0; n < N; n++) {
				int topic=0;;
				int maxCount=0;
				// find the maxCount of this term and the topic that has maxCount
				for(int k=0; k<K; k++){
					if(nwTrain[documents[m][n]][k]> maxCount){
						maxCount= nwTrain[documents[m][n]][k];
						topic= k;
					}
				}
				// assign a random topic if this word does not occur in the training data
				if(maxCount==0)
					topic = (int) (Math.random() * K);
				z[m][n] = topic;
				// number of instances of word i assigned to topic j
				nw[documents[m][n]][topic]++;
				// number of words in document i assigned to topic j.
				nd[m][topic]++;
				// total number of words assigned to topic j.
				nwSum[topic]++;
			}
			// total number of words in document i
			ndSum[m] = N;
		}
	}

	public void gibbs() throws IOException {

		// Initialize all the regression coefficients to either -1 or 1 (as in SLDA R package by Chang)
		if(phase.equalsIgnoreCase("training")){
			double[] p= new double[2]; p[0]=0.5;p[1]=0.5;
			int sampledValue;
			for(int kk=0; kk<=K; kk++){
				sampledValue= Utilities.sampleFromDistribution(p);
				if(sampledValue==0)
					b[kk]=-1;
				else
					b[kk]=1;
			}

			likelihoods= new double[m_iterations];
			System.out.println("M Iteration: "+ 0);
			// E Step
			sampleZBar(false);
			// M Step
			estimateCoeffs();
			Utilities.printArray("Coeffs ", b);
			for(int i=0; i< m_iterations-1; i++){
				System.out.println("M Iteration: "+ (i+1));
				// E Step
				if(i== m_iterations-2)
					sampleZBar(true);
				else
					sampleZBar(false);
				// M Step
				estimateCoeffs();
				// dump the current model to disk if it is a training phase
				try{
					// make a clone of this object
					SLDAGibbs clonedModel= (SLDAGibbs)this.clone();
					clonedModel.cleanUpVariables();
					// write the object to file
					String modelFilePath= modelPath+"/"+baseModelName+"_"+(i+2)+"_"+ITERATIONS+".model";
					ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelFilePath));
					oos.writeObject(clonedModel);
					oos.close();
				}
				catch(CloneNotSupportedException cnse)
				{cnse.printStackTrace();}
				likelihoods[i]= modelLogLikelihood();
				Utilities.printArray("Coeffs ", b);
			}
		}
		else{ // if the phase is not training i.e unseen data
			for (int i = 0; i < ITERATIONS; i++) {
				System.out.println(" Iteration: "+(i+1));
				// for all z_i
				for (int m = 0; m < z.length; m++) {
					//System.out.println();
					for (int n = 0; n < z[m].length; n++) {
						//System.out.print("m: "+m+" n: "+documents[m][n]);
						// (z_i = z[m][n])
						// sample from p(z_i|z_-i, w)
						int topic = sampleZ(m, n);
						z[m][n] = topic;
					}
				}
				// get statistics after burn-in
				if ((i > BURN_IN) && (SAMPLE_LAG > 0) && (i % SAMPLE_LAG == 0)) {
					updateParams();
				}
			}
		}
	}

	public void sampleZBar(boolean updateParams) throws IOException {

		// Initialize all the regression coefficients to either -1 or 1 (as in SLDA R package by Chang)
		for (int i = 0; i < ITERATIONS; i++) {
			System.out.println("\tE Iteration: "+(i+1));
			// Sample all the topics
			for (int m = 0; m < z.length; m++) {
				//System.out.println();
				for (int n = 0; n < z[m].length; n++) {
					//System.out.print("m: "+m+" n: "+documents[m][n]);
					// (z_i = z[m][n])
					// sample from p(z_i|z_-i, w)
					int topic = sampleZ(m, n);
					z[m][n] = topic;
				}
				/*if(m==10)
					Utilities.printArray(""+i, z_bar[m]);*/
			}
			if(updateParams)
				updateParams();
			//likelihoods[i]= modelLogLikelihood();
		}

	}

	/**
	 * Sample a topic z_i from the conditional distribution: p(z_i = j |
	 * z_-i, w) = (n_-i,j(w_i) + beta)/(n_-i,j(.) + W * beta) * (n_-i,j(d_i) +
	 * alpha)/(n_-i,.(d_i) + K * alpha) * exp (2 (b_k/N_d)(y_d- b^T*z_d^-i) - (b_k/N_d)^2)
	 * Look into Eq D.24 from Jonathan Chang's SLDA equations.
	 * where N_d - total # of words in document d
	 * @param m
	 *            document
	 * @param n
	 *            word
	 */
	private int sampleZ(int m, int n) {

		// remove z_i from the count variables
		int topic = z[m][n];
		nw[documents[m][n]][topic]--;
		nd[m][topic]--;
		nwSum[topic]--;
		ndSum[m]--;
		double bz_bar=0;
		if(phase.equalsIgnoreCase("training")){
			// Normalize the existing topic assignments to this document
			for(int k=0; k<K; k++){
				z_bar[m][k]= ((double)nd[m][k])/ndSum[m];
			}
			// calculate b_bar*z_bar which is the current estimate of actual y[m]
			for(int k=0; k<K; k++){
				bz_bar+= b[k]*z_bar[m][k];
			}
		}
		double[] p = new double[K];
		boolean infiniteFlag= false; // some times the coeffs may have huge values and p may become infinite
		for (int k = 0; k < K; k++) {
			try{
				p[k] = (nw[documents[m][n]][k] + beta[k][documents[m][n]]) / (nwSum[k] + betaSum[k])
						* (nd[m][k] + alpha[k]) / (ndSum[m] + alphaSum);
				// multiply with the extra factor from the labels if the phase is training
				// if the phase is not training then its just plain LDA
				if(phase.equalsIgnoreCase("training")){
					double change= b[k]/ndSum[m];
					p[k]*= Math.exp( change*((y[m]-bz_bar)- (change/2))/0.25 );
					if(Double.isInfinite(p[k])){
						infiniteFlag= true;
						break;
					}
				}
			}catch(IndexOutOfBoundsException iobe){
				System.out.println("Document: "+m+" Word " +n+": "+ documents[m][n]+" Topic: "+k);
				System.out.println("nw[documents[m][n]][k]: "+ nw[documents[m][n]][k]);
				System.out.println("nwSum[k]: "+ nwSum[k]);
				System.out.println("nd[m][k]: "+nd[m][k]);
				System.out.println(y[m]);
				iobe.printStackTrace();
				throw iobe;
			}
		}
		if(infiniteFlag) // if p becomes infinite for some reason sample topic randomly
		{
			topic = (int) (Math.random() * K);
			System.out.println("Sampled randomly");
		}
		else
			topic= Utilities.sampleFromDistribution(p);
		// add newly estimated z_i to count variables
		nw[documents[m][n]][topic]++;
		nd[m][topic]++;
		nwSum[topic]++;
		ndSum[m]++;
		if(phase.equalsIgnoreCase("training")){
			// Normalize the existing topic assignments to this document
			for(int k=0; k<K; k++){
				z_bar[m][k]= ((double)nd[m][k])/ndSum[m];
			}
		}
		return topic;
	}

	/**
	 * Using the current distribution of topics for all documents the method estimate the
	 * regression coefficients (linear or logistic depending on the model type)
	 */
	public void estimateCoeffs() {
		// Write z_bar and y to an arff file format
		String randStr= Utilities.generateRandomString(15);
		String filename= modelPath+"/"+randStr+".arff";
		try{
			PrintWriter toFile= new PrintWriter(new File(filename));
			toFile.println("@relation "+randStr+"\n");
			for(int kk=0; kk<K; kk++){
				toFile.println("@attribute t"+kk+" NUMERIC");
			}
			toFile.println("@attribute class NUMERIC");toFile.println("@data");
			int M= documents.length;
			for(int m=0; m<M; m++){
				//double[] scaledz_bar= Utilities.scaleData(z_bar[m], 1, 100, true) ;
				//Utilities.printArray("", z_bar[m]);
				//Utilities.printArray("", scaledz_bar);
				for(int kk=0; kk<K; kk++){
					//toFile.print(String.format("%.3f,", Utilities.scalePoint(z_bar[m][kk], 0, 1, 1, 100)));
					toFile.print(String.format("%.3f,", z_bar[m][kk]));
					//toFile.print(String.format("%.3f,", scaledz_bar[kk]));
				}
				toFile.println(y[m]);
			}
			toFile.close();
		}
		catch(IOException ioe){
			System.err.println("Error caught in SLDA estimateCoeffs()");
			ioe.printStackTrace();
		}
		boolean flag= false;
		ArffFileStream dataStream= null;
		while(!flag){
			try{
				dataStream= new ArffFileStream(filename,K+1);
				flag= true;
			}
			catch(Exception e){

			}
		}
		dataStream.prepareForUse();
		try{
			LinearRegression regressionModel= new LinearRegression();
			ArffReader reader= new ArffReader(new FileReader(filename));
			Instances trainingInstances= reader.getData();
			trainingInstances.setClassIndex(K);
			regressionModel.buildClassifier(trainingInstances);
			//System.out.println(regressionModel);
			// get the learnt weights and assign them to b
			double[] coeffs= regressionModel.coefficients();
			for(int kk=0; kk<K; kk++)
				b[kk]= coeffs[kk];
			b[K]= coeffs[K+1];// this is K+1 because the K+1 coefffs are for K topics + 1 class 
                              //and then the intercept is in K+1 position
			// delete the temporary arff file
			new File(filename).delete();
		}
		catch(Exception e){
			System.err.println("Error caught in estimateCoeffs in SLDA");
			e.printStackTrace();
		}
	}

	/**
	 * @param iterations
	 *            number of total gibbs sampling terations
	 * @param m_iterations
	 *            # of times the regression coefficients have to be estimated           
	 */
	public void setIterations(int e_iterations, int m_iterations, int burnin) {
		ITERATIONS = e_iterations;
		this.m_iterations= m_iterations;
		BURN_IN= burnin;
	}

	/**
	 * Add to the statistics the values of theta and phi for the current state.
	 */
	private void updateParams() {
		for (int m = 0; m < documents.length; m++) {
			for (int k = 0; k < K; k++) {
				thetasum[m][k] += (nd[m][k] + alpha[k]) / (ndSum[m] + alphaSum);
			}
		}
		for (int k = 0; k < K; k++) {
			for (int w = 0; w < V; w++) {
				phisum[k][w] += (nw[w][k] + beta[k][w]) / (nwSum[k] + betaSum[k]);
			}
		}
		numstats++;
	}

	public static double logGammaStirling(double z) {
		int shift = 0;
		while (z < 2) {
			z++;
			shift++;
		}
		double result = (Math.log(2 * Math.PI) / 2) + (z - 0.5) * Math.log(z) - z +
				1/(12 * z) - 1 / (360 * z * z * z) + 1 / (1260 * z * z * z * z * z);
		while (shift > 0) {
			shift--;
			z--;
			result -= Math.log(z);
		}
		return result;
	}

	public double modelLogLikelihood() {
		double logLikelihood = 0.0;
		// The likelihood of the model is a combination of a 
		// Dirichlet-multinomial for the words in each topic
		// and a Dirichlet-multinomial for the topics in each
		// document.
		// The likelihood function of a dirichlet multinomial is
		//	 Gamma( sum_i alpha_i )	 prod_i Gamma( alpha_i + N_i )
		//	prod_i Gamma( alpha_i )	  Gamma( sum_i (alpha_i + N_i) )
		// So the log likelihood is 
		//	logGamma ( sum_i alpha_i ) - logGamma ( sum_i (alpha_i + N_i) ) + 
		//	 sum_i [ logGamma( alpha_i + N_i) - logGamma( alpha_i ) ]
		// Do the documents first
		double[] topicLogGammas = new double[K];
		for (int topic=0; topic < K; topic++) {
			topicLogGammas[ topic ] = logGammaStirling( alpha[topic] );
		}
		for (int doc=0; doc < documents.length; doc++) {
			for (int topic=0; topic < K; topic++) {
				if (nd[doc][topic] > 0) {
					logLikelihood += (logGammaStirling(alpha[topic] + nd[doc][topic]) -
							topicLogGammas[ topic ]);
				}
			}
			// subtract the (count + parameter) sum term
			logLikelihood -= logGammaStirling(alphaSum + documents[doc].length);
		}
		// add the parameter sum term
		logLikelihood += documents.length * logGammaStirling(alphaSum);
		// And the topics
		double[][] termLogGammas = new double[K][V];
		for (int topic=0; topic < K; topic++) {
			for (int term=0; term < V; term++) {
				termLogGammas[topic][ term ] = logGammaStirling( beta[topic][term] );
			}
		}
		for (int topic=0; topic < K; topic++) {
			for (int term=0; term < V; term++) {
				if (nw[term][topic] > 0) {
					logLikelihood += (logGammaStirling(beta[topic][term] + nw[term][topic]) -
							termLogGammas[topic][ term ]);
				}
			}
			// subtract the (count + parameter) sum term
			logLikelihood -= logGammaStirling(betaSum[topic] + nwSum[topic]);
		}
		// add the parameter sum term
		for (int topic=0; topic < K; topic++) 
			logLikelihood +=  logGammaStirling(betaSum[topic]);
		return logLikelihood;
	}
	
	public void configure(int iterations, int burnIn, int thinInterval,
			int sampleLag) {
		ITERATIONS = iterations;
		BURN_IN = burnIn;
		THIN_INTERVAL = thinInterval;
		SAMPLE_LAG = sampleLag;
	}

	/**
	 * This method cleans up all unnecessary variables so that the object becomes much lighter
	 */
	public void cleanUpVariables(){
		phisum= null;
		thetasum= null;
		z= null;
		documents= null;
		initialZs= null;
		z_bar= null;
	}

	/**
	 * Retrieve estimated document--topic associations. If sample lag > 0 then
	 * the mean value of all sampled statistics for theta[][] is taken.
	 * 
	 * @return theta multinomial mixture of document topics (M x K)
	 */
	public double[][] getTheta() {
		double[][] theta = new double[documents.length][K];

		if (SAMPLE_LAG > 0) {
			for (int m = 0; m < documents.length; m++) {
				for (int k = 0; k < K; k++) {
					theta[m][k] = thetasum[m][k] / numstats;
				}
			}
		} else {
			for (int m = 0; m < documents.length; m++) {
				for (int k = 0; k < K; k++) {
					theta[m][k] = (nd[m][k] + alpha[k]) / (ndSum[m] + alphaSum);
				}
			}
		}
		return theta;
	}

	/**
	 * Retrieve estimated topic--word associations. If sample lag > 0 then the
	 * mean value of all sampled statistics for phi[][] is taken.
	 * 
	 * @return phi multinomial mixture of topic words (K x V)
	 */
	public double[][] getPhi() {
		double[][] phi = new double[K][V];
		if (SAMPLE_LAG > 0) {
			for (int k = 0; k < K; k++) {
				for (int w = 0; w < V; w++) {
					phi[k][w] = phisum[k][w] / numstats;
				}
			}
		} else {
			for (int k = 0; k < K; k++) {
				for (int w = 0; w < V; w++) {
					phi[k][w] = (nw[w][k] + beta[k][w]) / (nwSum[k] + betaSum[k]);
				}
			}
		}
		return phi;
	}

	public double[] getB() {
		return b;
	}

	public int[][] getNw() {
		return nw;
	}

	public int[] getNwSum() {
		return nwSum;
	}

}
