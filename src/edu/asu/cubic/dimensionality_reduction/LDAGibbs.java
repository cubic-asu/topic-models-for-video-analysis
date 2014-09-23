package edu.asu.cubic.dimensionality_reduction;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;

/**
 * Gibbs sampler for estimating the best assignments of topics for words and
 * documents in a corpus. The algorithm is introduced in Tom Griffiths' paper
 * "Gibbs sampling in the generative model of Latent Dirichlet Allocation"
 * (2002).
 * 
 * @author heinrich
 */

public class LDAGibbs implements Serializable , Cloneable{

	private static final long serialVersionUID = -6173930724611415371L;
	String phase;// training or unseen
	/**
	 * document data (term lists)
	 */
	int[][] documents;
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
	 * max iterations
	 */
	int ITERATIONS = 100;
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

	public LDAGibbs(int[][] documents, int V, int K, double[] alpha, double[][] beta, String mPath, String bmName) {
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
	}

	public LDAGibbs(int[][] documents, LDAGibbs model, int K) {
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

	/**
	 * Main method: Select initial state ? Repeat a large number of times: 1.
	 * Select an element 2. Update conditional on other elements. If
	 * appropriate, output summary for each run.
	 * 
	 * @param K
	 *            number of topics
	 * @param prior_alpha
	 *            symmetric prior parameter on document--topic associations
	 * @param beta
	 *            symmetric prior parameter on topic--term associations
	 */
	public void gibbs() throws IOException {
		// initial state of the Markov chain:
		//initialState();
		//System.out.println("In LDATrain Gibbs");
		//System.out.println("Sampling " + ITERATIONS
			//	+ " iterations with burn-in of " + BURN_IN + " (B/S="
				//+ THIN_INTERVAL + ").");
		likelihoods= new double[ITERATIONS];
		for (int i = 0; i < ITERATIONS; i++) {
			System.out.println(" Iteration: "+(i+1));
			// for all z_i
			for (int m = 0; m < z.length; m++) {
				//System.out.println();
				for (int n = 0; n < z[m].length; n++) {
					//System.out.print("m: "+m+" n: "+documents[m][n]);
					// (z_i = z[m][n])
					// sample from p(z_i|z_-i, w)
					int topic = sampleFullConditional(m, n);
					z[m][n] = topic;
				}
			}
			if ((i < BURN_IN) && (i % THIN_INTERVAL == 0)) {
				//System.out.print("B");
				dispcol++;
			}
			// display progress
			if ((i > BURN_IN) && (i % THIN_INTERVAL == 0)) {
				//System.out.print("S");
				dispcol++;
			}
			// get statistics after burn-in
			if ((i > BURN_IN) && (SAMPLE_LAG > 0) && (i % SAMPLE_LAG == 0)) {
				updateParams();
				//System.out.print("|");
				if (i % THIN_INTERVAL != 0)
					dispcol++;
			}
			if (dispcol >= 10) {
				//System.out.println();
				dispcol = 0;
			}
			//System.out.println(String.format("%d %.3f ", (i+1),modelLogLikelihood()));

			likelihoods[i]= modelLogLikelihood();
			// dump the current model to disk if it is a training phase
			if((i+1)%100==0 && phase.equalsIgnoreCase("training")){
				try{
				// make a clone of this object
				LDAGibbs clonedModel= (LDAGibbs)this.clone();
				clonedModel.cleanUpVariables();
				// write the object to file
				String modelFilePath= modelPath+"/"+baseModelName+"_"+(i+1)+".model";
				ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelFilePath));
				oos.writeObject(clonedModel);
				oos.close();
				}
				catch(CloneNotSupportedException cnse)
				{cnse.printStackTrace();}
			}
		}
	}

	/**
	 * Sample a topic z_i from the full conditional distribution: p(z_i = j |
	 * z_-i, w) = (n_-i,j(w_i) + beta)/(n_-i,j(.) + W * beta) * (n_-i,j(d_i) +
	 * alpha)/(n_-i,.(d_i) + K * alpha)
	 * 
	 * @param m
	 *            document
	 * @param n
	 *            word
	 */
	private int sampleFullConditional(int m, int n) {
		// remove z_i from the count variables
		int topic = z[m][n];
		nw[documents[m][n]][topic]--;
		nd[m][topic]--;
		nwSum[topic]--;
		ndSum[m]--;
		// do multinomial sampling via cumulative method:
		double[] p = new double[K];
		int maxK=0;double maxP=0.0;
		for (int k = 0; k < K; k++) {
			/*p[k] = (nw[documents[m][n]][k] + beta) / (nwsum[k] + V * beta)
			 * (nd[m][k] + alpha) / (ndsum[m] + K * alpha);*/
			try{
				if(phase.equalsIgnoreCase("training"))
					p[k] = (nw[documents[m][n]][k] + beta[k][documents[m][n]]) / (nwSum[k] + betaSum[k])
						* (nd[m][k] + alpha[k]) / (ndSum[m] + alphaSum);
				else
					p[k] = (nwTrain[documents[m][n]][k] + nw[documents[m][n]][k] + beta[k][documents[m][n]]) / (nwSum[k] + betaSum[k])
						* (nd[m][k] + alpha[k]) / (ndSum[m] + alphaSum);
			}catch(IndexOutOfBoundsException iobe){
				System.out.println(documents[m][n]+beta[k][documents[m][n]]);
			}
			if(p[k]>maxP){
				maxK=k;
				maxP= p[k];
			}
		}
		//System.out.print(" "+(maxK+1));
		// cumulate multinomial parameters
		for (int k = 1; k < p.length; k++) {
			p[k] += p[k - 1];
		}
		// scaled sample because of unnormalised p[]
		double u = Math.random() * p[K - 1];
		for (topic = 0; topic < p.length; topic++) {
			if (u < p[topic])
				break;
		}
		// add newly estimated z_i to count variables
		nw[documents[m][n]][topic]++;
		nd[m][topic]++;
		nwSum[topic]++;
		ndSum[m]++;

		return topic;
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
		/*for (int k = 0; k < K; k++) {
			for (int w = 0; w < V; w++) {
				phisum[k][w] += (nw[w][k] + beta[k][w]) / (nwSum[k] + betaSum[k]);
			}
		}*/
		numstats++;
	}

	public int[][] getZ(){
		return z;
	}

	public int[][] getNw() {
		return nw;
	}

	public int[] getNwSum() {
		return nwSum;
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

	public int[][] getInitialZs(){
		return initialZs;
	}
	
	public double[] getLikelihoods(){
		return likelihoods;
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
	}
	/**
	 * Configure the gibbs sampler
	 * 
	 * @param iterations
	 *            number of total iterations
	 * @param burnIn
	 *            number of burn-in iterations
	 * @param thinInterval
	 *            update statistics interval
	 * @param sampleLag
	 *            sample interval (-1 for just one sample at the end)
	 */
	public void configure(int iterations, int burnIn, int thinInterval,
			int sampleLag) {
		ITERATIONS = iterations;
		BURN_IN = burnIn;
		THIN_INTERVAL = thinInterval;
		SAMPLE_LAG = sampleLag;
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
	
}
