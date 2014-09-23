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
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import no.uib.cipr.matrix.Matrices;
import weka.core.matrix.Matrix;
import edu.asu.cubic.util.Utilities;

public class SLDAVB implements Serializable, Cloneable {

	private static final long serialVersionUID = -343920565627682987L;

	double alpha;
	double delta; // this is the dispersion parameter in the GLM model
	double[] eta; // these are the regression coefficients in the GLM model  
	/* log probability per topic per term */
	double log_prob_w[][];
	/* number of topics */
	int K;
	/* total vocabulary size */
	int V;
	/* counts of term assignments per topic */
	double class_word[][];
	/* counts of total terms assigned per topic */
	double class_total[];
	double alpha_suffstats;
	// variational parameters
	/* counts of topics per document */
	double var_gamma[][];
	/* counts of terms in each doc assigned to each topic */
	//double phi[][][];
	/* topic assignments to words in each document */
	//int z[][];
	/* documents in term:count format */
	//ArrayList<HashMap<Integer,Integer>> documents;
	int[][] docs;
	int[] docLengths;
	double[] ys; // since this is supervised model the ys are the labels
	/* max number of EM Iterations allowed */
	int EMMaxIters;
	/* minimum change needed to achieve convergence */
	double EMConverged;
	/* max number of Iterations within each variational step allowed */
	int VarMaxIters;
	/* minimum change needed to achieve convergence in a variational step*/
	double VarConverged; 
	// The name with which this model will be dumped to disk every 100 iterations
	String modelPath;
	String baseModelName;
	boolean writeIterations;
	ArrayList<Double> likelihoods;

	final int LAG = 5;

	public SLDAVB(){}

	public SLDAVB(int[][] docs, double[] labels ,int V, int K, double alpha, double delta, String mPath, String bmName, int EMIters, double EMConverged,
			int VarIters, double VarConverged, boolean writeIters) {
		this.V = V;
		this.K = K;
		//labels= Utilities.scaleData(labels, 1, 100);
		ys= labels;
		this.delta= delta;
		this.alpha = alpha;
		modelPath= mPath;
		baseModelName= bmName;
		EMMaxIters= EMIters;
		this.EMConverged= EMConverged;
		VarMaxIters= VarIters;
		this.VarConverged= VarConverged;
		writeIterations= writeIters;
		this.docs= docs;
		//documents=  new ArrayList<HashMap<Integer,Integer>>();
		docLengths= new int[docs.length];
		//z= new int[docs.length][];
		// convert the docs to term:count format
		for(int doc=0; doc<docs.length; doc++){
			//z[doc]= new int[docs[doc].length];
			try{
				docLengths[doc]= docs[doc].length;
			}
			catch(NullPointerException npe){
				//System.out.println("Document "+(doc+1)+" is null");
				//npe.printStackTrace();
				//System.exit(1);
				docLengths[doc]=0;
			}
			//System.out.println(map);
		}
		//Utilities.printArray(docLengths);
		likelihoods= new ArrayList<Double>();
		eta= new double[K];
	}

	public SLDAVB(int[][] docs, SLDAVB model, String mPath, String bmName, int VarIters, double VarConverged) {
		modelPath= mPath;
		baseModelName= bmName;
		VarMaxIters= VarIters;
		this.VarConverged= VarConverged;
		this.docs= docs;
		//documents=  new ArrayList<HashMap<Integer,Integer>>();
		docLengths= new int[docs.length];
		// convert the docs to term:count format
		for(int doc=0; doc<docs.length; doc++){
			docLengths[doc]= docs[doc].length;
			//System.out.println(map);
		}
		K= model.K;
		V= model.V;
		alpha= model.alpha;
		delta= model.delta;
		log_prob_w= model.getBeta();
		eta= model.getCoeffs();
	}

	/**
	 * This method cleans up all unnecessary variables so that the object becomes much lighter
	 */
	public void cleanUpVariables(){
		//documents= null;
		docs= null;
		//class_total= null;
		//class_word= null;
		docLengths= null;
		//phi= null;
	}

	/**
	 * This method cleans up all unnecessary variables so that the object becomes much lighter
	 */
	public void cleanUpVariablesTemp(){
		//documents= null;
		docLengths= null;
		docs= null;
		//phi= null;
	}

	/*
	 * compute likelihood bound
	 *
	 */
	double compute_likelihood(int doc, double[][] phi)
	{
		double likelihood = 0, digsum = 0, var_gamma_sum = 0;
		double[] dig=new double[K];
		//HashMap<Integer,Integer> currDoc= documents.get(doc);
		int total= docs[doc].length;
		/*int[] termsInDoc= new int[total];
		int index=0;
		for(Integer key: currDoc.keySet()){
			termsInDoc[index]= key;
			index++;
		}*/
		for (int k = 0; k < K; k++)
		{
			dig[k] = Utilities.digamma(var_gamma[doc][k]);
			var_gamma_sum += var_gamma[doc][k];
		}
		digsum = Utilities.digamma(var_gamma_sum);
		//Utilities.printArray("Gamma:",var_gamma[doc] );
		//System.out.println(var_gamma_sum);
		likelihood = Utilities.LogGamma(alpha * K)- K * Utilities.LogGamma(alpha)- (Utilities.LogGamma(var_gamma_sum));
		// Add log h(y,delta) = log(1/((2*pi)^(1/2)delta)) - y^2/2 term
		likelihood += Math.log(1/(Math.sqrt(2*Math.PI)*delta))- (Math.pow(ys[doc],2)/2);
		// term1= \eta^T(E[\bar{Z}]y) where E[\bar{Z}] = (1/N)\sim_{n=1}^N\phi_n
		double term1=0;
		double[] EZ= new double[K];
		for (int k = 0; k < K; k++){
			EZ[k]=0;
			for (int n = 0; n < total; n++)
			{
				EZ[k]+=phi[n][k];
			}
			EZ[k]/=total;
			term1+= eta[k]*EZ[k]*ys[doc];
		}
		// term2= E[A(\eta^T\bar{Z})]= (1/(2N^2))(\sum_n=1^N\sum_{m\neq n}\phi_n\phi_m^T + \sum_{n=1}^N diag{\phi_n})
		double term2=0;
		Matrix result= new Matrix(K, K, 0);
		for(int n=0; n<total; n++){
			Matrix phin= new Matrix(phi[n], K);
			for(int m=0; m<total; m++){
				if(m!=n){
					Matrix phim=  new Matrix(phi[m],K);
					result= result.plusEquals(phin.times(phim.transpose()));
				}
			}
			Matrix diag= new Matrix(K, K, 0);
			for(int k=0; k<K; k++)
				diag.set(k, k, phi[n][k]);
			result= result.plusEquals(diag);
		}
		term2= new Matrix(eta,K).transpose().times(result).times(new Matrix(eta,K)).getArray()[0][0];
		term2/=(2*total*total);
		likelihood += (term1-term2)/delta;
		//System.out.println("1: "+likelihood);
		for (int k = 0; k < K; k++)
		{
			likelihood += (alpha - 1)*(dig[k] - digsum) + Utilities.LogGamma(var_gamma[doc][k])
					- (var_gamma[doc][k] - 1)*(dig[k] - digsum);

			//System.out.println("2: "+likelihood);
			for (int n = 0; n < total; n++)
			{
				/*System.out.println(counts[n]*
							(phi[n][k]*((dig[k] - digsum) - Math.log(phi[n][k])
									+ log_prob_w[k][termsInDoc[n]])));*/
				if (phi[n][k] > 0)
				{
					likelihood += 
							(phi[n][k]*((dig[k] - digsum) - Math.log(phi[n][k])
									+ log_prob_w[k][docs[doc][n]]));

				}
			}
			//System.out.println();
		}
		return(likelihood);
	}

	/**
	 * This method computes likelihod for unseen doc
	 * @param doc
	 * @param phi
	 * @return
	 */
	double compute_likelihoodUnseenDoc(int doc, double[][] phi)
	{
		double likelihood = 0, digsum = 0, var_gamma_sum = 0;
		double[] dig=new double[K];
		int total= docLengths[doc];
		for (int k = 0; k < K; k++)
		{
			dig[k] = Utilities.digamma(var_gamma[doc][k]);
			var_gamma_sum += var_gamma[doc][k];
		}
		digsum = Utilities.digamma(var_gamma_sum);
		//Utilities.printArray("Gamma:",var_gamma[doc] );
		//System.out.println(var_gamma_sum);
		likelihood = Utilities.LogGamma(alpha * K)- K * Utilities.LogGamma(alpha)- (Utilities.LogGamma(var_gamma_sum));
		//System.out.println("1: "+likelihood);
		for (int k = 0; k < K; k++)
		{
			likelihood += (alpha - 1)*(dig[k] - digsum) + Utilities.LogGamma(var_gamma[doc][k])
					- (var_gamma[doc][k] - 1)*(dig[k] - digsum);

			//System.out.println("2: "+likelihood);
			for (int n = 0; n < total; n++)
			{
				/*System.out.println(counts[n]*
							(phi[n][k]*((dig[k] - digsum) - Math.log(phi[n][k])
									+ log_prob_w[k][termsInDoc[n]])));*/
				if (phi[n][k] > 0)
				{
					likelihood += 
							(phi[n][k]*((dig[k] - digsum) - Math.log(phi[n][k])
									+ log_prob_w[k][docs[doc][n]]));
					/*System.out.println(phi[n][k]);
					System.out.println((dig[k] - digsum));
					System.out.println(- Math.log(phi[n][k]));
					System.out.println( log_prob_w[k][termsInDoc[n]]);
					System.out.println(counts[n]*
							(phi[n][k]*((dig[k] - digsum) - Math.log(phi[n][k])
									+ log_prob_w[k][termsInDoc[n]])));
					System.out.println("3: "+likelihood);*/
				}
			}
			//System.out.println();
		}
		return(likelihood);
	}

	public double[][] getBeta(){
		return log_prob_w;
	}

	public double[] getCoeffs(){
		return eta;
	}

	/*
	 * Normalize the var_gamma vectors and return them
	 */
	public double[][] getGamma(){
		double[][] normGamma= new double[var_gamma.length][K];
		for(int d=0; d< var_gamma.length; d++){
			normGamma[d]= Utilities.normalize(var_gamma[d]);
		}
		return normGamma;
	}

	/*
	 * inference only
	 *
	 */
	public void infer() throws IOException
	{
		double likelihood;
		int numDocs= docs.length;
		var_gamma= new double[numDocs][K];
		//phi= new double[numDocs][maxLength][K];
		likelihoods= new ArrayList<Double>();
		//PrintWriter likelihoodFile= new PrintWriter(new File(modelPath+"/"+filename+"-lda-lhood.dat"));
		//PrintWriter tempPhiFile= new PrintWriter(new FileWriter(modelPath+"//"+baseModelName+"Temp.txt",true));
		//PrintWriter tempTermsFile= new PrintWriter(new FileWriter(modelPath+"//"+baseModelName+"TempTerms.txt",true));
		for (int d = 0; d < numDocs; d++)
		{
			//if (((d % 100) == 0) && (d>0)) System.out.println("document "+d+"\n");
			double[][] phi= new double[docLengths[d]][K];
			likelihood = slda_inference(d,phi);
			likelihoods.add(likelihood);

		}

		// if any of the documents is empty then set gamma to NaN
		for(int d=0; d< docs.length; d++){
			if(docs[d].length==0){
				for(int k=0; k<K; k++)
					var_gamma[d][k]= Double.NaN;
			}
		}
	}

	/*
	 * variational inference
	 *
	 */
	double slda_inference(int doc, double[][] phi)
	{
		double converged = 1;
		double phisum = 0, likelihood = 0;
		double likelihood_old = 0;
		double before, after;
		double[] oldphi= new double[K];
		int var_iter;
		double[] digamma_gam=new double[K];
		//HashMap<Integer, Integer> currDoc= documents.get(doc);
		int total= docLengths[doc];// this the the toal unique words
		/*int[] termsInDoc= new int[total];
		int[] counts= new int[total];
		int index=0;
		for(Integer key: currDoc.keySet()){
			termsInDoc[index]= key;
			counts[index]= currDoc.get(key);
			index++;
		}*/
		// compute posterior dirichlet
		for (int k = 0; k < K; k++)
		{
			var_gamma[doc][k] = alpha + (docLengths[doc]/((double) K));
			digamma_gam[k] = Utilities.digamma(var_gamma[doc][k]);
			for (int n = 0; n < total; n++)
				phi[n][k] = 1.0/K;
		}
		var_iter = 0;
		if(docs[doc].length!=0){
			//Utilities.printArray("Before ", var_gamma[doc]);
			while ((converged > VarConverged) &&
					((var_iter < VarMaxIters) || (VarMaxIters == -1)))
			{
				before= System.currentTimeMillis();
				var_iter++;
				//System.out.println("\tIter: "+var_iter);
				for (int n = 0; n < total; n++)
				{
					/*if(var_iter==2)
					Utilities.printArray("Phi", phi[n]);*/
					phisum = 0;
					for (int k = 0; k < K; k++)
					{
						oldphi[k] = phi[n][k];
						phi[n][k] = digamma_gam[k] + log_prob_w[k][docs[doc][n]];

						if (k > 0)
							phisum = Utilities.log_sum(phisum, phi[n][k]);
						else
							phisum = phi[n][k]; // note, phi is in log space
					}

					for (int k = 0; k < K; k++)
					{
						phi[n][k] = Math.exp(phi[n][k] - phisum);
					}
					for (int k = 0; k < K; k++){
						var_gamma[doc][k] =
								var_gamma[doc][k] + (phi[n][k] - oldphi[k]);
						// !!! a lot of extra digamma's here because of how we're computing it
						// !!! but its more automatically updated too.
						digamma_gam[k] = Utilities.digamma(var_gamma[doc][k]);
					}
					//Utilities.printArray("", var_gamma[doc]);
					//System.out.println();
				}
				after= System.currentTimeMillis();
				//System.out.println("One Inference Iteration has taken "+(after-before)/1000+" secs");
				before= System.currentTimeMillis();
				likelihood = compute_likelihoodUnseenDoc(doc,phi);
				after= System.currentTimeMillis();
				//System.out.println("computeLikelihood has taken "+(after-before)/1000+" secs");
				assert(!Double.isNaN(likelihood));
				//System.out.println("Likelihood: "+ likelihood_old+"  "+ likelihood);
				converged = (likelihood_old - likelihood) / likelihood_old;
				likelihood_old = likelihood;

			}
		}
		//Utilities.printArray("Gamma:", var_gamma[doc]);
		return(likelihood);
	}

	/*
	 * compute MLE lda model from sufficient statistics
	 *
	 */
	void slda_mle() throws Exception
	{
		for (int k = 0; k < K; k++)
		{
			for (int w = 0; w < V; w++)
			{
				if (class_word[k][w] > 0)
				{
					log_prob_w[k][w] =
							Math.log(class_word[k][w]) -
							Math.log(class_total[k]);
				}
				else
					log_prob_w[k][w] = -100;
			}
		}
		// load phis
		BufferedReader tempPhiFile= new BufferedReader(new FileReader(modelPath+"//"+baseModelName+"Temp.txt"));
		// estimate the regression coefficients \eta
		// eta= (E[XTX])^-1*E[X]^T*y
		double[][] EXT = new double[K][docs.length];
		Matrix EXTX= new Matrix(K, K);
		//Utilities.printArray(docLengths);
		for(int d=0; d<docs.length;d++){
			//System.out.println("Doc: "+d);
			String newline= tempPhiFile.readLine();
			if(docLengths[d]==0){
				for (int k = 0; k < K; k++){
					EXT[k][d]= 1/K;
				}
			}
			else{
				//System.out.println("Doc "+(d+1));
				Matrix temp= new Matrix(K,K);
				//System.out.println("Reading phi for Doc: "+(d+1));
				//System.out.println(newline);
				double[][] phi= new double[docLengths[d]][K];
				String[] tokens= newline.split(";");
				for(int n=0; n<tokens.length; n++){
					String[] subTokens= tokens[n].split(",");
					for(int k=0; k<K; k++)
						phi[n][k]= Double.parseDouble(subTokens[k]);
				} 
				for (int k = 0; k < K; k++){
					EXT[k][d]=0;
					for (int n = 0; n < docLengths[d]; n++)
					{
						EXT[k][d]+=phi[n][k];
					}
					EXT[k][d]/=docLengths[d];
				}
				//Utilities.printArray(EXT);
				for(int n=0; n<docLengths[d]; n++){
					Matrix phin= new Matrix(phi[n], K);
					for(int m=0; m<docLengths[d]; m++){
						if(m!=n){
							Matrix phim=  new Matrix(phi[m],K);
							temp= temp.plusEquals(phin.times(phim.transpose()));
						}
					}
					Matrix diag= new Matrix(K, K, 0);
					for(int k=0; k<K; k++)
						diag.set(k, k, phi[n][k]);
					//Utilities.printArray(diag.getArray());
					temp= temp.plusEquals(diag);
				}
				//Utilities.printArray(temp.getArray());
				temp=temp.times(((double)1/Math.pow(docLengths[d],2)));
				//Utilities.printArray(temp.getArray());
				EXTX=EXTX.plusEquals(temp);
			}
		}
		tempPhiFile.close();
		//Utilities.printArray(EXT);
		//System.out.println();
		//Utilities.printArray(EXTX.getArray());
		// add small noise to EXTX so that there i noe problem in finding inverse
		for(int j=0; j<K; j++)
			EXTX.set(j, j, EXTX.get(j, j)+(Math.random()/10000)) ;
		if(Utilities.containsNaN(EXT) || Utilities.containsInfinity(EXT))
			throw new Exception("NaN or Inifinity in EXT");
		if(Utilities.containsNaN(EXTX.getArray()) || Utilities.containsInfinity(EXTX.getArray()))
			throw new Exception("NaN or Inifinity in EXTX");
		double[][] EXTXInv= Utilities.matrixInverse(EXTX.getArray());
		Matrix result= new Matrix(EXTXInv).times(new Matrix(EXT)).times(new Matrix(ys,docs.length));
		//Utilities.printArray(result.getArray());
		for(int k=0; k<K; k++){
			eta[k]= result.get(k, 0);
		}
		Utilities.printArray("eta", eta);
		if(Utilities.containsNaN(eta) || Utilities.containsInfinity(eta))
			throw new Exception("NaN or Inifinity in eta");
		new File(modelPath+"//"+baseModelName+"Temp.txt").delete();
	}

	public void random_initialize_ss() throws IOException{

		int numDocs= docs.length;
		var_gamma= new double[numDocs][K];
		//phi= new double[numDocs][maxLength][K];
		log_prob_w= new double[K][V];
		for(int i=0; i< K; i++)
			for(int j=0; j< V; j++)
				log_prob_w[i][j]= 0;
		class_total= new double[K];
		class_word= new double[K][V];
		String[] tokens= baseModelName.split("_");
		// check if the the initial assignments file exists
		if(new File(modelPath+"//"+tokens[0]+"_"+tokens[1]+"Init.txt").exists()){
			String[][] tokens1= Utilities.readCSVFile(modelPath+"//"+tokens[0]+"_"+tokens[1]+"Init.txt", false);
			for (int k = 0; k < K; k++)
				for (int n = 0; n < V; n++)
				{
					class_word[k][n]+= Double.parseDouble(tokens1[k][n]); 
					class_total[k] += class_word[k][n];
				}
		}
		else{
			PrintWriter initialAsstsFile= new PrintWriter(new File(modelPath+"//"+tokens[0]+"_"+tokens[1]+"Init.txt"));
			for (int k = 0; k < K; k++){
				for (int n = 0; n < V; n++)
				{
					double val= 1/V + Math.random();
					class_word[k][n]+= val; 
					class_total[k] += class_word[k][n];
					initialAsstsFile.print(String.format("%.3f",val));
					if(n<V-1)
						initialAsstsFile.print(",");
				}
				initialAsstsFile.println();
			}
			initialAsstsFile.close();
		}
		for (int k = 0; k < K; k++)
		{
			for (int w = 0; w < V; w++)
			{
				if (class_word[k][w] > 0)
				{
					log_prob_w[k][w] =
							Math.log(class_word[k][w]) -
							Math.log(class_total[k]);
				}
				else
					log_prob_w[k][w] = -100;
			}
			eta[k]= ((double)1/K);
		}
		//Utilities.printArray("eta", eta);
	}

	public void runParallelEM() throws Exception
	{
		int threads = Runtime.getRuntime().availableProcessors();
		System.out.println("Num of threads: "+threads);
		// initialize model
		random_initialize_ss();
		double before, after;
		//lda_mle();
		double converged= 1;
		int iter= 0;
		double likelihood= 0,likelihood_old = 0;

		for(int i=1; i<=EMMaxIters;i++){
			if(new File(modelPath+"/"+baseModelName+"_"+(i)+".model").exists()){
				// if the model for this exists
				iter= i;
				SLDAVB model= (SLDAVB)new ObjectInputStream(new FileInputStream(modelPath+"/"+baseModelName+"_"+(i)+".model")).readObject();
				class_total= model.class_total;
				class_word= model.class_word;
				log_prob_w= model.log_prob_w;
				likelihoods= model.likelihoods;
				likelihood_old= likelihoods.get(likelihoods.size()-1);
			}
		}
		//PrintWriter likelihoodFile= new PrintWriter(new File(modelPath+"/"+baseModelName+".likelihood"));
		if(new File(modelPath+"//"+baseModelName+"Temp.txt").exists()){
			new File(modelPath+"//"+baseModelName+"Temp.txt").delete();
		}
		while (((converged < 0) || (converged > EMConverged) || (iter <= 2)) && (iter <= EMMaxIters)){
			iter++; System.out.println("EM iteration "+ iter);
			likelihood = 0;
			zero_initialize_ss();
			// e-step
			for (int d = 0; d < docs.length; d+=threads)
			{
				if ((d % 1000) == 0) System.out.println("\t Doc: "+(d+1)+" Likelihood "+likelihood);
				before= System.currentTimeMillis();
				//double[][] phi= new double[docLengths[d]][K];
				//likelihood += doc_e_step(d,phi);
				ExecutorService service = Executors.newFixedThreadPool(threads);
				List<Future<EStepOutput>> futures = new ArrayList<Future<EStepOutput>>();
				// Map
				int numThreads= Math.min(threads, docs.length-d);
				for(int i=0; i<numThreads;i++){
					Callable<EStepOutput> mapper= this.new Mapper(d+i,i);
					futures.add(service.submit(mapper));
				}
				double[][][] phi= new double[numThreads][][];
				int threadNums[]= new int[numThreads];
				// Reduce
				for(Future<EStepOutput> futureoutput: futures){
					phi[futureoutput.get().threadIndex]= futureoutput.get().phi;
					threadNums[futureoutput.get().threadIndex]= futureoutput.get().docIndex;
					likelihood+= futureoutput.get().likelihood;
				}
				after= System.currentTimeMillis();
				service.shutdown();
				// write all phis to file
				PrintWriter tempPhiFile= new PrintWriter(new FileWriter(modelPath+"//"+baseModelName+"Temp.txt",true));
				for(int t=0; t<numThreads; t++){
					//System.out.println("Writing phi for doc "+ threadNums[t]);
					for(int n=0; n<docLengths[d+t]; n++){
						/*// normalize phi
						double totalPhi=0;
						for(int k=0; k<K; k++){
							totalPhi+=phi[t][n][k];
						}*/
						for(int k=0; k<K; k++){
							tempPhiFile.print(String.format("%.3f",(phi[t][n][k])));
							if(k<K-1)
								tempPhiFile.print(",");
						}
						if(n<docLengths[d+t]-1)
							tempPhiFile.print(";");
					}
					tempPhiFile.println();
				}
				tempPhiFile.close();
				//System.out.println("\tTime take in secs: "+(after-before)/1000);
				//Utilities.printArray("Gamma: ", var_gamma[d]);
			}
			// m-step
			slda_mle();

			// check for convergence
			converged = (likelihood_old - likelihood) / (likelihood_old);
			if (converged < 0) VarMaxIters = VarMaxIters * 2;
			likelihood_old = likelihood;

			// output model and likelihood
			System.out.println(String.format("\t %10.10f\t%5.5e\n", likelihood, converged));
			//likelihoodFile.println(String.format("\t %10.10f\t%5.5e\n", likelihood, converged));
			likelihoods.add(likelihood);

			if(new File(modelPath+"/"+baseModelName+"_"+(iter-1)+".model").exists()){
				// delete the model from prev iteration
				new File(modelPath+"/"+baseModelName+"_"+(iter-1)+".model").delete();
			}
			// make a clone of this object
			SLDAVB clonedModel= (SLDAVB)this.clone();
			clonedModel.cleanUpVariablesTemp();
			// write the object to file
			String modelFilePath= modelPath+"/"+baseModelName+"_"+(iter)+".model";
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelFilePath));
			oos.writeObject(clonedModel);
			oos.close();
		}
		// output the final model
		//save_lda_model(baseModelName);
		//save_gamma(baseModelName);
		
		// before leaving check if there is any empty document if so set Gamma to NaN for that document
		for(int d=0; d< docs.length; d++){
			if(docLengths[d]==0){
				for(int k=0; k<K; k++)
					var_gamma[d][k]= Double.NaN;
			}
		}
		// write the log_prob_w to file
		PrintWriter pw= new PrintWriter(new File(modelPath+"/"+baseModelName+"Beta.csv"));
		for (int k = 0; k < K; k++)
		{
			for (int w = 0; w < V; w++)
			{
				pw.print(log_prob_w[k][w]);
				if(w<V-1)
					pw.print(",");
			}
			pw.println();
		}
		pw.close();
		if(new File(modelPath+"/"+baseModelName+"_"+(iter)+".model").exists()){
			// delete the model from prev iteration
			new File(modelPath+"/"+baseModelName+"_"+(iter)+".model").delete();
		}
		//likelihoodFile.close();
	}

	/*
	 * various intializations for the sufficient statistics
	 *
	 */
	void zero_initialize_ss()
	{
		for (int k = 0; k < K; k++)
		{
			class_total[k] = 0;
			for (int w = 0; w < V; w++)
			{
				class_word[k][w] = 0;
			}
		}

	}

	class EStepOutput {
		int docIndex;
		public double likelihood;
		public double[][] phi;
		int threadIndex;
		EStepOutput(int doc, double lik, double[][] r, int tIndex){
			docIndex= doc;
			likelihood= lik;
			phi= r;
			threadIndex= tIndex;
		}
	}

	/**
	 * This Mapper class maps each document to a processor (thread) and executes the 
	 * E-Step 
	 * @author prasanthl
	 *
	 */
	public class Mapper implements Callable<EStepOutput> {
		int docIndex;
		double likelihood;
		double[][] phi;
		int threadIndex;
		Mapper(int doc, int tIndex){
			docIndex= doc;
			threadIndex= tIndex;
		}
		public EStepOutput call() throws Exception {
			/*if(docIndex<10)
				System.out.println("\tThread "+threadIndex+" For Doc: "+docIndex);*/
			double[][] phi= new double[docLengths[docIndex]][K];
			double converged = 1;
			double phisum = 0, likelihood = 0;
			double likelihood_old = 0;
			double before, after;
			double[] oldphi= new double[K];
			int var_iter;
			double[] digamma_gam=new double[K];
			//HashMap<Integer, Integer> currDoc= documents.get(docIndex);
			int total= docLengths[docIndex];
			/*int[] termsInDoc= new int[total];
			int[] counts= new int[total];
			int index=0;
			for(Integer key: currDoc.keySet()){
				termsInDoc[index]= key;
				counts[index]= currDoc.get(key);
				index++;
			}*/
			// compute posterior dirichlet
			for (int k = 0; k < K; k++)
			{
				var_gamma[docIndex][k] = alpha + (docLengths[docIndex]/((double) K));
				digamma_gam[k] = Utilities.digamma(var_gamma[docIndex][k]);
				for (int n = 0; n < total; n++)
					phi[n][k] = 1.0/K;
			}
			var_iter = 0;
			before= System.currentTimeMillis();
			if(docLengths[docIndex]!=0){
				//Utilities.printArray("Before ", var_gamma[doc]);
				while ((converged > VarConverged) && ((var_iter < VarMaxIters) || (VarMaxIters == -1)))
				{
					before= System.currentTimeMillis();
					/*for (int n = 0; n < total; n++){
					System.out.println(log_prob_w[0][termsInDoc[n]]);
				}*/
					double[] phiNegn= new double[K];
					for(int k=0; k<K; k++){
						phiNegn[k]=0;
					}
					//before= System.currentTimeMillis();
					// calcuate \phi_{-n}
					for(int m=0; m<total; m++){
						for(int k=0; k<K; k++){
							phiNegn[k]+= phi[m][k];
						}
					}
					var_iter++;
					//System.out.println("\tIter: "+var_iter);
					for (int n = 0; n < total; n++)
					{
						for(int k=0; k<K; k++){
							phiNegn[k]-= phi[n][k];
						}
						//after= System.currentTimeMillis();
						//System.out.println("Time taken1: "+(after-before)/1000+" secs");
						//before= System.currentTimeMillis();
						//double term= new Matrix(eta,K).transpose().times(new Matrix(phiNegn,K)).getArray()[0][0];
						double term=0;
						for(int k=0; k<K; k++)
							term+= eta[k]*phiNegn[k];
						//System.out.print(" "+term);
						//after= System.currentTimeMillis();
						//System.out.println("Time taken2: "+(after-before)/1000+" secs");
						//Utilities.printArray("eta",eta);
						phisum = 0;
						for (int k = 0; k < K; k++)
						{
							oldphi[k] = phi[n][k];
							// for slda extra term is
							// (-1/(2N^2\delta))(2(\eta^T\phi-{-j})\eta_k+\eta.*\eta)
							phi[n][k] = digamma_gam[k] + log_prob_w[k][docs[docIndex][n]]
									+ ((ys[docIndex]/(total*delta))*eta[k] )-( (1/(2*total*total*delta))*
											((2*term*eta[k]) + (eta[k]*eta[k]))	);
							/*System.out.println("term1: "+digamma_gam[k]);
							System.out.println("term2: "+log_prob_w[k][docs[docIndex][n]]);
							System.out.println("term3: "+((ys[docIndex]/(total*delta))*eta[k] ));
							System.out.println("term4: "+-( (1/(2*total*total*delta))*
											((2*term*eta[k]) + (eta[k]*eta[k]))	));*/
							if (k > 0)
								phisum = Utilities.log_sum(phisum, phi[n][k]);
							else
								phisum = phi[n][k]; // note, phi is in log space
							if(Double.isNaN(phisum)){
								System.out.println("NaN in phisum: ");
								Utilities.printArray("phi[n]" , phi[n]);
								System.exit(1);
							}
						}
						//System.out.println("phisum: "+phisum);
						//Utilities.printArray("phi[n]", phi[n]);
						for (int k = 0; k < K; k++)
						{
							phi[n][k] = Math.exp(phi[n][k] - phisum);
						}
						//Utilities.printArray("phi[n]", phi[n]);
						for (int k = 0; k < K; k++){
							var_gamma[docIndex][k] =
									var_gamma[docIndex][k] + (phi[n][k] - oldphi[k]);
							// !!! a lot of extra digamma's here because of how we're computing it
							// !!! but its more automatically updated too.
							digamma_gam[k] = Utilities.digamma(var_gamma[docIndex][k]);
						}
						if(Double.isNaN(phisum)){
							System.out.println("NaN in var_gamma: ");
							Utilities.printArray("", var_gamma[docIndex]);
							System.exit(1);
						}
						for(int k=0; k<K; k++){
							phiNegn[k]+= phi[n][k];
						}
						//System.out.println();
					}
					after= System.currentTimeMillis();
					//System.out.println("One Inference Iteration has taken "+(after-before)/1000+" secs");
					before= System.currentTimeMillis();
					likelihood = compute_likelihood(docIndex,phi);
					after= System.currentTimeMillis();
					//System.out.println("computeLikelihood has taken "+(after-before)/1000+" secs");
					assert(!Double.isNaN(likelihood));
					converged = (likelihood_old - likelihood) / likelihood_old;
					likelihood_old = likelihood;

					//System.out.println(String.format("[LDA INF] %8.5f %1.3e", likelihood, converged));
					//Utilities.printArray("Gamma:", var_gamma[docIndex]);
					//Utilities.printArray(phi);
					//if(var_iter==2)
					//System.out.println();
				}
				//Utilities.printArray(phi);
			}
			after= System.currentTimeMillis();
			//System.out.println("One Doc has taken "+var_iter+" Iterations in "+(after-before)/1000+" secs");
			//Utilities.printArray("Gamma:", var_gamma[docIndex]);
			for (int n = 0; n < total; n++)
			{
				for (int k = 0; k < K; k++)
				{
					class_word[k][docs[docIndex][n]] += phi[n][k];
					class_total[k] += phi[n][k];
				}
			}
			return (new SLDAVB()).new EStepOutput(docIndex, likelihood, phi, threadIndex);
		}
	}

}
