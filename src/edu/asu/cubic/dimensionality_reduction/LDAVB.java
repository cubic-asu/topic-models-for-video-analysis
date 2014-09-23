package edu.asu.cubic.dimensionality_reduction;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;



import edu.asu.cubic.util.LDAVBBuilder;
import edu.asu.cubic.util.Utilities;

/**
 * This class contains Variational Bayes estimation of LDA. An implementation of LDA-C by Blei in Java 
 * @author prasanthl
 *
 */
public class LDAVB implements Serializable, Cloneable {

	private static final long serialVersionUID = 5361536010680117719L;
	public static void main(String[] args) throws IOException {
		//LDAVBBuilder.extractLDAFeatures("Parameters.properties");
	}
	double alpha;
	double eta;
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
	ArrayList<HashMap<Integer,Integer>> z;
	/* documents in term:count format */
	ArrayList<HashMap<Integer,Integer>> documents;
	int[] docLengths;
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

	public LDAVB(){}

	public LDAVB(int[][] docs, int V, int K, double alpha, double eta, String mPath, String bmName, int EMIters, double EMConverged,
			int VarIters, double VarConverged, boolean writeIters) {
		this.V = V;
		this.K = K;
		this.alpha = alpha;
		modelPath= mPath;
		baseModelName= bmName;
		EMMaxIters= EMIters;
		this.EMConverged= EMConverged;
		VarMaxIters= VarIters;
		this.VarConverged= VarConverged;
		writeIterations= writeIters;
		documents=  new ArrayList<HashMap<Integer,Integer>>();
		docLengths= new int[docs.length];
		//z= new int[docs.length][];
		// convert the docs to term:count format
		for(int doc=0; doc<docs.length; doc++){
			HashMap<Integer,Integer> map= new HashMap<Integer, Integer>();
			//z[doc]= new int[docs[doc].length];
			try{
				for(int v=0; v<docs[doc].length; v++){
					if(map.containsKey(docs[doc][v])){
						map.put(docs[doc][v],map.get(docs[doc][v])+1);
					}
					else{
						map.put(docs[doc][v],1);
					}
				}
			}
			catch(NullPointerException npe){
				System.out.println("Document "+(doc+1)+" is null");
				npe.printStackTrace();
				System.exit(1);
			}
			documents.add(map);
			docLengths[doc]= docs[doc].length;
			//System.out.println(map);
		}

		likelihoods= new ArrayList<Double>();
	}

	public LDAVB(int[][] docs, LDAVB model, String mPath, String bmName, int VarIters, double VarConverged) {
		modelPath= mPath;
		baseModelName= bmName;
		VarMaxIters= VarIters;
		this.VarConverged= VarConverged;
		documents=  new ArrayList<HashMap<Integer,Integer>>();
		z= new ArrayList<HashMap<Integer,Integer>>();
		docLengths= new int[docs.length];
		// convert the docs to term:count format
		for(int doc=0; doc<docs.length; doc++){
			HashMap<Integer,Integer> map= new HashMap<Integer, Integer>();
			HashMap<Integer,Integer> map1= new HashMap<Integer, Integer>();
			for(int v=0; v<docs[doc].length; v++){
				if(map.containsKey(docs[doc][v])){
					map.put(docs[doc][v],map.get(docs[doc][v])+1);
				}
				else{
					map.put(docs[doc][v],1);
					map1.put(docs[doc][v],0); // default assigned to topic 0
				}
			}
			documents.add(map);
			z.add(map1);
			docLengths[doc]= docs[doc].length;
			//System.out.println(map);
		}
		K= model.K;
		V= model.V;
		alpha= model.alpha;
		eta= model.eta;
		log_prob_w= model.getBeta();
	}

	/**
	 * This method cleans up all unnecessary variables so that the object becomes much lighter
	 */
	public void cleanUpVariables(){
		documents= null;
		//class_total= null;
		//class_word= null;
		docLengths= null;
		//phi= null;
	}

	/**
	 * This method cleans up all unnecessary variables so that the object becomes much lighter
	 */
	public void cleanUpVariablesTemp(){
		documents= null;
		docLengths= null;
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
		HashMap<Integer,Integer> currDoc= documents.get(doc);
		int total= currDoc.size();
		int[] termsInDoc= new int[total];
		int[] counts= new int[total];
		int index=0;
		for(Integer key: currDoc.keySet()){
			termsInDoc[index]= key;
			counts[index]= currDoc.get(key);
			index++;
		}
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
					likelihood += counts[n]*
							(phi[n][k]*((dig[k] - digsum) - Math.log(phi[n][k])
									+ log_prob_w[k][termsInDoc[n]]));
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

	/*
	 * perform inference on a document and update sufficient statistics
	 *
	 */
	double doc_e_step(int doc, double[][] phi)
	{
		double likelihood;
		//double gamma_sum = 0;
		HashMap<Integer, Integer> currDoc= documents.get(doc);
		int total= currDoc.size();
		int[] termsInDoc= new int[total];
		int[] counts= new int[total];
		int index=0;
		for(Integer key: currDoc.keySet()){
			termsInDoc[index]= key;
			counts[index]= currDoc.get(key);
			index++;
		}

		//Utilities.printArray("", var_gamma[doc]);
		// posterior inference

		likelihood = lda_inference(doc,phi);

		// update sufficient statistics
		/*for (int k = 0; k < K; k++)
	    {
	        gamma_sum += var_gamma[doc][k];
	        ss->alpha_suffstats += digamma(gamma[k]);
	    }
	    ss->alpha_suffstats -= model->num_topics * digamma(gamma_sum);*/

		for (int n = 0; n < total; n++)
		{
			for (int k = 0; k < K; k++)
			{
				class_word[k][termsInDoc[n]] += counts[n]*phi[n][k];
				class_total[k] += counts[n]*phi[n][k];
			}
		}
		//Utilities.printArray("",class_total);
		//ss->num_docs = ss->num_docs + 1;

		return(likelihood);
	}

	public ArrayList<HashMap<Integer, Integer>> getZ(){
		return z;
	}
	
	public double[][] getBeta(){
		return log_prob_w;
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
	public void infer(String filename) throws IOException
	{
		double likelihood;
		int numDocs= documents.size();
		int maxLength= documents.get(0).size();
		for(int doc=0; doc<numDocs; doc++){
			if(documents.get(doc).size()>maxLength)
				maxLength= documents.get(doc).size();
		}
		var_gamma= new double[numDocs][K];
		//phi= new double[numDocs][maxLength][K];
		likelihoods= new ArrayList<Double>();
		//PrintWriter likelihoodFile= new PrintWriter(new File(modelPath+"/"+filename+"-lda-lhood.dat"));
		//PrintWriter tempPhiFile= new PrintWriter(new FileWriter(modelPath+"//"+baseModelName+"Temp.txt",true));
		//PrintWriter tempTermsFile= new PrintWriter(new FileWriter(modelPath+"//"+baseModelName+"TempTerms.txt",true));
		for (int d = 0; d < numDocs; d++)
		{
			//if (((d % 100) == 0) && (d>0)) System.out.println("document "+d+"\n");
			double[][] phi= new double[documents.get(d).size()][K];
			likelihood = lda_inference(d,phi);
			// find the best topic for each word using phi
			HashMap<Integer,Integer> temp= z.get(d);
			int index=0;
			int[] termsInDoc= new int[documents.get(d).size()];
			for(Integer key: documents.get(d).keySet()){
				termsInDoc[index]= key;
				index++;
			}
			for(int v=0; v<documents.get(d).size(); v++){
				//Utilities.printArray("", phi[v]);
				int topic= Utilities.max(phi[v],1)[0];
				temp.put(termsInDoc[v], topic);
			}
			z.set(d, temp);
			likelihoods.add(likelihood);
		}
		
		// if any of the documents is empty then set gamma to NaN
		for(int d=0; d< documents.size(); d++){
			if(documents.get(d).isEmpty()){
				for(int k=0; k<K; k++)
					var_gamma[d][k]= Double.NaN;
			}
		}
	}

	/*
	 * variational inference
	 *
	 */
	double lda_inference(int doc, double[][] phi)
	{
		double converged = 1;
		double phisum = 0, likelihood = 0;
		double likelihood_old = 0;
		double before, after;
		double[] oldphi= new double[K];
		int var_iter;
		double[] digamma_gam=new double[K];
		HashMap<Integer, Integer> currDoc= documents.get(doc);
		int total= currDoc.size();// this the the toal unique words
		int[] termsInDoc= new int[total];
		int[] counts= new int[total];
		int index=0;
		for(Integer key: currDoc.keySet()){
			termsInDoc[index]= key;
			counts[index]= currDoc.get(key);
			index++;
		}
		// compute posterior dirichlet
		for (int k = 0; k < K; k++)
		{
			var_gamma[doc][k] = alpha + (docLengths[doc]/((double) K));
			digamma_gam[k] = Utilities.digamma(var_gamma[doc][k]);
			for (int n = 0; n < total; n++)
				phi[n][k] = 1.0/K;
		}
		var_iter = 0;
		if(!documents.get(doc).isEmpty()){
			//Utilities.printArray("Before ", var_gamma[doc]);
			while ((converged > VarConverged) &&
					((var_iter < VarMaxIters) || (VarMaxIters == -1)))
			{
				before= System.currentTimeMillis();
				/*for (int n = 0; n < total; n++){
				System.out.println(log_prob_w[0][termsInDoc[n]]);
			}*/
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
						phi[n][k] = digamma_gam[k] + log_prob_w[k][termsInDoc[n]];

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
								var_gamma[doc][k] + counts[n]*(phi[n][k] - oldphi[k]);
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
				likelihood = compute_likelihood(doc,phi);
				after= System.currentTimeMillis();
				//System.out.println("computeLikelihood has taken "+(after-before)/1000+" secs");
				assert(!Double.isNaN(likelihood));
				converged = (likelihood_old - likelihood) / likelihood_old;
				likelihood_old = likelihood;

				//System.out.println(String.format("[LDA INF] %8.5f %1.3e", likelihood, converged));
				//Utilities.printArray("Gamma:", var_gamma[doc]);
				//Utilities.printArray(phi[doc]);
				//if(var_iter==2)
				//System.out.println();
			}
		}
		//Utilities.printArray("Gamma:", var_gamma[doc]);
		return(likelihood);
	}
	/*
	 * compute MLE lda model from sufficient statistics
	 *
	 */
	void lda_mle()
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
	}
	public void load_lda_model() throws IOException
	{

		FileReader inputFile= new FileReader(new File(modelPath+"/"+baseModelName+".other"));
		System.out.println("loading "+modelPath+"/"+baseModelName+".other\n");
		Scanner scanner= new Scanner(inputFile);
		scanner.next();K= scanner.nextInt();
		scanner.next();V= scanner.nextInt();
		scanner.next();alpha= scanner.nextDouble();
		inputFile.close();

		inputFile= new FileReader(new File(modelPath+"/"+baseModelName+".beta"));
		System.out.println("loading "+modelPath+"/"+baseModelName+".beta\n");
		scanner= new Scanner(inputFile);
		log_prob_w= new double[K][V];
		for (int i = 0; i < K; i++)
		{
			for (int j = 0; j < V; j++)
			{
				log_prob_w[i][j] = scanner.nextDouble();
			}
		}
		inputFile.close();
	}

	public void random_initialize_ss() throws IOException{

		int numDocs= documents.size();
		int maxLength= documents.get(0).size();
		for(int doc=0; doc<numDocs; doc++){
			if(documents.get(doc).size()>maxLength)
				maxLength= documents.get(doc).size();
		}
		System.out.println("MaxLength is:"+maxLength);
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
		}
	}
	public void runEM() throws Exception{
		// initialize model
		random_initialize_ss();
		double before, after;
		lda_mle();
		if(writeIterations)
			save_lda_model("000");
		double converged= 1;
		int iter= 0;
		double likelihood= 0,likelihood_old = 0;

		for(int i=1; i<EMMaxIters;i++){
			if(new File(modelPath+"/"+baseModelName+"_"+(i)+".model").exists()){
				// if the model for this exists
				iter= i;
				LDAVB model= (LDAVB)new ObjectInputStream(new FileInputStream(modelPath+"/"+baseModelName+"_"+(i)+".model")).readObject();
				class_total= model.class_total;
				class_word= model.class_word;
				log_prob_w= model.log_prob_w;
				likelihoods= model.likelihoods;
				likelihood_old= likelihoods.get(likelihoods.size()-1);
			}
		}
		PrintWriter likelihoodFile= new PrintWriter(new File(modelPath+"/"+baseModelName+".likelihood"));
		while (((converged < 0) || (converged > EMConverged) || (iter <= 2)) && (iter <= EMMaxIters)){
			iter++; System.out.println("EM iteration "+ iter);
			likelihood = 0;
			zero_initialize_ss();
			// e-step
			for (int d = 0; d < documents.size(); d++)
			{
				if ((d % 1000) == 0) System.out.println("\t Doc: "+(d+1)+" Likelihood "+likelihood);
				before= System.currentTimeMillis();
				double[][] phi= new double[documents.get(d).size()][K];
				likelihood += doc_e_step(d,phi);
				after= System.currentTimeMillis();
				//System.out.println("\tTime take in secs: "+(after-before)/1000);
				//Utilities.printArray("Gamma: ", var_gamma[d]);
			}
			// m-step
			lda_mle();

			// check for convergence
			converged = (likelihood_old - likelihood) / (likelihood_old);
			if (converged < 0) VarMaxIters = VarMaxIters * 2;
			likelihood_old = likelihood;

			// output model and likelihood

			System.out.println(String.format("\t %10.10f\t%5.5e\n", likelihood, converged));
			likelihoodFile.println(String.format("\t %10.10f\t%5.5e\n", likelihood, converged));
			likelihoods.add(likelihood);
			if ((iter % LAG) == 0 && writeIterations)
			{
				save_lda_model(String.format("%03d", iter));
				//save_gamma(String.format("%03d",iter));
			}

			if(new File(modelPath+"/"+baseModelName+"_"+(iter-1)+".model").exists()){
				// delete the model from prev iteration
				new File(modelPath+"/"+baseModelName+"_"+(iter-1)+".model").delete();
			}
			// make a clone of this object
			LDAVB clonedModel= (LDAVB)this.clone();
			clonedModel.cleanUpVariablesTemp();
			// write the object to file
			String modelFilePath= modelPath+"/"+baseModelName+"_"+(iter)+".model";
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelFilePath));
			oos.writeObject(clonedModel);
			oos.close();
		}
		// output the final model
		save_lda_model(baseModelName);
		save_gamma(baseModelName);
		if(new File(modelPath+"/"+baseModelName+"_"+(iter)+".model").exists()){
			// delete the model from prev iteration
			new File(modelPath+"/"+baseModelName+"_"+(iter)+".model").delete();
		}
		// output the word assignments (for visualization)
		likelihoodFile.close();
	}

	public void runParallelEM() throws Exception
	{
		int threads = Runtime.getRuntime().availableProcessors();
		System.out.println("Num of threads: "+threads);
		// initialize model
		random_initialize_ss();
		double before, after;
		lda_mle();
		if(writeIterations)
			save_lda_model("000");
		double converged= 1;
		int iter= 0;
		double likelihood= 0,likelihood_old = 0;

		for(int i=1; i<=EMMaxIters;i++){
			if(new File(modelPath+"/"+baseModelName+"_"+(i)+".model").exists()){
				// if the model for this exists
				iter= i;
				LDAVB model= (LDAVB)new ObjectInputStream(new FileInputStream(modelPath+"/"+baseModelName+"_"+(i)+".model")).readObject();
				class_total= model.class_total;
				class_word= model.class_word;
				log_prob_w= model.log_prob_w;
				likelihoods= model.likelihoods;
				likelihood_old= likelihoods.get(likelihoods.size()-1);
			}
		}
		//PrintWriter likelihoodFile= new PrintWriter(new File(modelPath+"/"+baseModelName+".likelihood"));
		while (((converged < 0) || (converged > EMConverged) || (iter <= 2)) && (iter <= EMMaxIters)){
			iter++; System.out.println("EM iteration "+ iter);
			likelihood = 0;
			zero_initialize_ss();
			// e-step
			for (int d = 0; d < documents.size(); d+=threads)
			{
				if ((d % 1000) == 0) System.out.println("\t Doc: "+(d+1)+" Likelihood "+likelihood);
				before= System.currentTimeMillis();
				//double[][] phi= new double[docLengths[d]][K];
				//likelihood += doc_e_step(d,phi);
				ExecutorService service = Executors.newFixedThreadPool(threads);
				List<Future<EStepOutput>> futures = new ArrayList<Future<EStepOutput>>();
				// Map
				int numThreads= Math.min(threads, documents.size()-d);
				for(int i=0; i<numThreads;i++){
					Callable<EStepOutput> mapper= this.new Mapper(d+i,i);
					futures.add(service.submit(mapper));
				}
				// Reduce
				for(Future<EStepOutput> futureoutput: futures){
					likelihood+= futureoutput.get().likelihood;
				}
				after= System.currentTimeMillis();
				service.shutdown();
				//System.out.println("\tTime take in secs: "+(after-before)/1000);
				//Utilities.printArray("Gamma: ", var_gamma[d]);
			}
			// m-step
			lda_mle();

			// check for convergence
			converged = (likelihood_old - likelihood) / (likelihood_old);
			if (converged < 0) VarMaxIters = VarMaxIters * 2;
			likelihood_old = likelihood;

			// output model and likelihood

			System.out.println(String.format("\t %10.10f\t%5.5e\n", likelihood, converged));
			//likelihoodFile.println(String.format("\t %10.10f\t%5.5e\n", likelihood, converged));
			likelihoods.add(likelihood);
			if ((iter % LAG) == 0 && writeIterations)
			{
				save_lda_model(String.format("%03d", iter));
				//save_gamma(String.format("%03d",iter));
			}

			if(new File(modelPath+"/"+baseModelName+"_"+(iter-1)+".model").exists()){
				// delete the model from prev iteration
				new File(modelPath+"/"+baseModelName+"_"+(iter-1)+".model").delete();
			}
			// make a clone of this object
			LDAVB clonedModel= (LDAVB)this.clone();
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
		if(new File(modelPath+"/"+baseModelName+"_"+(iter)+".model").exists()){
			// delete the model from prev iteration
			new File(modelPath+"/"+baseModelName+"_"+(iter)+".model").delete();
		}
		// before leaving check if there is any empty document if so set Gamma to NaN for that document
		for(int d=0; d< documents.size(); d++){
			if(documents.get(d).isEmpty()){
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
		//likelihoodFile.close();
	}

	/*
	 * saves the gamma parameters of the current dataset
	 *
	 */
	public void save_gamma(String iteration) throws IOException
	{
		PrintWriter gammaFile= new PrintWriter(new File(modelPath+"/"+iteration+".gamma"));

		for (int d = 0; d < documents.size(); d++)
		{
			for (int k = 0; k < K; k++)
			{
				gammaFile.print(String.format(" %5.10f", var_gamma[d][k]));
			}
			gammaFile.print("\n");
		}
		gammaFile.close();
	}

	/*
	 * save an lda model
	 *
	 */
	void save_lda_model(String iteration) throws IOException
	{
		PrintWriter betaFile= new PrintWriter(new File(modelPath+"/"+iteration+".beta"));
		//PrintWriter otherFile= new PrintWriter(new File(modelPath+"/"+iteration+".other"));
		//otherFile.print(String.format("num_topics %d\n", K));
		//otherFile.print(String.format("num_terms %d\n", V));
		//otherFile.print(String.format("alpha %5.10f\n", alpha));
		for (int i = 0; i < K; i++)
		{
			for (int j = 0; j < V; j++)
			{
				betaFile.print(String.format(" %.3f", log_prob_w[i][j]));
			}
			betaFile.println();
		}
		betaFile.close();
		//otherFile.close();
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
			double[][] phi= new double[documents.get(docIndex).size()][K];
			double converged = 1;
			double phisum = 0, likelihood = 0;
			double likelihood_old = 0;
			double before, after;
			double[] oldphi= new double[K];
			int var_iter;
			double[] digamma_gam=new double[K];
			HashMap<Integer, Integer> currDoc= documents.get(docIndex);
			int total= currDoc.size();
			int[] termsInDoc= new int[total];
			int[] counts= new int[total];
			int index=0;
			for(Integer key: currDoc.keySet()){
				termsInDoc[index]= key;
				counts[index]= currDoc.get(key);
				index++;
			}
			// compute posterior dirichlet
			for (int k = 0; k < K; k++)
			{
				var_gamma[docIndex][k] = alpha + (docLengths[docIndex]/((double) K));
				digamma_gam[k] = Utilities.digamma(var_gamma[docIndex][k]);
				for (int n = 0; n < total; n++)
					phi[n][k] = 1.0/K;
			}
			var_iter = 0;
			if(!currDoc.isEmpty()){
				//Utilities.printArray("Before ", var_gamma[doc]);
				while ((converged > VarConverged) &&
						((var_iter < VarMaxIters) || (VarMaxIters == -1)))
				{
					before= System.currentTimeMillis();
					/*for (int n = 0; n < total; n++){
					System.out.println(log_prob_w[0][termsInDoc[n]]);
				}*/
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
							try{
							phi[n][k] = digamma_gam[k] + log_prob_w[k][termsInDoc[n]];
							}
							catch(Exception e){
								e.printStackTrace();
							}
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
							var_gamma[docIndex][k] =
									var_gamma[docIndex][k] + counts[n]*(phi[n][k] - oldphi[k]);
							// !!! a lot of extra digamma's here because of how we're computing it
							// !!! but its more automatically updated too.
							digamma_gam[k] = Utilities.digamma(var_gamma[docIndex][k]);
						}
						//Utilities.printArray("", var_gamma[doc]);
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
					//Utilities.printArray("Gamma:", var_gamma[doc]);
					//Utilities.printArray(phi[doc]);
					//if(var_iter==2)
					//System.out.println();
				}
			}
			for (int n = 0; n < total; n++)
			{
				for (int k = 0; k < K; k++)
				{
					class_word[k][termsInDoc[n]] += eta+counts[n]*phi[n][k];
					class_total[k] += eta+counts[n]*phi[n][k];
				}
			}
			return (new LDAVB()).new EStepOutput(docIndex, likelihood, phi, threadIndex);
		}
	}
}
