import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class LogReg {
	int ITERATIONS = 1000;
	double[] weights;
	ArrayList<Integer> docwordcounts = new ArrayList<Integer>();
	LogReg(int size)
	{
		weights = new double[size];
		Arrays.fill(weights, 0);
		
	}
	void initWeights(int size)
	{
		weights = new double[size];
		Arrays.fill(weights, 0.1);
	}
	int getWordCounts(String fileName, HashMap<String, Integer> vocab, HashMap<String, Integer>counts)
	{
		int uniquewordcount=0;
		 try {
             // FileReader reads text files in the default encoding.
             FileReader fileReader = new FileReader(fileName);

             // Always wrap FileReader in BufferedReader.
             BufferedReader bufferedReader = new BufferedReader(fileReader);
             String line;
             for(String t : vocab.keySet())
             {
            	 counts.put(t, 0);
             }
             while((line = bufferedReader.readLine()) != null) {
            	 String[] words = line.split(" ");
            	 for(String s : words)
            	 {
            		 if(counts.containsKey(s))
            		 {
            			 if(counts.get(s)==0)
            				 uniquewordcount++;
            			 counts.put(s,counts.get(s)+1);
            		 }
            		
            	 }
                // System.out.println(line);
             }   

             // Always close files.
             bufferedReader.close();         
         }
         catch(FileNotFoundException ex) {
             System.out.println("Unable to open file '" + fileName + "'");                
         }
         catch(IOException ex) {
             System.out.println("Error reading file '" + fileName + "'");                  
             // Or we could just do this: 
             // ex.printStackTrace();
         }
		 return uniquewordcount;
	}
	void computeXs(String dirname, HashMap<String, Integer>vocab, ArrayList<HashMap<String, Integer>> x)
	{
		File dir = new File(dirname);
		File[] directoryListing = dir.listFiles();
		  if (directoryListing != null) {
		    for (File child : directoryListing) {
		      // Do something with child
		    	String fileName = child.getName();
		    	HashMap<String, Integer> hm = new HashMap<String, Integer>();
		    	int temp = getWordCounts(dirname+"\\"+fileName, vocab, hm);
		    	docwordcounts.add(temp);
		    	x.add(hm);
		    }
		  }
	}
	
	//overloaded
	void computeXs(String dirname, HashMap<String, Integer>vocab, Integer[][] x)
	{
		File dir = new File(dirname);
		File[] directoryListing = dir.listFiles();
		int j=0;
		  if (directoryListing != null) {
		    for (File child : directoryListing) {
		      // Do something with child
		    	String fileName = child.getName();
		    	HashMap<String, Integer> hm = new HashMap<String, Integer>();
		    	int temp = getWordCounts(dirname+"\\"+fileName, vocab, hm);
		    	docwordcounts.add(temp);

		  //  	for(String s : hm.keySet())
		  //  		System.out.println(s+" "+hm.get(s));
		    	x[j] = (Integer[]) (hm.values().toArray(new Integer[hm.values().size()]));
		    	j++;
		    }
		  }
	}
	
	
	
	double sigmoid(double z) {
		return 1.0 / (1.0 + Math.exp(z));
	}
	
	
	double classify(HashMap<String, Integer> m)
	{
		double pred = .0;
		int i=0;
		for (String t : m.keySet())  {
			pred += weights[i] * m.get(t);
			i++;
		}
		return sigmoid(pred);
	}
	
	
	///overloaded
	double classify(Integer[] m)
	{
		double pred = .0;
		int i=0;
		Integer tmp=0;
		try{
		for (Integer t : m)  {
			if(t!= null)
			{
				tmp=t;
				pred += weights[i] * t;
				i++;
			}
		}
	//	System.out.println("Pred: "+ pred);
		}
		catch(NullPointerException ex)
		{
			System.out.println("Exception" + i + tmp);
		}
		return sigmoid(pred);
	}
	
	void train_lr(String dirname, HashMap<String, Integer> vocab, double rate, double lamda, int categ)
	{
		//ArrayList<HashMap<String, Integer>> x = new ArrayList<HashMap<String, Integer>>();
		File dir = new File(dirname);
		int size = dir.listFiles().length;
		Integer[][] x = new Integer[size][vocab.size()];
		/*
		try {
			System.setOut(new PrintStream(new BufferedOutputStream(new FileOutputStream("out.txt"))));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		*/
		computeXs(dirname, vocab, x);
		//initWeights(vocab.size());
		double prior = (double)x.length/826;
	/*	
		for(int iter=0; iter<100; iter++)
		{
			for(HashMap<String, Integer> hm : x)
			{
					 
				double predicted = classify(hm);
				int label = categ;
				int i=0;
				for(String t : hm.keySet())
				{
					weights[i] = weights[i] + rate * (predicted - label)/(prior - label) * hm.get(t) - rate*lamda*weights[i];
					i++;
				}
			}
		}
		*/
	//batch ascent	
	/*
		double temp =0;
		HashMap<String, Integer> hm = x.get(0);
		int i=0;
			for(String t : hm.keySet())
			{
				for(int l=0; l<x.size(); l++)
				{
					double prediction = classify(x.get(l));
					temp += (double)x.get(l).get(t) * (double)(categ - prediction)/ (double)(prior - prediction);
				}
				weights[i] += (weights[i] + rate*temp - (rate*lamda*weights[i]));
				i++;
			}
		System.out.println("Done 1");
		*/
		 
	/*	
		System.out.println("weights");
		for(double w : weights)
			System.out.println(w);
	*/	
		//Using arrays
	/*	
		int i=0, j=0;
		try{
		for(i=0;i<x[0].length-1;i++)
		{
			double temp = 0;
			for(j=0; j<x.length-1; j++)
			{
				double prediction = classify(x[j]);
				temp += (double)x[j][i] * (double)(categ - prediction)/ (double)(prior - prediction);
			}
			weights[i] += (weights[i] + rate*temp - (rate*lamda*weights[i]));

		}
		
	}
	
	catch(Exception ex)
	{
		System.out.println("Caught exception");
	}
	*/
		for(int iter=0; iter<300; iter++)
		{
			for(int i=0; i<x.length; i++)
			{
			    for(int j=0;j<x[0].length; j++)
			    {
			    	if(x[i][j]!=0)
			    	{
			    		double predicted = classify(x[i]);
			    		if(categ == 0)
			    			predicted = 1-predicted;
				    	//System.out.println("Prediction: "+predicted );
						int label = categ;
						//weights[j] = weights[j] + rate * (predicted - label)/docwordcounts.get(i) * x[i][j] - rate*lamda*weights[j];
						weights[j] = weights[j] + (rate * (label - predicted) * x[i][j]) - (rate*lamda*weights[j]*weights[j]);
			    	}
			    }
			}
		}
	
	}
	void test_lr(String dirname, HashMap<String, Integer> vocab, int[] preds )
	{
		File dir = new File(dirname);
		File[] directoryListing = dir.listFiles();
		  if (directoryListing != null) {
		    for (File child : directoryListing) {
		    	String fileName = child.getName();
		    	HashMap<String, Integer> testCounts = new HashMap<String, Integer>();
		    	getWordCounts(dirname+"//"+fileName,vocab, testCounts);
		    	double prediction = classify(testCounts);
		    	//System.out.println(prediction);
		    	if(prediction>0.5)
		    		preds[1]++;
		    	else
		    		preds[0]++;
		    }
	}
	}

}
