import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

public class NaieveBayes {
	static int NO_OF_CLASSES =2;
	
	
	HashMap<String, Double> condprob_s = new HashMap<String, Double>();
	HashMap<String, Double> condprob_h = new HashMap<String, Double>();
	ArrayList<HashMap<String, Integer>> classifiedvocab = new ArrayList<HashMap<String, Integer>>();
	double[] prior = new double[]{0,0};

	
	void mergeVocab(HashMap<String, Integer> hm1, HashMap<String, Integer> hm2, HashMap<String, Integer> hm3)
	{
		hm3.putAll(hm1);
		for(String key : hm2.keySet())
		{
			if(hm3.containsKey(key))
			{
				Integer temp = hm2.get(key);
				temp += hm2.get(key);
				hm3.put(key, temp);
			}
			else
				hm3.put(key, hm2.get(key));
		}
		
	}
	
	void extractVocabulary(String dirname, HashMap<String, Integer> hm)
	{
		File dir = new File(dirname);
		File[] directoryListing = dir.listFiles();
		  if (directoryListing != null) {
		    for (File child : directoryListing) {
		      // Do something with child
		    	String fileName = child.getName();
		    	String line;
		    	//System.out.println(child.getName());
		    	 try {
		             // FileReader reads text files in the default encoding.
		             FileReader fileReader = new FileReader(dirname+"\\"+fileName);

		             // Always wrap FileReader in BufferedReader.
		             BufferedReader bufferedReader = new BufferedReader(fileReader);

		             while((line = bufferedReader.readLine()) != null) {
		            	 String[] words = line.split(" ");
		            	 for(String s : words)
		            	 {
		            		 if(hm.containsKey(s))
		            		 {
		            			 hm.put(s,hm.get(s)+1);
		            		 }
		            		 else
		            		 {
		            			 hm.put(s,1);
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

		    }
		  }
	}
	
	int countdocs(String dirname)
	{
		int count=0;
		File dir = new File(dirname);
		  File[] directoryListing = dir.listFiles();
		  if (directoryListing != null) {
		    for (File child : directoryListing) {
		      // Do something with child
		    	count++;
		    }
		  }
		  return count;
	}
	
	int getWordCount(HashMap<String, Integer> vocab)
	{
		int count=0;
		for(String s: vocab.keySet())
		{
			count+= vocab.get(s);
		}
		return count;
	}
	
	private Integer countTokensOfTerm(String key, HashMap<String, Integer> hashMap) {
		// TODO Auto-generated method stub
		if(hashMap.containsKey(key))
			return hashMap.get(key);
		else
			return 0;
	}
	
	private void extractTokens(String filename, HashMap<String, Integer> tokens)
	{
		 try {
             // FileReader reads text files in the default encoding.
			 String line;
             FileReader fileReader = new FileReader(filename);

             // Always wrap FileReader in BufferedReader.
             BufferedReader bufferedReader = new BufferedReader(fileReader);

             while((line = bufferedReader.readLine()) != null) {
            	 String[] words = line.split(" ");
            	 for(String s : words)
            	 {
            		 if(tokens.containsKey(s))
            			 tokens.put(s, tokens.get(s)+1);       		
            		 else
            			 tokens.put(s, 1);
            	 }
                // System.out.println(line);
             }   

             // Always close files.
             bufferedReader.close();         
         }
         catch(FileNotFoundException ex) {
             System.out.println("Unable to open file '" + filename + "'");                
         }
         catch(IOException ex) {
             System.out.println("Error reading file '" + filename + "'");                  
             // Or we could just do this: 
             // ex.printStackTrace();
         }
	}
	
	void removeStopWords(HashMap hm , String filename)
	{
		try {
            // FileReader reads text files in the default encoding.
			 String line;
            FileReader fileReader = new FileReader(filename);

            // Always wrap FileReader in BufferedReader.
            BufferedReader bufferedReader = new BufferedReader(fileReader);

            while((line = bufferedReader.readLine()) != null) {
            	if(hm.containsKey(line))
            	{
            		hm.remove(line);
            	}
               // System.out.println(line);
            }   

            // Always close files.
            bufferedReader.close();         
        }
        catch(FileNotFoundException ex) {
            System.out.println("Unable to open file '" + filename + "'");                
        }
        catch(IOException ex) {
            System.out.println("Error reading file '" + filename + "'");                  
            // Or we could just do this: 
            // ex.printStackTrace();
        }
		
	}
	//////////////////////////////////////////////////////////////
	void train(NaieveBayes nb, HashMap<String, Integer> hamvocab, HashMap<String, Integer> spamvocab, HashMap<String, Integer> vocab, int stopw, String stopwordfile, String traindir )
	{
		nb.extractVocabulary(traindir+"\\ham", hamvocab);
		nb.extractVocabulary(traindir+"\\spam", spamvocab);
		
		if(stopw == 1)
			{
			removeStopWords(hamvocab, stopwordfile);
			removeStopWords(spamvocab, stopwordfile);
			}
		
		nb.mergeVocab(spamvocab, hamvocab, vocab);
		nb.classifiedvocab.add(spamvocab);
		nb.classifiedvocab.add(hamvocab);

		int[] N = new int[]{0,0};
		N[0] = nb.countdocs(traindir+"\\spam");
		N[1] = nb.countdocs(traindir+"\\ham");
		for(int i=0; i<NO_OF_CLASSES; i++)
		{
			nb.prior[i]= (double)N[i]/(double)(N[0]+N[1]);
			//conditional probabilities for each word to be a spam word
			
			//Tct is the number of occurrences of t in training documents from class c, including multiple occurrences of a term in a document
			for(String key : vocab.keySet())
			{
				int tct = countTokensOfTerm(key, nb.classifiedvocab.get(i));
				Double prob;
				int occurances = 0;
				if(nb.classifiedvocab.get(i).containsKey(key))
					{
					occurances = nb.classifiedvocab.get(i).get(key);
					}
				prob = (double) ((double)(occurances+1)/(double)(nb.getWordCount(nb.classifiedvocab.get(i)) + nb.classifiedvocab.get(i).size()));
				if(i==0)
					nb.condprob_s.put(key, prob);
				else
					nb.condprob_h.put(key, prob);
				//System.out.println(key+" "+prob);
		    }
			//Add 1/count for unknown words
			Double unk = (double) ((double)(1)/(double)(nb.getWordCount(nb.classifiedvocab.get(i)) + nb.classifiedvocab.get(i).size()));
			if(i==0)
				nb.condprob_s.put("unk", unk);
			else
				nb.condprob_h.put("unk", unk);
		}
	}
	
	int[] test(NaieveBayes nb, String dirname, String stopfilename)
	{
		int[] countpred = new int[]{0,0};
		File dir = new File(dirname);
		  File[] directoryListing = dir.listFiles();
		  if (directoryListing != null) {
		    for (File child : directoryListing) {
		    	int pred = nb.applyMultinomialNB(nb, dirname+"\\"+child.getName(), stopfilename);
		    	countpred[pred]++;
				//System.out.println("Predicted class "+ pred);		   
		    	}
		  }
		return countpred;
	}

	int applyMultinomialNB(NaieveBayes nb, String filename, String stopfilename)
	{
		HashMap<String, Integer> tokens = new HashMap<String, Integer>();
		extractTokens(filename, tokens);
		if(stopfilename!=null)
			removeStopWords(tokens, stopfilename);
		double[] score = new double[]{0,0};
		for(int i=0; i<NO_OF_CLASSES ; i++)
		{
			score[i] = Math.log(nb.prior[i]);
			for(String t : tokens.keySet())
			{
				if(i==0)
				{
					if(nb.condprob_s.containsKey(t))
						score[i]+= Math.log(nb.condprob_s.get(t));
					else 
					{
						score[i]+= Math.log(nb.condprob_s.get("unk"));	//if word not found in train set is encountered
					}
				}
				else
				{
					if(nb.condprob_h.containsKey(t))
						score[i]+= Math.log(nb.condprob_h.get(t));
					else
						score[i]+= Math.log(nb.condprob_h.get("unk"));
				}
			}
		}
		if(score[0]>score[1])
			return 0;
		else
			return 1;
	}
	
}
