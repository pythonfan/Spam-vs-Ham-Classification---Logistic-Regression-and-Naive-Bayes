import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;
public class MainClass {

	static HashMap<String, Integer> spamvocab = new HashMap<String, Integer>();
	static HashMap<String, Integer> hamvocab = new HashMap<String, Integer>();
	static HashMap<String, Integer> vocab = new HashMap<String, Integer>();
	
	
	public static void main(String args[])
	{
		
		NaieveBayes nb = new NaieveBayes();
		int[] preds = new int[]{0,0};
		int totalcorrect = 0, totalfiles =0;
		
		String traindir, testdir , stopwordsfile;
		double eta, lambda;
		if(args.length==0)
		{
			traindir = "C:\\Users\\Shakti\\Downloads\\hw2_train\\train";
			testdir =  "C:\\Users\\Shakti\\Downloads\\hw2_test\\test";
			stopwordsfile = "stopwords.txt";
			eta = 0.01;
			lambda = 0.05;
		}
		else
		{
			traindir = args[0];
			testdir = args[1];
			stopwordsfile = args[2];
			eta = Double.parseDouble(args[3]);
			lambda = Double.parseDouble(args[4]);
		}
		
		nb.train(nb, hamvocab, spamvocab, vocab, 0, null, traindir);
		
		preds = nb.test(nb, testdir+"\\ham", null);
		System.out.println("Ham: "+preds[1]+ " Spam: "+preds[0]);
		System.out.println("Accuracy: "+ preds[1]/(float)(preds[0]+preds[1]));
		totalcorrect += preds[1];
		totalfiles += preds[1]+preds[0];
		preds = nb.test(nb, testdir+"\\spam", null);
		System.out.println("Ham: "+preds[1]+ "Spam: "+preds[0]);
		System.out.println("Accuracy: "+ preds[0]/(float)(preds[0]+preds[1]));
		totalcorrect += preds[0];
		totalfiles += preds[1]+preds[0];
		System.out.println("Overall Accuracy: "+ totalcorrect/(float)totalfiles);

		 
		
		System.out.println("Logistic regression");
		totalcorrect = 0;
		totalfiles = 0;
		LogReg lr = new LogReg(vocab.size());
		System.out.println("Train spam");

		lr.train_lr(traindir+"\\spam", vocab, eta, lambda,0);

		System.out.println("Ham");
		lr.train_lr(traindir+"\\ham", vocab, eta, lambda,1);

		lr.test_lr(testdir+"\\ham", vocab, preds);
		System.out.println("Ham: "+preds[1]+ "Spam: "+preds[0]);
		System.out.println("Accuracy: "+ preds[1]/(float)(preds[0]+preds[1]));
		totalcorrect += preds[1];
		totalfiles += preds[1]+preds[0];
		System.out.println("Spam");

		lr.test_lr(testdir+"\\spam", vocab, preds);
		System.out.println("Ham: "+preds[1]+ "Spam: "+preds[0]);
		System.out.println("Accuracy: "+ preds[0]/(float)(preds[0]+preds[1]));
		totalcorrect +=preds[0];
		totalfiles += preds[1]+preds[0];
		
		System.out.println("Overall Accuracy: "+ totalcorrect/(float)totalfiles);
		
		////////////////////////////////////////////////////////////////////////////////////
		vocab.clear();
		spamvocab.clear();
		hamvocab.clear();
		//After removing stopwords
		
		NaieveBayes nb1 = new NaieveBayes();
		int[] preds1 = new int[]{0,0};
		int totalcorrect1 = 0, totalfiles1 =0;
		nb1.train(nb1, hamvocab, spamvocab, vocab, 1, stopwordsfile, traindir);
		
		preds1 = nb1.test(nb1, testdir+"\\ham", stopwordsfile);
		System.out.println("Ham: "+preds1[1]+ " Spam: "+preds1[0]);
		System.out.println("Accuracy: "+ preds1[1]/(float)(preds1[0]+preds1[1]));
		totalcorrect1 += preds1[1];
		totalfiles1 += preds1[1]+preds1[0];
		preds1 = nb1.test(nb1, testdir+"\\spam", stopwordsfile);
		System.out.println("Ham: "+preds1[1]+ "Spam: "+preds1[0]);
		System.out.println("Accuracy: "+ preds1[0]/(float)(preds1[0]+preds1[1]));
		totalcorrect1 += preds1[0];
		totalfiles1 += preds1[1]+preds1[0];
		System.out.println("Overall Accuracy: "+ totalcorrect1/(float)totalfiles1);

		 
		
		System.out.println("Logistic regression");
		totalcorrect1 = 0;
		totalfiles1 = 0;
		LogReg lr1 = new LogReg(vocab.size());
		System.out.println("Train spam");

		lr1.train_lr(traindir+"\\spam", vocab, eta, lambda,0);

		System.out.println("Ham");
		lr1.train_lr(traindir+"\\ham", vocab, eta, lambda,1);

		lr1.test_lr(testdir+"\\ham", vocab, preds1);
		System.out.println("Ham: "+preds1[1]+ "Spam: "+preds1[0]);
		System.out.println("Accuracy: "+ preds1[1]/(float)(preds1[0]+preds1[1]));
		totalcorrect1 += preds1[1];
		totalfiles1 += preds1[1]+preds1[0];
		System.out.println("Spam");

		lr1.test_lr(testdir+"\\spam", vocab, preds1);
		System.out.println("Ham: "+preds1[1]+ "Spam: "+preds1[0]);
		System.out.println("Accuracy: "+ preds1[0]/(float)(preds1[0]+preds1[1]));
		totalcorrect1 +=preds1[0];
		totalfiles1 += preds1[1]+preds1[0];
		
		System.out.println("Overall Accuracy: "+ totalcorrect1/(float)totalfiles1);

	}
	

}
