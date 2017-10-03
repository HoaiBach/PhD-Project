package ExtractResults;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class RankTable {

	static String[] datasets = new String[]{
			"wine","australian", 
			"vehicle", "german",
			"wbcd", "ionosphere" ,
			"sonar", "hillvalley", 
			"musk1"
//			,"arrhythmia","madelon","isolet5"
			};
	
	static String[] inds = new String[]{"IGD","Volumes"};
	static List<String> methods = new ArrayList<String>();
	static int runs = 50;

	public static void main(String[] args) throws FileNotFoundException{
//		for(int rate = 50; rate<=90;rate = rate+10){
//			for(int noInterval = 3; noInterval <=9; noInterval=noInterval+2){
//				methods.add(rate+"-"+noInterval);
//			}
//		}

//		methods.add("NSGAII"); methods.add("NSGAIII");
//		methods.add("SPEA2"); methods.add("OMOPSO");
//		methods.add("MOEAD"); methods.add("MOEADREF");
//		methods.add("MOEADREF");
		methods.add("MOEADDYNRan");
		methods.add("MOEADDYNSeqAcc");methods.add("MOEADDYNSeqRan");
		methods.add("MOEADDYNDE");
		
		//read data
		Map<String,Map<String,Map<String,double[][]>>> results= new HashMap<String,Map<String,Map<String,double[][]>>>();
		for(String dataset: datasets){
			String dirDataset = "Processed/Full/Data/"+dataset;
			Map<String,Map<String,double[][]>> indMethod = new HashMap<String,Map<String,double[][]>>();
			results.put(dataset,indMethod);
			for(String ind: inds){
				String dirInd = dirDataset+"/"+ind;
				Map<String,double[][]> methodRun = new HashMap<String,double[][]>();
				indMethod.put(ind, methodRun);
				for(String method: methods){
					File runFile = new File(dirInd+"/"+method+".txt");
					Scanner sc = new Scanner(runFile);
					sc.nextLine();
					double[][] rr = new double[2][runs];//0: train, 1: test
					int index=0;
					while(sc.hasNextLine()){
						String line = sc.nextLine();
						String[] split = line.split(",");
						rr[0][index] = Double.parseDouble(split[0].trim());
						rr[1][index] = Double.parseDouble(split[1].trim());
						index++;
					}
					sc.close();
					methodRun.put(method, rr);
				}
			}
		}

		//now process
		String header = "\\begin{table}\n"
				+ "\\centering"				
				+ "\\begin{tabular}{llllll}\n"
				+ "\\\\\\hline\n"
				+ "Method & TrainIGD & TestIGD & TrainVolumes & TestVolumes & Average\\\\\\hline\n";
		String end = "\\end{tabular}\n";
		
		File out = new File("Processed/Full/Table/table.tex");
		PrintStream pt = new PrintStream(out);
		pt.println("\\documentclass{report}\n"
				+ "\\usepackage[cm]{fullpage}\n"
				+ "\\begin{document}\n");
		
		for(String dataset: datasets){
			//dataset -> indicator -> (method, train/test)
			Map<String,Map<String,double[][]>> datasetResult = results.get(dataset);
			
			String toPrint = header;
			
			//to store rank for each method on a dataset.
			Map<String,double[]> methodRank = new HashMap<String,double[]>();
			for(String method:methods){
				double[] ave = new double[4];
				//0-trI, 1-teI, 2-trV, 3-teV
				methodRank.put(method, ave);
			}

			int index = 0;
			for(String ind: inds){
				Map<String,double[][]> indResult = datasetResult.get(ind);
				//now sort
				List<Map<String,int[]>> sorts = rankTrainTest(indResult);
				Map<String,int[]> sortTrain = sorts.get(0);
				Map<String,int[]> sortTest = sorts.get(1);
				for(String method: methods){
					methodRank.get(method)[index] = average(sortTrain.get(method));
					methodRank.get(method)[index+1] = average(sortTest.get(method));
				}
				index = index+2;
			}
			
			//now print to the table
			DecimalFormat f = new DecimalFormat("##.00");
			for(String method: methods){
				double[] ave = methodRank.get(method);
				toPrint += method+
						" & "+f.format(ave[0])+" & "+f.format(ave[1])+
						" & "+f.format(ave[2])+" & "+f.format(ave[3])+" & "+ f.format((ave[0]*0+ave[1]*1+ave[2]*0+ave[3]*1)/2)+"\\\\\\hline\n";
				
			}

			toPrint+= end;
			toPrint+= "\\caption{"+dataset+"}\n";
			toPrint+= "\\end{table}\n";
			
			pt.println(toPrint);
		}
		
		pt.println("\\end{document}\n ");
	}

	public static double average(int[] values){
		double sum = 0;
		for(int i=0;i<values.length;i++){
			sum+= values[i];
		}
		return sum/values.length;
	}
	

	public static List<Map<String,int[]>> rankTrainTest(Map<String,double[][]> results){
		Map<String,int[]> rankTrain = new HashMap<String,int[]>();
		Map<String,int[]> rankTest = new HashMap<String,int[]>();
		for(String method: methods){
			int[] rankRunTrain = new int[runs];
			rankTrain.put(method, rankRunTrain);
			int[] rankRunTest = new int[runs];
			rankTest.put(method, rankRunTest);
		}

		for(int run=0;run < runs;run++){
			double[] test = new double[methods.size()];
			double[] train = new double[methods.size()];
			String[] testN = new String[methods.size()];
			String[] trainN = new String[methods.size()];

			for(int i=0;i<methods.size();i++){
				testN[i] = trainN[i] = methods.get(i);
				train[i] = results.get(methods.get(i))[0][run];
				test[i] = results.get(methods.get(i))[1][run];
			}

			sort(train,trainN);sort(test,testN);
			for(int i=0;i<trainN.length;i++){
				rankTrain.get(trainN[i])[run] = i;
				rankTest.get(testN[i])[run] = i;
			}
		}

		List<Map<String,int[]>> ranks = new ArrayList<Map<String,int[]>>();
		ranks.add(rankTrain);
		ranks.add(rankTest);
		return ranks;
	}

	/**
	 * highest go first
	 * @param values
	 * @param methods
	 */
	public static void sort(double[] values, String[] methods){
		for(int i=0;i<values.length-1;i++){
			for(int j=i+1;j<values.length;j++){
				if(values[i] < values[j]){
					double tmpValues = values[i];
					String tmpName = methods[i];
					values[i] = values[j];
					methods[i] = methods[j];
					values[j] = tmpValues;
					methods[j] = tmpName;
				}
			}
		}
	}

}
