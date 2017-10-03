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

import Utility.NewMath;

public class ResultTable {


	static String[] datasets = new String[]{
			"wine", 
			"australian",
			"vehicle", 
			"german", 
			"wbcd", 
			"ionosphere" , 
			"sonar", 
			"hillvalley",
			"musk1", 
			"arrhythmia",
			"madelon",
			"isolet5"
			};
	static String[] inds = new String[]{"IGD","Volumes"};
	static List<String> methods = new ArrayList<String>();
	static String methodMaster;
	static int runs = 10;

	public static void main(String[] args) throws FileNotFoundException{
//
//		methods.add("NSGAII"); 
//		methods.add("NSGAIII");
//		methods.add("SPEA2"); 
//		methods.add("OMOPSO");methodMaster = "OMOPSO";
//		methods.add("MOEAD"); 
//		methods.add("MOEADDYNRan");
//		methods.add("MOEADDYNSeqAcc");
//		methods.add("MOEADDYNDE");
//		methods.add("MOEADREF");methodMaster = "MOEADREF";
		methods.add("3");methods.add("4");methods.add("5");methods.add("6");methods.add("7");
		methods.add("8");methodMaster="3";
//		methods.add("MOEADDYNSeqAcc");methodMaster = "MOEADDYNSeqAcc";
//		methods.add("MOEADDYNRan");methodMaster = "MOEADDYNRan";
//		methods.add("MOEADDYNSeqRan");methodMaster="MOEADDYNSeqRan";

		//read data
		Map<String,Map<String,Map<String,double[][]>>> results= new HashMap<String,Map<String,Map<String,double[][]>>>();
		for(String dataset: datasets){
			String dirDataset = "Processed/OptimizedNn/Data/"+dataset;
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
		String format = "{l";
		for(int i=0;i<methods.size();i++)
			format+="l";
		format += "}";
		String first = "Dataset";
		for(String method: methods)
			first += " & "+method;
		String header = "\\begin{table}\n"
				+ "\\centering\n"
				+ "\\scriptsize\n"				
				+ "\\begin{tabular}"+format+"\n"
				+ "\\\\\\hline\n"
				+ first+"\\\\\\hline\n";
		String end = "\\end{tabular}\n";

		File out = new File("Processed/OptimizedNn/Table/table.tex");
		PrintStream pt = new PrintStream(out);
		pt.println("\\documentclass{report}\n"
				+ "\\usepackage[cm]{fullpage}\n"
				+ "\\usepackage{rotating}\n"
				+ "\\begin{document}\n");

		String igdtr = header;
		String igdte = header;
		String voltr = header;
		String volte = header;

		DecimalFormat f = new DecimalFormat("#0.000");

		for(String dataset: datasets){
			igdtr += dataset;
			igdte += dataset;
			voltr += dataset;
			volte += dataset;

			// indicator -> (method, train/test)
			Map<String,Map<String,double[][]>> datasetResult = results.get(dataset);

			int index = 1;
			for(String ind: inds){

				Map<String,double[][]> indResult = datasetResult.get(ind);
				for(int i=0;i<=1;i++){

					String toAdd = "";

					double[] masterResult = indResult.get(methodMaster)[i];
					for(String method: methods){
						double[] result = indResult.get(method)[i];
						String ave = f.format(NewMath.mean(result));
						String std = f.format(NewMath.calculateSTD(result, NewMath.mean(result)));

						if(!method.equals(methodMaster)){
							//index == 1-> igd: smaller better
							//index == 2-> vol: higher better
							String sig = index <=1 ? WilsonTestBing.TestBingNew(result, masterResult):
								WilsonTestBing.TestBingNew(masterResult, result);
							toAdd += " & "+ave +"$\\pm$"+std+
									"("+sig+")";
						}
						else{
							toAdd += " & "+ave +"$\\pm$"+std;
						}	
					}

					//now add to string
					if(index == 1 && i ==0)
						igdtr += toAdd;
					else if(index == 1 && i == 1)
						igdte += toAdd;
					else if(index ==2 && i ==0)
						voltr += toAdd;
					else
						volte += toAdd;
				}
				index++;
			}

			igdtr += "\\\\\\hline\n";
			igdte += "\\\\\\hline\n";
			voltr += "\\\\\\hline\n";
			volte += "\\\\\\hline\n";
		}

		//end table
		igdtr += end
				+"\\caption{IGD training}\n"
				+ "\\end{table}\n";
		igdte += end
				+"\\caption{IGD testing}\n"
				+ "\\end{table}\n";
		voltr += end
				+"\\caption{Volumes training}\n"
				+ "\\end{table}\n";
		volte += end
				+"\\caption{Volumes testing}\n"
				+ "\\end{table}\n";

		//now print to the table
		pt.println(igdtr);
		pt.println(igdte);
		pt.println(voltr);
		pt.println(volte);

		pt.println("\\end{document}\n ");
		pt.close();
	}


}
