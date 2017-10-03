package ExtractResults;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import Utility.NewMath;

public class HyperVolumeTable {

	public static void main(String[] args) throws FileNotFoundException{
		int noRun = 50;
		NumberFormat formatter = new DecimalFormat("0.000");
		NumberFormat timeFormatter = new DecimalFormat("#.0");

		List<String> datasets = new ArrayList<String>();
		datasets.add("wine");
		datasets.add("australian");
//		datasets.add("zoo");
		datasets.add("vehicle");
		datasets.add("german");
		datasets.add("wbcd");
		datasets.add("ionosphere");
//		datasets.add("lung");
		datasets.add("sonar");
//		datasets.add("hillvalley");
//		datasets.add("musk1");
//		datasets.add("arrhythmia");
//		datasets.add("madelon");
//		datasets.add("isolet5");
//		datasets.add("multiplefeatures");
		List<String> methods = new ArrayList<String>();
		//methods.add("GA");
		//methods.add("SPEA2");
		//methods.add("MOEAD-lowRef");
		//methods.add("MOEAD-multiRefs");
		methods.add("MOEAD-MFs");
		//methods.add("MOPSO");
		methods.add("MOEAD");

		String dir = "ProcessedResult/Data/";

		File output = new File("ProcessedResult/Latex/volumes.txt");
		PrintStream pt = new PrintStream(output);

		File outputTime = new File("ProcessedResult/Latex/times.txt");
		PrintStream ptTime = new PrintStream(outputTime);

		String s ="";
		s+= "\\begin{table}\n";
		s+= "\\centering\n";
		s+= "\\caption{"+"Hyper volume"+"}\n";

		String stime = "";
		stime+= "\\begin{table}\n";
		stime+= "\\centering\n";
		stime+= "\\caption{"+"Computation time in seconds"+"}\n";

		s+= "\\begin{tabular}{|l|c|";//c|c|c|c|c|c|}\n";
		stime+= "\\begin{tabular}{|l|";
		int nMethods = methods.size();
		for(String method: methods){
			System.out.println(method);
			s+= "c|";
			stime+="l|";
		}
		s+= "}\n";stime+="}\n";
		s+= "\\hline\n";stime+= "\\hline\n";
		s+= "Dataset &";stime+= "Dataset";
		String master = "MOEAD-MFs";
		s+= "& MFs";
		for(String method: methods){
			if(!method.equals(master))
				s+= " &"+method;
			stime+=" &"+method;
		}
		s+= "\\\\\n";stime+="\\\\\n";
		s+= "\\hline\n";stime+="\\hline\n";
		for(String dataset: datasets){

			stime+= dataset;

			Map<String,double[]> train = new HashMap<String,double[]>();
			Map<String,double[]> test  = new HashMap<String,double[]>();

			for(String method: methods){
				File volumeFile = new File(dir+"/"+dataset+"/Volumes/"+method+".txt");
				Scanner sc = new Scanner(volumeFile);
				double[] trainVolumes = new double[noRun];
				double[] testVolumes = new double[noRun];
				sc.nextLine();
				int index = 0;
				while(sc.hasNextLine()){
					String line = sc.nextLine();
					String[] split = line.split(",");
					trainVolumes[index] = Double.parseDouble(split[0]);
					testVolumes[index] = Double.parseDouble(split[1]);
					index++;
				}
				train.put(method, trainVolumes);
				test.put(method, testVolumes);
				sc.close();

				sc = new Scanner(new File(dir+"/"+dataset+"/Time/"+method+".txt"));
				double time = Double.parseDouble(sc.nextLine().trim())/(1000*60);
				stime += "&"+timeFormatter.format(time);
			}
			stime+="\\\\\n";
			stime+="\\hline\n";

			//now build the latex file
			s+= "\\multirow{2}{*}{"+dataset+"}";
			s += "& Training ";
			double[] masterTrain = train.get(master);
			s += "&" +formatter.format(NewMath.mean(masterTrain));
			for(String method: methods){
				if(!method.equals(master)){
					double[] rowTrain = train.get(method);
					s += "&" +formatter.format(NewMath.mean(rowTrain))+"("+
					WilsonTestBing.TestBingNew(rowTrain, masterTrain)+")";
				}
			}
			s+= "\\\\\n";
			s+= "\\cline{2-"+(nMethods+2)+"}\n";

			s += "&Testing ";
			double[] masterTest = test.get(master);
			s += "&" +formatter.format(NewMath.mean(masterTest));
			for(String method: methods){
				if(!method.equals(master)){
				double[] rowTest = test.get(method);
				s += "&" +formatter.format(NewMath.mean(rowTest))+"("+
						WilsonTestBing.TestBingNew(rowTest, masterTest)+")";
				}
			}
			s+= "\\\\\n";
			s+= "\\hline\n";
		}

		s+= "\\end{tabular}\n";
		s+= "\\end{table}\n\n";

		stime+= "\\end{tabular}\n";
		stime+= "\\end{table}\n\n";

		pt.println(s);
		pt.close();

		ptTime.println(stime);
		ptTime.close();
	}
}
