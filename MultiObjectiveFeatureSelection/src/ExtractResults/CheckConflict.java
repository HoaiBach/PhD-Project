package ExtractResults;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Scanner;
import java.util.TreeMap;

public class CheckConflict {

	public static void main(String[] args) throws FileNotFoundException{
		int runs = 50;

		List<String> datasets = new ArrayList<String>();
		datasets.add("arrhythmia"); datasets.add("australian");
		datasets.add("hillvalley"); datasets.add("madelon");
		datasets.add("multiplefeatures");datasets.add("musk1");
		datasets.add("sonar");datasets.add("vehicle");
		datasets.add("wbcd");

		List<String> methods = new ArrayList<String>();
		methods.add("Exact");
		methods.add("Rectangle");

		for(String dataset: datasets){
			File dataout = new File("Processed/CheckConflict/Data/"+dataset+"/");
			dataout.mkdirs();
			for(String method: methods){
				File output = new File("Processed/CheckConflict/Data/"+dataset+"/"+method+".txt");
				PrintStream pt = new PrintStream(output);

				//contains the best accuracy for each number of features
				Map<Double,Double> nfAcc = new TreeMap<Double,Double>();

				//now go through each file
				for(int run=1;run<=runs;run++){
					Scanner sc = new Scanner(new File("Results/"+dataset+"/"+method+"/FUN"+run+".txt"));

					while(sc.hasNextLine()){
						String line = sc.nextLine();
						String[] splits = line.split(", ");
						double nf = Double.parseDouble(splits[0].trim());
						double err = Double.parseDouble(splits[1].trim());

						//for all
						if(nfAcc.containsKey(nf)){
							if(err < nfAcc.get(nf))
								nfAcc.put(nf, err);
						}
						else{
							if(nf>0){
								nfAcc.put(nf, err);
							}
						}
						
						//for nondominated
//						List<Double> toRemove = new ArrayList<Double>();
//						boolean beDominated = false;
//						for(Entry<Double,Double> entry: nfAcc.entrySet()){
//							//check whether the new one dominate any other
//							//if the new solution dominate some solutions
//							//the the dominated solutions are removed
//							if((entry.getKey() > nf && entry.getValue() >= err)
//									|| (entry.getKey() >= nf && entry.getValue() > err)){
//								toRemove.add(entry.getKey());
//							}
//							//if the new solution is dominated by anyother
//							//it will not be added
//							else if((entry.getKey() <= nf && entry.getValue() < err)
//									|| (entry.getKey() < nf && entry.getValue() <= err)){
//								beDominated = true;
//							}
//						}
//						
//						for(Double key: toRemove){
//							nfAcc.remove(key);
//						}
//						
//						if(!beDominated){
//							if(!nfAcc.containsKey(nf) && nf>0){
//								nfAcc.put(nf, err);
//							}
//						}
					}

					sc.close();
				}

				//now output the nfAcc map
				for(Entry<Double,Double> entry: nfAcc.entrySet()){
					pt.println(entry.getKey()+","+entry.getValue());
				}
			}
		}
	}
}
