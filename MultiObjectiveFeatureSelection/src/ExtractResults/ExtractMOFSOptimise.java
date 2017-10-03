package ExtractResults;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Map.Entry;

import org.omg.CORBA.TRANSACTION_UNAVAILABLE;
import org.uma.jmetal.qualityindicator.impl.InvertedGenerationalDistance;
import org.uma.jmetal.qualityindicator.impl.hypervolume.PISAHypervolume;
import org.uma.jmetal.solution.DoubleSolution;
import org.uma.jmetal.util.front.Front;
import org.uma.jmetal.util.front.imp.ArrayFront;
import org.uma.jmetal.util.front.util.FrontNormalizer;
import org.uma.jmetal.util.point.Point;
import org.uma.jmetal.util.point.impl.ArrayPoint;

public class ExtractMOFSOptimise {

	public static void main(String[] args) throws FileNotFoundException{
		String dir = "Results/OptimizeNn";
		Map<String,Integer> fullSizes = new HashMap<String,Integer>();
		List<String> datasets = new ArrayList<String>();
				datasets.add("wine"); fullSizes.put("wine", 13 );
				datasets.add("australian");fullSizes.put("australian", 14);
		//		datasets.add("zoo"); fullSizes.put("zoo", 17);
				datasets.add("vehicle"); fullSizes.put("vehicle", 18);
				datasets.add("german");fullSizes.put("german", 24);
				datasets.add("wbcd");fullSizes.put("wbcd", 30);
				datasets.add("ionosphere");fullSizes.put("ionosphere", 34);
		//		datasets.add("lung");fullSizes.put("lung", 56);
				datasets.add("sonar");fullSizes.put("sonar", 60);
				datasets.add("hillvalley");fullSizes.put("hillvalley", 101 );
				datasets.add("musk1");fullSizes.put("musk1", 166);
				datasets.add("arrhythmia");fullSizes.put("arrhythmia", 279);
				datasets.add("madelon");fullSizes.put("madelon", 500);
				datasets.add("isolet5"); fullSizes.put("isolet5", 617);
		//		datasets.add("multiplefeatures");fullSizes.put("multiplefeatures", 649);


		List<String> methods = new ArrayList<String>();
//		methods.add("NSGAII"); 
//		methods.add("NSGAIII");
//		methods.add("SPEA2");
//		methods.add("OMOPSO");
//		methods.add("MOEAD");
//		methods.add("MOEADREF");
//		methods.add("MOEADDYNRan");
		methods.add("3");methods.add("4");methods.add("5");methods.add("6");methods.add("7");
		methods.add("8");
//		methods.add("MOEADDYNSeqAcc");
//		methods.add("MOEADDYNSeqRan");
//		methods.add("MOEADDYNDE");
//		for(int rate = 50; rate <= 90; rate = rate+10){
//			for(int noI= 3; noI<=9; noI = noI+2){
//				methods.add(rate+"-"+noI);
//			}
//		}

		String saveTo = "Processed/OptimizedNn/Data";
		int noRun = 10;
		for(String dataset: datasets){

			//extract optimal pf
			List<List<ObjectiveRecords>> optimal = extractOptimalFront(dir, dataset, methods, noRun);
			List<ObjectiveRecords> trainpf = optimal.get(0);//= new ArrayList();trainpf.add(new ObjectiveRecords(0, 0));

			List<ObjectiveRecords> testpf = optimal.get(1);//new ArrayList();testpf.add(new ObjectiveRecords(0, 0));
			
			List<ObjectiveRecords> trainall = optimal.get(2);
			List<ObjectiveRecords> testall = optimal.get(3);
			//
			Front frontTrain = list2Front(trainpf);
			FrontNormalizer fnTrain = new FrontNormalizer(list2Front(trainall));
			//frontTrain = fnTrain.normalize(frontTrain);

			Front frontTest = list2Front(testpf);
			FrontNormalizer fnTest = new FrontNormalizer(list2Front(testall));
			//frontTest = fnTest.normalize(frontTest);


			PISAHypervolume<DoubleSolution> hyperTrain = new PISAHypervolume<DoubleSolution>();
			hyperTrain.setReferenceParetoFront(frontTrain);

			InvertedGenerationalDistance<DoubleSolution> igdTrain = new InvertedGenerationalDistance<>();
			igdTrain.setReferenceParetoFront(frontTrain);

			PISAHypervolume<DoubleSolution> hyperTest = new PISAHypervolume<DoubleSolution>();
			hyperTest.setReferenceParetoFront(frontTest);

			InvertedGenerationalDistance<DoubleSolution> igdTest = new InvertedGenerationalDistance<>();
			igdTest.setReferenceParetoFront(frontTest);

			for(String method: methods){
				extract(dir, dataset, method, saveTo, noRun, hyperTrain, hyperTest, igdTrain, igdTest, fnTrain,fnTest);
			}
		}
	}

	public static List<List<ObjectiveRecords>> extractOptimalFront(String dir,
			String dataset, List<String> methods, int noRun) throws FileNotFoundException{
		List<ObjectiveRecords> bestPFTrain = new ArrayList<ObjectiveRecords>();
		List<ObjectiveRecords> bestPFTest = new ArrayList<ObjectiveRecords>();

		for(String method: methods)
		{
			String location = dir+"/"+dataset+"/"+method;

			for(int i=1;i<=noRun;i++){

				Scanner scp = new Scanner(new File(location+"/FUN"+i));
				//skip previous iteration
				while(scp.hasNextLine()){
					String line = scp.nextLine();
					if(line.contains("Final"))
						break;
				}
				while(scp.hasNextLine()){
					String line = scp.nextLine();
					String[] splits = line.split(",");
					double featureRate = Double.parseDouble(splits[0].trim());
					double trainError = Double.parseDouble(splits[1].trim());
					double testError = Double.parseDouble(splits[2].trim());
					if(featureRate>0){
						ObjectiveRecords train = new ObjectiveRecords(featureRate, trainError);
						ObjectiveRecords test = new ObjectiveRecords(featureRate, testError);
						if(!bestPFTrain.contains(train))
							bestPFTrain.add(train);
						if(!bestPFTest.contains(test))
							bestPFTest.add(test);
					}
				}
				scp.close();
			}
		}
		List<ObjectiveRecords> allPFTrain = new ArrayList<ObjectiveRecords>(bestPFTrain);
		List<ObjectiveRecords> allPFTest = new ArrayList<ObjectiveRecords>(bestPFTest);
		bestPFTrain = getNondominated(bestPFTrain);
		bestPFTest = getNondominated(bestPFTest);

		List<List<ObjectiveRecords>> lists = new ArrayList<List<ObjectiveRecords>>();
		lists.add(bestPFTrain); lists.add(bestPFTest);
		lists.add(allPFTrain); lists.add(allPFTest);
		return lists;
	}

	public static Front list2Front(List<ObjectiveRecords> list){
		Front front = new ArrayFront(list.size(),2);
		for(int i=0;i<list.size();i++){
			ObjectiveRecords or = list.get(i);
			Point point = new ArrayPoint(2);
			point.setDimensionValue(0, or.getFeatureRate());
			point.setDimensionValue(1, or.getErrorRate());
			front.setPoint(i, point);
		}
		return front;
	}

	public static PISAHypervolume<DoubleSolution> list2hyper(List<ObjectiveRecords> list) throws FileNotFoundException{
		PISAHypervolume<DoubleSolution> hyper = new PISAHypervolume<DoubleSolution>();
		Front front = list2Front(list);
		hyper.setReferenceParetoFront(front);
		return hyper;
	}

	public static void extract(String dir, String dataset,
			String method, String saveTo,
			int noRun,
			PISAHypervolume<DoubleSolution> hyperTrain,
			PISAHypervolume<DoubleSolution> hyperTest,
			InvertedGenerationalDistance<DoubleSolution> igdTrain,
			InvertedGenerationalDistance<DoubleSolution> igdTest,
			FrontNormalizer fnTrain,
			FrontNormalizer fnTest) throws FileNotFoundException{
		String location = dir+"/"+dataset+"/"+method;
		saveTo = saveTo+"/"+dataset;
		new File(saveTo).mkdirs();
		List<ObjectiveRecords> allTrainSolutions = new ArrayList<ObjectiveRecords>();
		List<ObjectiveRecords> allTestSolutions = new ArrayList<ObjectiveRecords>();
		double time = 0.0;
		int count = 0;

		String saveToVolume = saveTo+"/Volumes";
		new File(saveToVolume).mkdirs();
		PrintStream ptVolume = new PrintStream(new File(saveToVolume+"/"+method+".txt"));

		String saveToIgd = saveTo+"/IGD";
		new File(saveToIgd).mkdirs();
		PrintStream ptIgd = new PrintStream(new File(saveToIgd+"/"+method+".txt"));

		ptVolume.println("Training,Testing");
		ptIgd.println("Training,Testing");
		for(int i=1;i<=noRun;i++){
			Scanner sct = new Scanner(new File(location+"/Time"+i));
			time += Double.parseDouble(sct.nextLine().trim());
			sct.close();

			//to calculate volume each run
			List<ObjectiveRecords> trainRun = new ArrayList<ObjectiveRecords>();
			List<ObjectiveRecords> testRun = new ArrayList<ObjectiveRecords>();

			Scanner scp = new Scanner(new File(location+"/FUN"+i));

			//skip previous iteration
			while(scp.hasNextLine()){
				String line = scp.nextLine();
				if(line.contains("Final"))
					break;
			}

			while(scp.hasNextLine()){
				String line = scp.nextLine();
				String[] splits = line.split(",");
				double featureRate = Double.parseDouble(splits[0].trim());
				double trainError = Double.parseDouble(splits[1].trim());
				double testError = Double.parseDouble(splits[2].trim());
				if(featureRate>0){
					ObjectiveRecords train = new ObjectiveRecords(featureRate, trainError);
					ObjectiveRecords test = new ObjectiveRecords(featureRate, testError);
					if(!allTrainSolutions.contains(train))
						allTrainSolutions.add(train);
					if(!allTestSolutions.contains(test))
						allTestSolutions.add(test);
					count++;

					if(!trainRun.contains(train))
						trainRun.add(train);
					if(!testRun.contains(test))
						testRun.add(test);
				}
			}
			scp.close();

			trainRun = getNondominated(trainRun);
			double trainVolume = list2Volume(trainRun, hyperTrain, fnTrain);
			double trainIgd = list2IGD(trainRun, igdTrain, fnTrain);
			testRun = getNondominated(testRun);
			double testVolume = list2Volume(testRun, hyperTest, fnTest);
			double testIgd = list2IGD(testRun, igdTest, fnTest);

			ptVolume.println(trainVolume+","+testVolume);
			ptIgd.println(trainIgd+","+testIgd);
		}
		ptVolume.close();
		ptIgd.close();

		time = time/noRun;

		String saveToTime = saveTo+"/Time";
		new File(saveToTime).mkdirs();
		PrintStream pt = new PrintStream(new File(saveToTime+"/"+method+".txt"));
		pt.println(time);
		pt.close();

		//List<ObjectiveRecords> bestTrain = getNondominated(allTrainSolutions);
		//List<ObjectiveRecords> bestTest = getNondominated(allTestSolutions);
		//List<ObjectiveRecords> aveTrain = getAverage(allTrainSolutions);
		//List<ObjectiveRecords> aveTest = getAverage(allTestSolutions);

		//printResult(saveTo+"/TrainB",method+".txt",bestTrain);
		//printResult(saveTo+"/TrainA",method+".txt",aveTrain);
		//printResult(saveTo+"/TestB",method+".txt",bestTest);
		//printResult(saveTo+"/TestA",method+".txt",aveTest);
	}

	public static double list2Volume(List<ObjectiveRecords> list,
			PISAHypervolume<DoubleSolution> hyper, FrontNormalizer fn){
		Front front = list2Front(list);
		//front = fn.normalize(front);
		return hyper.hypervolume(front, hyper.getReferenceFront());
	}

	public static double list2IGD(List<ObjectiveRecords> list,
			InvertedGenerationalDistance<DoubleSolution> igd, FrontNormalizer fn){
		Front front = list2Front(list);
		//front = fn.normalize(front);
		return igd.invertedGenerationalDistance(front, igd.getReferenceFront());
	}

	public static void printResult(String saveTo, String fileName, List<ObjectiveRecords> list) throws FileNotFoundException{
		new File(saveTo).mkdirs();
		PrintStream pt = new PrintStream(new File(saveTo+"/"+fileName));
		pt.println("fRate,eRate");
		for(ObjectiveRecords rc: list){
			pt.println(rc);
		}
	}

	public static List<ObjectiveRecords> getNondominated(List<ObjectiveRecords> origin){
		List<ObjectiveRecords> toReturn = new ArrayList<ObjectiveRecords>();
		for(int i=0;i<origin.size();i++){
			boolean beDominated = false;
			for(int j=0;j<origin.size();j++){
				if(j!=i){
					if(origin.get(i).compareTo(origin.get(j)) < 0){
						//j dominates i
						beDominated = true;
						break;
					}
				}
			}
			if(!beDominated)
				toReturn.add(origin.get(i));
		}
		return toReturn;
	}

	public static List<ObjectiveRecords> getAverage(List<ObjectiveRecords> origin){
		Map<Double,Double> f2e = new HashMap<Double,Double>();
		Map<Double,Integer> f2c = new HashMap<Double,Integer>();

		for(ObjectiveRecords or: origin){
			double frate = or.getFeatureRate();
			double erate = or.getErrorRate();
			if(f2e.containsKey(frate)){
				f2e.put(frate,f2e.get(frate)+erate);
				f2c.put(frate, f2c.get(frate)+1);
			}
			else{
				f2e.put(frate, erate);
				f2c.put(frate, 1);
			}
		}

		List<ObjectiveRecords> toReturn = new ArrayList<ObjectiveRecords>();
		for(Entry<Double,Double> entry: f2e.entrySet()){
			double frate = entry.getKey();
			double erate = entry.getValue();
			double crate = f2c.get(frate);
			erate = erate/crate;
			ObjectiveRecords or = new ObjectiveRecords(frate, erate);
			toReturn.add(or);
		}
		return toReturn;
	}

}
