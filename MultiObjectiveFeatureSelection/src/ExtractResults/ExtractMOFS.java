package ExtractResults;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Scanner;

import org.uma.jmetal.qualityindicator.impl.hypervolume.PISAHypervolume;
import org.uma.jmetal.solution.DoubleSolution;
import org.uma.jmetal.util.front.Front;
import org.uma.jmetal.util.front.imp.ArrayFront;
import org.uma.jmetal.util.front.util.FrontUtils;
import org.uma.jmetal.util.point.Point;
import org.uma.jmetal.util.point.impl.ArrayPoint;

public class ExtractMOFS {

	public static void main(String[] args) throws FileNotFoundException{
		String dir = "Result";
		Map<String,Integer> fullSizes = new HashMap<String,Integer>();
		List<String> datasets = new ArrayList<String>();
		datasets.add("wine"); fullSizes.put("wine", 13 );
		datasets.add("australian");fullSizes.put("australian", 14);
//		datasets.add("zoo"); fullSizes.put("zoo", 17);
		datasets.add("vehicle"); fullSizes.put("vehicle", 18);
		datasets.add("german");fullSizes.put("german", 24);
		datasets.add("wbcd");fullSizes.put("wbcd", 30);
//		datasets.add("ionosphere");fullSizes.put("ionosphere", 34);
//		datasets.add("lung");fullSizes.put("lung", 56);
		datasets.add("sonar");fullSizes.put("sonar", 60);
		datasets.add("hillvalley");fullSizes.put("hillvalley", 101 );
		datasets.add("musk1");fullSizes.put("musk1", 166);
//		datasets.add("arrhythmia");fullSizes.put("arrhythmia", 279);
//		datasets.add("madelon");fullSizes.put("madelon", 500);
//		datasets.add("isolet5"); fullSizes.put("isolet5", 617);
//		datasets.add("multiplefeatures");fullSizes.put("multiplefeatures", 649);


		List<String> methods = new ArrayList<String>();
		List<String> tempMethods = new ArrayList<String>();
		//methods.add("GA");
		tempMethods.add("GA");
		//methods.add("SPEA2");
		tempMethods.add("SPEA2");
		methods.add("MOEAD-MFs");
		tempMethods.add("MOEAD-MFs");
		//methods.add("MOPSO");
		tempMethods.add("MOPSO");
		methods.add("MOEAD");
		tempMethods.add("MOEAD");
		String saveTo = "ProcessedResult/Data";
		int noRun = 50;
		for(String dataset: datasets){

			//extract optimal pf
			List<List<ObjectiveRecords>> optimal = extractOptimalFront(dir, dataset, tempMethods, noRun);
			List<ObjectiveRecords> trainpf = optimal.get(0);//= new ArrayList();trainpf.add(new ObjectiveRecords(0, 0));

			List<ObjectiveRecords> testpf = optimal.get(1);//new ArrayList();testpf.add(new ObjectiveRecords(0, 0));
			//
			PISAHypervolume<DoubleSolution> hyperTrain = list2hyper(trainpf);
			PISAHypervolume<DoubleSolution> hyperTest = list2hyper(testpf);

			for(String method: methods){
				extract(dir, dataset, method, saveTo, noRun, hyperTrain, hyperTest);
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
		bestPFTrain = getNondominated(bestPFTrain);
		bestPFTest = getNondominated(bestPFTest);

		List<List<ObjectiveRecords>> lists = new ArrayList<List<ObjectiveRecords>>();
		lists.add(bestPFTrain); lists.add(bestPFTest);
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
			PISAHypervolume<DoubleSolution> hyperTest) throws FileNotFoundException{
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
		ptVolume.println("Training,Testing");
		for(int i=1;i<=noRun;i++){
			Scanner sct = new Scanner(new File(location+"/Time"+i));
			time += Double.parseDouble(sct.nextLine().trim());
			sct.close();

			//to calculate volume each run
			List<ObjectiveRecords> trainRun = new ArrayList<ObjectiveRecords>();
			List<ObjectiveRecords> testRun = new ArrayList<ObjectiveRecords>();

			Scanner scp = new Scanner(new File(location+"/FUN"+i));
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
			double trainVolume = list2Volume(trainRun, hyperTrain);
			testRun = getNondominated(testRun);
			double testVolume = list2Volume(testRun, hyperTest);

			ptVolume.println(trainVolume+","+testVolume);

		}
		ptVolume.close();

		time = time/noRun;
		System.out.println("Total solutions of "+method+" on "+dataset+" is: "+count);
		System.out.println("Size training: "+allTrainSolutions.size());
		System.out.println("Size testing: "+allTestSolutions.size());

		String saveToTime = saveTo+"/Time";
		new File(saveToTime).mkdirs();
		PrintStream pt = new PrintStream(new File(saveToTime+"/"+method+".txt"));
		pt.println(time);
		pt.close();

		List<ObjectiveRecords> bestTrain = getNondominated(allTrainSolutions);
		List<ObjectiveRecords> bestTest = getNondominated(allTestSolutions);
		List<ObjectiveRecords> aveTrain = getAverage(allTrainSolutions);
		List<ObjectiveRecords> aveTest = getAverage(allTestSolutions);

		printResult(saveTo+"/TrainB",method+".txt",bestTrain);
		printResult(saveTo+"/TrainA",method+".txt",aveTrain);
		printResult(saveTo+"/TestB",method+".txt",bestTest);
		printResult(saveTo+"/TestA",method+".txt",aveTest);
	}

	public static double list2Volume(List<ObjectiveRecords> list,
			PISAHypervolume<DoubleSolution> hyper){
		Front front = list2Front(list);
		Front invertedFront = FrontUtils.getInvertedFront(front);
		int noObjectives = 2;
		return hyper.calculateHypervolume(
				FrontUtils.convertFrontToArray(invertedFront),
				invertedFront.getNumberOfPoints(),
				noObjectives);
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
