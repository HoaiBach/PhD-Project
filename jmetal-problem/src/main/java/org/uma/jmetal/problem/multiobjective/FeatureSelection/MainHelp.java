package org.uma.jmetal.problem.multiobjective.FeatureSelection;
/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */


import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.tools.data.FileHandler;

/**
 *
 * @author xuebing
 */
public class MainHelp {
	//    This is a continuous PSO for feature selection
	static List<int[]> clusters;
	static int noFeatures;

	public static int noFeature() throws IOException {
		noFeatures = Integer.parseInt(ReadResults.read1Line("noFeatures.txt"));
		System.out.println("noFeatures (read from file)  " + noFeatures);
		return noFeatures;
	}

	public static List<int[]> getCluster(String filename) throws IOException{
		clusters = new ArrayList<int[]>();
		int numClusters = Integer.parseInt((ReadResults.readOneCertainLine(filename, 2).split(" "))[3]);
		System.out.println(numClusters);
		String[] input = ReadResults.readOneCertainLine(filename, 3).split("]");
		int i=0;
		int count=0;
		while(count!=numClusters){
			//List<Integer> list = new ArrayList<Integer>();
			String[] tmp = input[i].split("\\[");
			if(!tmp[tmp.length-1].trim().equals(","))
			{
				System.out.println("String "+tmp[tmp.length-1]);
				String[] s = tmp[tmp.length-1].split(",");
				int[] list = new int[s.length];
				for(int j=0;j<s.length;j++){
					//list.add(Integer.parseInt(s[j].trim()));
					list[j] = Integer.parseInt(s[j].trim());
				}
				clusters.add(list);
				count++;
			}
			i++;
		}
		System.out.println(clusters.size());
		//		for(int i=0;i<numClusters;i++){
		//			List<Integer> test = clusters.get(i);
		//			for(int j=0;j< test.size();j++){
		//				System.out.print(test.get(j)+" ");
		//			}
		//			System.out.println();
		//		}
		return clusters;
	}

	public static Dataset[] readBingData(int noFeatures) throws IOException {
		Dataset data = FileHandler.loadDataset(new File("Data.data"), noFeatures, ",");
		System.out.println("Number of features (dimension):  " + data.noAttributes());
		System.out.println("Dataset[] folds = data.folds((divfolds), new Random(0))");
		System.out.println("Dataset[] foldsTrain = training.folds(numFolds, new Random(1))");
		System.out.println("Dataset[] foldsTest = testing.folds(numFolds, new Random(500))");
		System.out.println("");

		//        Dataset[] foldsTest = testing.folds(divfolds, new Random(500));
		//        Dataset[] foldsTrain = training.folds(numFolds, RandomBing.Create());
		//        Dataset[] folds = data.folds((divfolds), RandomBing.Create());
		Dataset[] folds = data.folds((10), new Random(100));
		Dataset training = new DefaultDataset();
		Dataset testing = new DefaultDataset();
		int[] tr = {0, 2, 3, 5, 6, 8, 9};
		int[] te = {1, 4, 7};   // 7, 4 and 6,5 changes
		for (int i = 0; i < 7; i++) {
			training.addAll(folds[tr[i]]);
		}
		for (int i = 0; i < 3; i++) {
			testing.addAll(folds[te[i]]);
		}

		Dataset[] trainTest = {training, testing};
		return trainTest;
	}

	public static Dataset[] readData4OtherLA(String path, int noFeatures) throws IOException {
		Dataset data = FileHandler.loadDataset(new File(path + "Data.data"), noFeatures, ",");
		Dataset[] folds = data.folds((10), new Random(100));
		Dataset training = new DefaultDataset();
		Dataset testing = new DefaultDataset();
		int[] tr = {0, 2, 3, 5, 6, 8, 9};
		int[] te = {1, 4, 7};   // 7, 4 and 6,5 changes
		for (int i = 0; i < 7; i++) {
			training.addAll(folds[tr[i]]);
		}
		for (int i = 0; i < 3; i++) {
			testing.addAll(folds[te[i]]);
		}

		Dataset[] trainTest = {training, testing};
		return trainTest;
	}

	public static String[] files(int number_of_runs, int noTotalRuns, int rSeeder) throws IOException {
		String saveFiles = "ParaSettings/SaveFiles.txt";
		ArrayList<String> filesNameList = ReadResults.readAll2list(saveFiles);
		String[] files = new String[filesNameList.size()];
		for (int i = 0; i < filesNameList.size(); i++) {
			files[i] = filesNameList.get(i);
		}

		//** Followings is to change the name of the files based on the number of run
		//

		int noJobs = noTotalRuns / number_of_runs;
		if (number_of_runs == noTotalRuns) {
			if (noJobs != 1) {
				System.out.println("Wrongf value of -noJobs- in the file of (Parameters.txt) !!!!!");
			}
		} else if (number_of_runs < noTotalRuns) {
			for (int i = 0; i < files.length; i++) {
				files[i] = Integer.toString(rSeeder + number_of_runs) + files[i];
			}
		} else {
			System.out.println("Wrongf value of number_of_runs and noTotalRuns in the file of (Parameters.txt),  (number_of_runs > noTotalRuns) !!!!!");
		}

		System.out.println(files[0]);

		return files;
	}

	public static int[] arRSeeder(int noTotalRuns, int number_of_runs) {
		int noJobs = noTotalRuns / number_of_runs;
		int[] arRSeeder = new int[noJobs];

		for (int i = 0; i < noJobs; i++) {
			arRSeeder[i] = i * number_of_runs;
		}
		return arRSeeder;

	}

	public static long[] rSeeders(int number_of_runs, int rSeeder) {
		System.out.println("The " + (rSeeder + 1) + "  Jobs");
		System.out.println("rSeeder:  " + rSeeder);
		long[] Seeder = new long[number_of_runs];
		for (int r = 0; r < number_of_runs; r++) {
			Seeder[r] = 2 * rSeeder;
			//            Seeder[r] = rSeeder * rSeeder * rSeeder * 135 + rSeeder * rSeeder * 246 + 98;
			rSeeder++;
		}


		//        System.out.println("RandomBing seeders setting:   " + " Seeder[r] = r * r * r * 135 + r * r * 246+98" + "      int[] test = {1, 4, 7}; 7-6,0-1");
		System.out.println("2* rSeeder");
		return Seeder;
	}



	public static int sizeSubset(double[] features) {
		int count =0;
		for(int i=0;i<features.length;i++){
			if(features[i] == 1.0)
				count++;
		}
		return count;
	}

	public static int roundUp(double d){
		int r = (int) d;
		if(r==d)
			return r;
		else return r+1;
	}


	/**
	 *
	 * @param p1
	 * @param p2
	 * @return: the distane between two positions p1 and p2,
	 * if they are continuous: return Euclidean distance
	 * if they are binary: return Manhattan distance
	 */
	public static double distanceBetweenContinuousPositions(List<Double> p1, List<Double> p2){
		double distance = 0;

		if(p1.size() != p2.size()){
			try {
				throw new Exception("Length of two positions are not equal!!!!");
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		for(int i=0;i<p1.size();i++){
			distance += (p1.get(i)- p2.get(i))*(p1.get(i)-p2.get(i));
		}

		distance = Math.sqrt(distance);

		return distance;
	}

	public static double distanceBetweenBinaryPositions(List<Double> p1, List<Double> p2){
		double distance = 0;

		if(p1.size() != p2.size()){
			try {
				throw new Exception("Length of two positions are not equal!!!!");
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		for(int i=0;i<p1.size();i++){
			distance += Math.abs(p1.get(i) - p2.get(i));
		}

		return distance;
	}

	public static void main(String[] args){
		System.out.println(roundUp(1));
		System.out.println(roundUp(1.2));
		System.out.println(roundUp(1.5));
		System.out.println(roundUp(1.7));
		String s ="";
		System.out.println(s.split(",").length);
	}

}
