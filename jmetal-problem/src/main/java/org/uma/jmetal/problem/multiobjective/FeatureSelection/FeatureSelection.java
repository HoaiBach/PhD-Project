package org.uma.jmetal.problem.multiobjective.FeatureSelection;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.uma.jmetal.problem.impl.AbstractDoubleProblem;
import org.uma.jmetal.solution.DoubleSolution;
import org.uma.jmetal.solution.impl.DefaultDoubleSolution;
import org.uma.jmetal.util.pseudorandom.JMetalRandom;

import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;

@SuppressWarnings("serial")
public class FeatureSelection extends AbstractDoubleProblem {

	private int noFeatures;
	private Dataset training, testing;
	private MyClassifier classifier;
	private Random random;
	private double threshold = 0.6;
	private double[][] initializePop;
	private double currentNoFeatureRate = 0.0;
	private double featureRateStep=0.0;
	public int[] orderFeatures;//the smaller classification accracy, the topper the faeture


	public FeatureSelection(Random random, boolean test){
		//1. Read data file
		readData(test);

		//now set properties
		this.setNumberOfObjectives(2);
		this.setNumberOfVariables(noFeatures);
		this.setName("Feature Selection");

		//create an initialize pop
		featureRateStep = 1.0/this.getNumberOfVariables();
		createInitializePopRandom();

		//set lower, upper bound for each position
		List<Double> lowerLimit = new ArrayList<>(getNumberOfVariables());
		List<Double> upperLimit = new ArrayList<>(getNumberOfVariables());

		for(int i=0;i<this.getNumberOfVariables();i++){
			lowerLimit.add(0.0);
			upperLimit.add(1.0);
		}

		this.setLowerLimit(lowerLimit);
		this.setUpperLimit(upperLimit);

		this.random = random;

		initialiseOrderFeatures();
	}

	private void initialiseOrderFeatures() {
		orderFeatures = new int[this.getNumberOfVariables()];
		double[] accs = new double[this.getNumberOfVariables()];
		double[] features = new double[this.getNumberOfVariables()];

		for(int i=0;i<this.getNumberOfVariables();i++){
			orderFeatures[i] = i;
			features[i] = 1.0;

			double error = this.evaluateClassificationError(features, this.training);
			double acc = 1-error;
			accs[i] = acc;

			features[i] = 0.0;
		}

		minFastSort(accs, orderFeatures, this.getNumberOfVariables(), this.getNumberOfVariables());
	}

	public void minFastSort(double x[], int idx[], int n, int m) {
		for (int i = 0; i < m; i++) {
			for (int j = i + 1; j < n; j++) {
				if (x[i] > x[j]) {
					double temp = x[i];
					x[i] = x[j];
					x[j] = temp;
					int id = idx[i];
					idx[i] = idx[j];
					idx[j] = id;
				}
			}
		}
	}

	private void createInitializePopSequential() {
		System.out.println("Starting building sequential...");
		initializePop = new double[this.getNumberOfVariables()][this.getNumberOfVariables()];
		double[] currentSelected = new double[this.getNumberOfVariables()];
		List<Integer> unSelected = new ArrayList<Integer>();
		for(int i=0;i<this.getNumberOfVariables();i++){
			unSelected.add(i);
		}

		for(int i=0;i<this.getNumberOfVariables();i++){
			double lowestError = Double.MAX_VALUE;
			int toAdd = -1;
			for(Integer fIndex: unSelected){
				currentSelected[fIndex] = 1.0;
				double error = this.evaluateClassificationError(currentSelected,this.training);
				if(error< lowestError){
					lowestError = error;
					toAdd = fIndex;
				}
				currentSelected[fIndex] = 0.0;
			}

			//now update
			currentSelected[toAdd] = 1.0;
			unSelected.remove(new Integer(toAdd));
			initializePop[i] = Arrays.copyOf(currentSelected, currentSelected.length);
			System.out.println(Arrays.toString(currentSelected));
		}

		System.out.println("Finish building...");
	}

	private void createInitializePopRandom() {
		System.out.println("Starting building random...");
		initializePop = new double[this.getNumberOfVariables()][this.getNumberOfVariables()];

		for(int noFeature=1;noFeature<=this.getNumberOfVariables();noFeature++){
			int[] selected = new int[this.getNumberOfVariables()];
			randomPermutation(selected, this.getNumberOfVariables());
			double[] features= new double[this.getNumberOfVariables()];
			for(int j=0;j<noFeature;j++){
				features[selected[j]] = 1.0;
			}
			initializePop[noFeature-1] = features;
		}
		System.out.println("End building random...");
	}

	public void randomPermutation(int[] perm, int size) {
		JMetalRandom randomGenerator = JMetalRandom.getInstance() ;
		int[] index = new int[size];
		boolean[] flag = new boolean[size];

		for (int n = 0; n < size; n++) {
			index[n] = n;
			flag[n] = true;
		}

		int num = 0;
		while (num < size) {
			int start = randomGenerator.nextInt(0, size - 1);
			while (true) {
				if (flag[start]) {
					perm[num] = index[start];
					flag[start] = false;
					num++;
					break;
				}
				if (start == (size - 1)) {
					start = 0;
				} else {
					start++;
				}
			}
		}
	}

	private void readData(boolean test) {
		try {
			noFeatures = MainHelp.noFeature();
			Dataset[] trainTest = MainHelp.readBingData(noFeatures);

			if(!test){
				setTraining(trainTest[0]);
				setTesting(trainTest[1]);
			}
			else{
				Dataset trainingHalf = new DefaultDataset();
				for(int i=0;i<trainTest[0].size()/2;i++){
					trainingHalf.add(trainTest[0].get(i));
				}
				Dataset testHalf = new DefaultDataset();
				for(int i=0;i<trainTest[1].size()/2;i++){
					testHalf.add(trainTest[1].get(i));
				}
				setTraining(trainingHalf);
				setTesting(testHalf);
			}

			classifier = new MyClassifier(new Random(1));
			classifier.ClassifierBingKNN(5);
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(0);
		}

	}


	public void evaluate(DoubleSolution sol) {
		double[] features = solutionToBits(sol);
		int sizeSubset = this.sizeSubset(features);
		double featureRate = (sizeSubset+0.0)/this.getNumberOfVariables();

		//find accuracy
		Dataset temTrain = this.training.copy();
		temTrain = HelpDataset.removeFeatures(temTrain, features);
		int folds = 10;
		Dataset[] foldsTem = temTrain.folds(folds, new Random(1));
		double accuracy = 0.0;
		for (int f = 0; f < folds ; f++) {
			Dataset testTem = foldsTem[f];
			Dataset trainTem = new DefaultDataset();
			for (int j = 0; j < folds; j++) {
				if (j != f) {
					trainTem.addAll(foldsTem[j]);
				}
			}
			accuracy += this.classifier.classify(trainTem, testTem);
		}
		double error = 1.0-accuracy/folds;

		//build array of objectives, 0- featureRate, 1- error
		if(featureRate ==0)
			error = 1.0;
		sol.setObjective(0, featureRate);
		sol.setObjective(1, error);
	}

	public double evaluateClassificationError(double[] features, Dataset dataset) {
		int sizeSubset = this.sizeSubset(features);
		double featureRate = (sizeSubset+0.0)/this.getNumberOfVariables();

		//find accuracy
		Dataset temDataset = dataset.copy();
		temDataset = HelpDataset.removeFeatures(temDataset, features);
		int folds = 10;
		Dataset[] foldsTem = temDataset.folds(folds, new Random(1));
		double accuracy = 0.0;
		for (int f = 0; f < folds ; f++) {
			Dataset testTem = foldsTem[f];
			Dataset trainTem = new DefaultDataset();
			for (int j = 0; j < folds; j++) {
				if (j != f) {
					trainTem.addAll(foldsTem[j]);
				}
			}
			accuracy += this.classifier.classify(trainTem, testTem);
		}
		double error = 1.0-accuracy/folds;

		//build array of objectives, 0- featureRate, 1- error
		if(featureRate ==0)
			error = 1.0;
		return error;
	}

	public double[] solutionToBits(DoubleSolution sol){
		double[] features = new double[sol.getNumberOfVariables()];
		for(int i=0;i<sol.getNumberOfVariables();i++){
			if(sol.getVariableValue(i)>this.threshold){
				features[i] = 1.0;
			}
			else{
				features[i] = 0.0;
			}
		}
		return features;
	}

	public int sizeSubset(double[] features){
		int count = 0;
		for(int i=0;i<features.length;i++){
			if(features[i]>this.threshold)
				count++;
		}
		return count;
	}

	public DoubleSolution createSolution() {
		//currentNoFeatureRate += this.featureRateStep;
		//return createSolution(currentNoFeatureRate);
		return createSolutionRandom();
	}

	public DoubleSolution createSolutionRandom(){
		DoubleSolution solution = new DefaultDoubleSolution(this);
		for(int i=0;i<this.getNumberOfVariables();i++){
			solution.setVariableValue(i, this.random.nextDouble());
		}
		return solution;
	}

	public DoubleSolution createSolutionRandom(double featureRate){
		DoubleSolution solution = new DefaultDoubleSolution(this);
		int noFeatures = (int) (featureRate*this.getNumberOfVariables());
		int[] selected = new int[this.getNumberOfVariables()];
		randomPermutation(selected, this.getNumberOfVariables());
		double[] features= new double[this.getNumberOfVariables()];
		for(int i=0;i<noFeatures;i++){
			features[selected[i]] = 1.0;
		}
		for(int i=0; i<this.getNumberOfVariables();i++){
			solution.setVariableValue(i, features[i]);
		}
		return solution;
	}

	public DoubleSolution createSolution(double featureRate){
		DoubleSolution solution = new DefaultDoubleSolution(this);
		int noFeatures = (int) (featureRate*this.getNumberOfVariables());
		if(noFeatures < 1)
			noFeatures = 1;
		if(noFeatures > this.getNumberOfVariables())
			noFeatures = this.getNumberOfVariables();
		double[] features = initializePop[noFeatures-1];
		for(int i=0;i<this.getNumberOfVariables();i++){
			if(features[i] ==1.0){
				solution.setVariableValue(i, this.threshold + this.random.nextDouble()*(1.0-this.threshold));
			}
			else{
				solution.setVariableValue(i, this.random.nextDouble()*this.threshold);
			}
		}
		return solution;
	}

	/**
	 * add more features to a solution
	 * @param sol
	 * @param refRate
	 */
	public void increaseSize(DoubleSolution sol, double refRate){
		List<Integer> selected = new ArrayList<Integer>();
		List<Integer> unselected = new ArrayList<Integer>();
		for(int i=0;i<sol.getNumberOfVariables();i++){
			if(sol.getVariableValue(i) > getThreshold()){
				selected.add(i);
			}
			else{
				unselected.add(i);
			}
		}

		//build an array of sorted unselected features
		//0-size: increasing classification accuracy
		int[] idxs = new int[unselected.size()];
		int index=0;
		for(int i=0;i<orderFeatures.length;i++){
			int fIndex = orderFeatures[i];
			if(unselected.contains(fIndex)){
				idxs[index] = fIndex;
				index++;
			}
		}

		//start adding features from the higher clasisifcaiton accuracy
		index = unselected.size()-1;

		while(refRate >= (selected.size()+1)/(sol.getNumberOfVariables()+0.0) && index>=0){
			//if selecting more features
			//removing features
			int fToAdd = idxs[index];
			selected.add(fToAdd);
			sol.setVariableValue(fToAdd, getThreshold()*1.1);
			index--;
		}
	}

	/**
	 * add more features to a solution (random way)
	 * @param sol
	 * @param refRate
	 */
	public void increaseSizeRandom(DoubleSolution sol, double refRate){
		List<Integer> selected = new ArrayList<Integer>();
		List<Integer> unselected = new ArrayList<Integer>();
		for(int i=0;i<sol.getNumberOfVariables();i++){
			if(sol.getVariableValue(i) > getThreshold()){
				selected.add(i);
			}
			else{
				unselected.add(i);
			}
		}

		//shuffle the unselected
		Collections.shuffle(unselected, this.random);

		//start adding features

		while(refRate >= (selected.size()+1)/(sol.getNumberOfVariables()+0.0) && !unselected.isEmpty()){
			//if selecting more features
			//removing features
			int fToAdd = unselected.get(0);
			unselected.remove(0);
			selected.add(fToAdd);
			sol.setVariableValue(fToAdd, getThreshold()*1.1);
		}
	}

	/**
	 * reduce features from a set to meet the refRate
	 * @param sol
	 * @param refRate
	 */
	public void reduceSize(DoubleSolution sol, double refRate){
		List<Integer> selected = new ArrayList<Integer>();
		List<Integer> unselected = new ArrayList<Integer>();
		for(int i=0;i<sol.getNumberOfVariables();i++){
			if(sol.getVariableValue(i) > getThreshold()){
				selected.add(i);
			}
			else{
				unselected.add(i);
			}
		}

		if(refRate >= selected.size()/(sol.getNumberOfVariables()+0.0))
			return;

		//build an array of sorted features
		int[] idxs = new int[selected.size()];
		int index=0;
		for(int i=0;i<orderFeatures.length;i++){
			int fIndex = orderFeatures[i];
			if(selected.contains(fIndex)){
				idxs[index] = fIndex;
				index++;
			}
		}

		//start removing features
		index = 0;

		while(refRate < selected.size()/(sol.getNumberOfVariables()+0.0) && index <idxs.length){
			//if selecting more features
			//removing features
			int fToRemove = idxs[index];
			selected.remove(new Integer(fToRemove));
			sol.setVariableValue(fToRemove, getThreshold());
			index++;
		}
	}

	/**
	 * reduce features from a set to meet the refRate randomly
	 * @param sol
	 * @param refRate
	 */
	public void reduceSizeRandom(DoubleSolution sol, double refRate){
		List<Integer> selected = new ArrayList<Integer>();
		List<Integer> unselected = new ArrayList<Integer>();
		for(int i=0;i<sol.getNumberOfVariables();i++){
			if(sol.getVariableValue(i) > getThreshold()){
				selected.add(i);
			}
			else{
				unselected.add(i);
			}
		}

		if(refRate >= selected.size()/(sol.getNumberOfVariables()+0.0))
			return;

		//shuffle the selected features
		Collections.shuffle(selected,this.random);

		//start removing features
		while(refRate < selected.size()/(sol.getNumberOfVariables()+0.0)){
			//if selecting more features
			//removing features
			int fToRemove = selected.get(0);
			selected.remove(0);
			sol.setVariableValue(fToRemove, getThreshold()*0.9);
		}
	}

	public Dataset getTraining() {
		return training;
	}

	public void setTraining(Dataset training) {
		this.training = training;
	}

	public Dataset getTesting() {
		return testing;
	}

	public void setTesting(Dataset testing) {
		this.testing = testing;
	}

	public Random getRandom() {
		return random;
	}

	public void setRandom(Random random) {
		this.random = random;
	}

	public double getThreshold() {
		return threshold;
	}

	public void setThreshold(double threshold) {
		this.threshold = threshold;
	}

	public double[][] getInitializePop() {
		return initializePop;
	}

	public void setInitializePop(double[][] initializePop) {
		this.initializePop = initializePop;
	}

	public void setCurrentFeatureRate(double featureRate){
		this.currentNoFeatureRate = featureRate;
	}

	public MyClassifier getClassifier(){
		return this.classifier;
	}
}
