package org.uma.jmetal.algorithm.multiobjective.moead;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.uma.jmetal.algorithm.multiobjective.moead.util.MOEADUtils;
import org.uma.jmetal.operator.CrossoverOperator;
import org.uma.jmetal.operator.MutationOperator;
import org.uma.jmetal.operator.impl.crossover.DifferentialEvolutionCrossover;
import org.uma.jmetal.problem.DoubleProblem;
import org.uma.jmetal.problem.Problem;
import org.uma.jmetal.problem.multiobjective.FeatureSelection.FeatureSelection;
import org.uma.jmetal.solution.DoubleSolution;
import org.uma.jmetal.solution.impl.DefaultDoubleSolution;
import org.uma.jmetal.util.JMetalException;

public class MOEADDYN extends AbstractMOEAD<DoubleSolution>{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private DifferentialEvolutionCrossover differentialEvolutionCrossover;
	private double[] refPoints;
	private int[] upIndex;
	private double[] upBoundaries;
	private int noFreeRef;
	private int noFixRef;
	private double weight = 1.0;
	private double threshold = 0.6;
	private Random random;

	//inner values
	private int noIntervals;
	private int iterationInterval;

	/**
	 * Remember to set random, threshold, fromPopulation, otherwise it will not work
	 * @param problem
	 * @param populationSize
	 * @param resultPopulationSize
	 * @param maxEvaluations
	 * @param crossoverOperator
	 * @param mutation
	 * @param functionType
	 * @param dataDirectory
	 * @param neighborhoodSelectionProbability
	 * @param maximumNumberOfReplacedSolutions
	 * @param neighborSize
	 * @param random
	 * @param threshold
	 * @param fromPopulation
	 */
	public MOEADDYN(Problem<DoubleSolution> problem,
			int populationSize,
			int resultPopulationSize,
			int maxIterations,
			CrossoverOperator<DoubleSolution> crossoverOperator,
			MutationOperator<DoubleSolution> mutation,
			FunctionType functionType,
			String dataDirectory,
			double neighborhoodSelectionProbability,
			int maximumNumberOfReplacedSolutions,
			int neighborSize) {
		super(problem, populationSize, resultPopulationSize, maxIterations, crossoverOperator, mutation, functionType,
				dataDirectory, neighborhoodSelectionProbability, maximumNumberOfReplacedSolutions, neighborSize);

		differentialEvolutionCrossover = (DifferentialEvolutionCrossover)crossoverOperator ;
	}


	public void initializeDynamic(double fixRate, int noInterval){
		//build the number of fixed reference points in each interval
		try {
			this.noIntervals = noInterval;
			this.iterationInterval = this.maxIterations/this.noIntervals;
			//noFixRef does not include 0
			noFixRef = (int)(fixRate*populationSize);
			//give 1 to the first 0
			if(noFixRef == populationSize)
				noFixRef = noFixRef-1;
			//-1 because the first one is allocated to 0
			noFreeRef = populationSize-noFixRef-1;

			if(noFixRef < noInterval)
				throw new Exception("Each interval has to have at least one individual!");

			//assign fixref to each interval
			int[] sizes = new int[noInterval];
			int totalSize = 0;
			int eachInterval = noFixRef/noInterval;
			for(int i=0;i<sizes.length;i++){
				sizes[i] = eachInterval;
				totalSize+=eachInterval;
			}
			int indexInterval = 0;
			while(totalSize < noFixRef){
				sizes[indexInterval]++;
				totalSize++;
				indexInterval = (indexInterval+1)%noInterval;
			}

			//for each interval store the upper bound and the upper
			//index in the population
			double step = 1.0/noInterval;
			this.upBoundaries = new double[noInterval];
			this.upIndex = new int[noInterval];
			for(int i=0;i<noInterval;i++){
				upBoundaries[i] = (i+1)*step;
				upIndex[i] = i>0? upIndex[i-1]+sizes[i]:sizes[i];
			}

			for(int i=1;i<upIndex.length;i++){
				if((upIndex[i]-upIndex[i-1])!=sizes[i])
					throw new Exception("Two upIndex must have the \"size\" distance!");
			}

			//now allocate fix reference points for each interval
			//note that the first one is allocated to 0
			refPoints = new double[populationSize];
			for(int intervalIndex =0; intervalIndex < noInterval; intervalIndex++){
				double[] refs = intervalIndex >0 ?
						allocateReferences(upBoundaries[intervalIndex-1], upBoundaries[intervalIndex], sizes[intervalIndex]):
							allocateReferences(0, upBoundaries[intervalIndex], sizes[intervalIndex]);
						int startIndex = intervalIndex >0? upIndex[intervalIndex-1]+1:1;
						for(int i=startIndex;i<=upIndex[intervalIndex];i++){
							refPoints[i] = refs[i-startIndex];
						}
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	/**
	 * allocate "size" reference points to the interval
	 * low (exclusive) and up (inclusive)
	 * @param low
	 * @param up
	 * @param size
	 */
	public double[] allocateReferences(double low, double up, int size){
		if(size>0){
			double[] refs = new double[size];
			double step = (up-low)/size;
			for(int i=1;i<=size;i++){
				refs[i-1] = low+i*step;
			}
			return refs;
		}
		else return null;
	}

	/**
	 * Allocate the free referecents in the interval (low,up]
	 * Every time the free referecens are reallocated 
	 * need to re-initialize the neighbour hood
	 * if necessary, repair the solutions
	 * @param low
	 * @param up
	 */
	public void allocateFreeReferences(double low, double up, boolean reallocate){
		if(noFreeRef>0){
			double[] freeRef = allocateReferencesChecked(low, up, noFreeRef);
			for(int i= noFixRef+1;i<populationSize;i++)
				refPoints[i] = freeRef[i-noFixRef-1];

			//update neighbor hood
			initializeNeighborhood();
			if(reallocate){
				for(int i= noFixRef+1;i<populationSize;i++)
					//using sequential add using single accuracy
					repairSolution(population.get(i), refPoints[i]);
			}
		}
		else
			initializeNeighborhood();
	}

	/**
	 * allocate "size" reference points to the interval
	 * low (exclusive) and up (inclusive), this one 
	 * checks to avoid duplicated reference
	 * @param low
	 * @param up
	 * @param size
	 */
	public double[] allocateReferencesChecked(double low, double up, int size){
		if(size>0){
			int lowNf = (int) (low*this.problem.getNumberOfVariables());
			int upNf = (int) (up*this.problem.getNumberOfVariables());
			List<Integer> unassignedNf = new ArrayList<Integer>();
			for(int nf=lowNf+1; nf <= upNf; nf++){
				unassignedNf.add(nf);
			}

			//go through to find the fixed nf in the interval
			//remove it from unassigned list
			for(int i=0;i<=noFixRef;i++){
				int nf = (int) (refPoints[i]*this.problem.getNumberOfVariables());
				if(lowNf < nf && nf <= upNf){
					unassignedNf.remove(new Integer(nf));
				}
			}

			//if the unassigned is smallr than size, add more reference points, starting from the middle
			int middle = (upNf+lowNf)/2;
			int currentBound = middle;
			while(unassignedNf.size() < size){
				if(currentBound >= lowNf){
					unassignedNf.add(currentBound);
					unassignedNf.add(middle + (middle - currentBound));
					currentBound--;
				}
				else{
					currentBound = middle;
				}
			}

			//now assign the unassigned nf
			double[] refs = new double[size];

			//make sure no allocation to 0
			for(int i=0;i<size;i++){
				int nf = unassignedNf.get(i);
				refs[i] = nf >0 ? (nf+0.0)/this.problem.getNumberOfVariables(): 1.0/this.problem.getNumberOfVariables();
			}

			return refs;
		}
		else return null;
	}

	/**
	 * repair solution, if the number of features selected by the solution:
	 * is higher than the refPoint -> remove
	 * is lower than the refPoint -> add as much
	 * @param child
	 * @param maxFRate
	 */
	private void repairSolution(DoubleSolution sol, double refRate) {
		if(sol.getObjective(0) < refRate){
			((FeatureSelection)this.problem).increaseSize(sol, refRate);
			this.problem.evaluate(sol);
		}
		else if(sol.getObjective(1) > refRate){
			((FeatureSelection)this.problem).reduceSize(sol, refRate);
			this.problem.evaluate(sol);
		}
	}

	/**
	 * repair solution, if the number of features selected by the solution:
	 * is higher than the refPoint -> remove
	 * is lower than the refPoint -> add as much (randomly)
	 * @param child
	 * @param maxFRate
	 */
	private void repairSolutionRandom(DoubleSolution sol, double refRate) {
		if(sol.getObjective(0) < refRate){
			((FeatureSelection)this.problem).increaseSizeRandom(sol, refRate);
			this.problem.evaluate(sol);
		}
		else if(sol.getObjective(1) > refRate){
			((FeatureSelection)this.problem).reduceSizeRandom(sol, refRate);
			this.problem.evaluate(sol);
		}
	}


	private DoubleSolution randomSolution(double refRate){
		DoubleSolution newSolution = ((FeatureSelection)this.problem).createSolutionRandom(refRate);
		this.problem.evaluate(newSolution);
		return newSolution;
	}

	/**
	 * Appy DE mechanism to generate the new solution.
	 * @param subProblemId
	 * @return
	 */
	public DoubleSolution repairSolutionDE(int subProblemId){
		NeighborType neighborType = chooseNeighborType() ;
		List<DoubleSolution> parents = parentSelection(subProblemId, neighborType) ;

		differentialEvolutionCrossover.setCurrentSolution(population.get(subProblemId));
		List<DoubleSolution> children = differentialEvolutionCrossover.execute(parents);

		DoubleSolution child = children.get(0) ;
		mutationOperator.execute(child);
		((FeatureSelection)this.problem).reduceSize(child, refPoints[subProblemId]);
		problem.evaluate(child);
		return child;
	}


	@Override
	public void run() {
		//now initialize the population
		initializePopulation();

		//allocate the free resouce to the first interval
		int currentInterval = 0;
		allocateFreeReferences(0, upBoundaries[0],false);
		initializeIdealPoint() ;

		boolean stillConflicting = true;

		int iteration = 0;
		while(iteration < this.maxIterations){

			if((iteration%this.stepIteration)==0){
				List<DoubleSolution> list = new ArrayList<DoubleSolution>();
				for(DoubleSolution sol: population){
					list.add((DoubleSolution) sol.copy());
				}
				recordSolutions.add(list);
			}

			int[] permutation = new int[populationSize];
			MOEADUtils.randomPermutation(permutation, populationSize);
			for (int i = 0; i < populationSize; i++) {
				int subProblemId = permutation[i];

				NeighborType neighborType = chooseNeighborType() ;
				List<DoubleSolution> parents = parentSelection(subProblemId, neighborType) ;

				differentialEvolutionCrossover.setCurrentSolution(population.get(subProblemId));
				List<DoubleSolution> children = differentialEvolutionCrossover.execute(parents);

				DoubleSolution child = children.get(0) ;
				mutationOperator.execute(child);
				((FeatureSelection)this.problem).reduceSize(child, refPoints[subProblemId]);
				problem.evaluate(child);
				updateIdealPoint(child);
				updateNeighborhood(child, subProblemId, neighborType);
			}

			//if 2 objectives are still conflicting and 
			//it is time to check whether the 2 objectives are conlicting
			if(stillConflicting && iteration == ((currentInterval+1)*iterationInterval-1))
			{
				//check to see whether they are still conflicting or not
				//System.out.println("======="+iteration+"========");
				stillConflicting = checkConflicting(currentInterval);
				//solutions in the previous interval
				//				if(currentInterval>0){
				//					System.out.println(Arrays.toString(refPoints));
				//					System.out.println("Previous:");
				//					for(int i=currentInterval >=2 ? upIndex[currentInterval-2]+1:0;i<=upIndex[currentInterval-1];i++)
				//					{
				//						DoubleSolution s = population.get(i);
				//						System.out.println(s.getObjective(0)+": "+s.getObjective(1));
				//					}
				//					//soultions in the current interval
				//					System.out.println("Current:");
				//					for(int i=upIndex[currentInterval-1]+1;i<=upIndex[currentInterval];i++)
				//					{
				//						DoubleSolution s = population.get(i);
				//						System.out.println(s.getObjective(0)+": "+s.getObjective(1));
				//					}
				//					System.out.println("Still conflicting: "+stillConflicting);
				//				}
				//if at this interval, they are not conflicting anymore, then stop figure out
				//allocate all resouces in the conflicting parts
				if(!stillConflicting){
					int focusInterval = (currentInterval-1) < 0? 0: (currentInterval-1);
					allocateFreeReferences(0, upBoundaries[focusInterval],true);
				}
				//if still conflicting, increase the current interval
				//reallocate free resources in the new interval
				else if(currentInterval < noIntervals-1)
				{
					currentInterval++;
					allocateFreeReferences(upBoundaries[currentInterval-1], upBoundaries[currentInterval],true);
				}
			}

			//check for the same one in the population before evaluating
			//if there exist some solutions then create a random position
			if(!stillConflicting){
				List<Integer> indexToRandom = checkForDuplicate();
				for(int index: indexToRandom){
					DoubleSolution sol = this.population.get(index);
					double refRate = this.refPoints[index];
					((FeatureSelection)this.problem).increaseSizeRandom(sol, refRate);
					this.problem.evaluate(sol);
				}
			}

			updateExternalPopulation();
			iteration++;

		} 
	}

	/**
	 * check for the same one in the population before evaluating
	 * the duplicated one with higher reference points will be random
	 * @return
	 */
	public List<Integer> checkForDuplicate(){
		List<Integer> duplicate = new ArrayList<Integer>();

		//start from one solution
		for(int i=0; i < populationSize-1; i++){
			//if the solution is not duplicated -> perform checking
			if(!duplicate.contains(i)){
				int master = i;
				DoubleSolution ms = this.population.get(i);
				//start checking from behind
				for(int j=master+1; j<populationSize;j++){
					if(!duplicate.contains(j)){
						DoubleSolution ss = this.population.get(j);
						//if they have the same objectives
						if(ss.getObjective(0) == ms.getObjective(0) &&
								ss.getObjective(1) == ms.getObjective(1)){
							//if the refpoint of master is smaller -> the other is duplicated
							if(refPoints[i] <= refPoints[j])
								duplicate.add(j);
							//otherwise the master is duplicated and stop checking for this master
							//add it to the duplicate list
							else{
								duplicate.add(i);
								break;
							}
						}
					}
				}
			}
		}

		return duplicate;
	}

	public DoubleSolution createSolution(double featureRate){
		int noFeatures = (int) (featureRate*problem.getNumberOfVariables());
		if(noFeatures < 1)
			noFeatures = 1;
		if(noFeatures > problem.getNumberOfVariables())
			noFeatures = problem.getNumberOfVariables();

		int[] selected = new int[problem.getNumberOfVariables()];
		MOEADUtils.randomPermutation(selected, problem.getNumberOfVariables());
		double[] features= new double[problem.getNumberOfVariables()];
		for(int j=0;j<noFeatures;j++){
			features[selected[j]] = 1.0;
		}

		DoubleSolution solution = new DefaultDoubleSolution((DoubleProblem) problem);
		for(int i=0;i<problem.getNumberOfVariables();i++){
			if(features[i] ==1.0){
				solution.setVariableValue(i, this.threshold + this.random.nextDouble()*(1.0-this.threshold));
			}
			else{
				solution.setVariableValue(i, this.random.nextDouble()*this.threshold);
			}
		}
		problem.evaluate(solution);
		return solution;
	}

	public boolean checkConflicting(int currentInterval){

		List<DoubleSolution> previous = new ArrayList<DoubleSolution>();
		List<DoubleSolution> current = new ArrayList<DoubleSolution>();

		if(currentInterval <=0 )
			return true;
		else if(currentInterval >= noIntervals){
			return false;
		}
		else{
			//solutions in the previous interval
			for(int i=currentInterval >=2? upIndex[currentInterval-2]+1:0;i<=upIndex[currentInterval-1];i++)
				previous.add(population.get(i));
			//soultions in the current interval
			for(int i=upIndex[currentInterval-1]+1;i<=upIndex[currentInterval];i++)
				current.add(population.get(i));

			//for each solution in the current interval
			boolean allBeDominatedEqual = true;

			for(DoubleSolution cs: current){
				boolean beDominatedEqual = false;

				for(DoubleSolution ps: previous){
					boolean compare = ps.getObjective(0)<=cs.getObjective(0)
							&& ps.getObjective(1) <= cs.getObjective(1);
					beDominatedEqual |= compare;
				}

				if(!beDominatedEqual){
					allBeDominatedEqual = false;
					break;
				}
			}
			//if all solutions of the current one are dominated by the previous one
			//it indicates that the two objectives are not conflicting any more
			if(allBeDominatedEqual)
				return false;
			else
				return true;
		}
	}

	protected void initializePopulation() {
		for (int i = 0; i < populationSize; i++) {
			DoubleSolution newSolution = (DoubleSolution)problem.createSolution();

			problem.evaluate(newSolution);
			population.add(newSolution);
		}
	}

	protected void initializeNeighborhood() {
		double[] x = new double[populationSize];
		int[] idx = new int[populationSize];
		//
		for (int i = 0; i < populationSize; i++) {
			// calculate the distances based on the ref points
			for (int j = 0; j < populationSize; j++) {
				x[j] = Math.abs(refPoints[i]-refPoints[j]);
				idx[j] = j;
			}
			//
			// find 'niche' nearest neighboring subproblems
			MOEADUtils.minFastSort(x, idx, populationSize, neighborSize);
			//
			System.arraycopy(idx, 0, neighborhood[i], 0, neighborSize);
		}
	}

	protected  void updateNeighborhood(DoubleSolution individual,
			int subProblemId,
			NeighborType neighborType) throws JMetalException {
		int size;
		int time;

		time = 0;

		if (neighborType == NeighborType.NEIGHBOR) {
			size = neighborhood[subProblemId].length;
		} else {
			size = population.size();
		}
		int[] perm = new int[size];

		MOEADUtils.randomPermutation(perm, size);

		for (int i = 0; i < size; i++) {
			int k;
			if (neighborType == NeighborType.NEIGHBOR) {
				k = neighborhood[subProblemId][perm[i]];
			} else {
				k = perm[i];
			}
			double f1, f2;

			f1 = fitnessFunction(population.get(k),refPoints[k]);
			f2 = fitnessFunction(individual, refPoints[k]);

			if (f2 < f1) {
				population.set(k, (DoubleSolution) individual.copy());
				time++;
			}

			if (time >= maximumNumberOfReplacedSolutions) {
				return;
			}
		}
	}

	/**
	 * this is designed for FS
	 * @param individual
	 * @return
	 * @throws JMetalException
	 */
	public double fitnessFunction(DoubleSolution individual, double refPoint) throws JMetalException{
		double fitness = 0;
		if(problem.getNumberOfObjectives() != 2){
			System.out.println("This is designed for feature selection only!!");
			System.exit(0);
		}

		fitness += weight*individual.getObjective(1);
		fitness += 100000*Math.max(individual.getObjective(0)-refPoint, 0);
		fitness += 0.01*individual.getObjective(0);

		return fitness;
	}

	@Override
	public String getName() {
		return "MOEADDYN";
	}

	@Override
	public String getDescription() {
		return null;
	}

	public void setWeight(double weight){
		this.weight = weight;
	}

	public void setRandom(Random random){
		this.random = random;
	}

	public void setThreshold(double threshold){
		this.threshold = threshold;
	}

	public double round(double value){
		return ((int)(value*100+0.5))/100.0;
	}
}
