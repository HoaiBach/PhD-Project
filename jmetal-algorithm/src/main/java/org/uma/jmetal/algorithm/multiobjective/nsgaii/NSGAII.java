package org.uma.jmetal.algorithm.multiobjective.nsgaii;

import org.uma.jmetal.algorithm.impl.AbstractGeneticAlgorithm;
import org.uma.jmetal.operator.CrossoverOperator;
import org.uma.jmetal.operator.MutationOperator;
import org.uma.jmetal.operator.SelectionOperator;
import org.uma.jmetal.problem.Problem;
import org.uma.jmetal.solution.DoubleSolution;
import org.uma.jmetal.solution.Solution;
import org.uma.jmetal.util.SolutionListUtils;
import org.uma.jmetal.util.comparator.CrowdingDistanceComparator;
import org.uma.jmetal.util.evaluator.SolutionListEvaluator;
import org.uma.jmetal.util.solutionattribute.Ranking;
import org.uma.jmetal.util.solutionattribute.impl.CrowdingDistance;
import org.uma.jmetal.util.solutionattribute.impl.DominanceRanking;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * @author Antonio J. Nebro <antonio@lcc.uma.es>
 */
@SuppressWarnings("serial")
public class NSGAII<S extends Solution<?>> extends AbstractGeneticAlgorithm<S, List<S>> {
	protected final int maxIterations;
	private List<List<S>> recordSolutions;
	private int stepIteration;

	protected final SolutionListEvaluator<S> evaluator;

	protected int iterations;

	/**
	 * Constructor
	 */
	public NSGAII(Problem<S> problem, int maxIterations, int populationSize,
			CrossoverOperator<S> crossoverOperator, MutationOperator<S> mutationOperator,
			SelectionOperator<List<S>, S> selectionOperator, SolutionListEvaluator<S> evaluator) {
		super(problem);
		this.maxIterations = maxIterations;
		setMaxPopulationSize(populationSize); ;

		this.crossoverOperator = crossoverOperator;
		this.mutationOperator = mutationOperator;
		this.selectionOperator = selectionOperator;

		this.evaluator = evaluator;

		stepIteration = maxIterations/10;
		recordSolutions = new ArrayList<List<S>>();
	}

	@Override protected void initProgress() {
		iterations = 0;
	}

	@Override protected void updateProgress() {
		iterations += 1 ;
	}

	@Override protected boolean isStoppingConditionReached() {
		return iterations >= maxIterations-1;
	}

	@Override protected List<S> evaluatePopulation(List<S> population) {
		population = evaluator.evaluate(population, getProblem());

		return population;
	}

	@Override protected List<S> replacement(List<S> population, List<S> offspringPopulation) {
		List<S> jointPopulation = new ArrayList<>();
		jointPopulation.addAll(population);
		jointPopulation.addAll(offspringPopulation);

		Ranking<S> ranking = computeRanking(jointPopulation);

		return crowdingDistanceSelection(ranking);
	}

	@Override public List<S> getResult() {
		return getNonDominatedSolutions(getPopulation());
	}

	protected Ranking<S> computeRanking(List<S> solutionList) {
		Ranking<S> ranking = new DominanceRanking<S>();
		ranking.computeRanking(solutionList);

		return ranking;
	}

	protected List<S> crowdingDistanceSelection(Ranking<S> ranking) {
		CrowdingDistance<S> crowdingDistance = new CrowdingDistance<S>();
		List<S> population = new ArrayList<>(getMaxPopulationSize());
		int rankingIndex = 0;
		while (populationIsNotFull(population)) {
			if (subfrontFillsIntoThePopulation(ranking, rankingIndex, population)) {
				addRankedSolutionsToPopulation(ranking, rankingIndex, population);
				rankingIndex++;
			} else {
				crowdingDistance.computeDensityEstimator(ranking.getSubfront(rankingIndex));
				addLastRankedSolutionsToPopulation(ranking, rankingIndex, population);
			}
		}

		return population;
	}

	protected boolean populationIsNotFull(List<S> population) {
		return population.size() < getMaxPopulationSize();
	}

	protected boolean subfrontFillsIntoThePopulation(Ranking<S> ranking, int rank, List<S> population) {
		return ranking.getSubfront(rank).size() < (getMaxPopulationSize() - population.size());
	}

	protected void addRankedSolutionsToPopulation(Ranking<S> ranking, int rank, List<S> population) {
		List<S> front;

		front = ranking.getSubfront(rank);

		for (S solution : front) {
			population.add(solution);
		}
	}

	protected void addLastRankedSolutionsToPopulation(Ranking<S> ranking, int rank, List<S> population) {
		List<S> currentRankedFront = ranking.getSubfront(rank);

		Collections.sort(currentRankedFront, new CrowdingDistanceComparator<S>());

		int i = 0;
		while (population.size() < getMaxPopulationSize()) {
			population.add(currentRankedFront.get(i));
			i++;
		}
	}

	protected List<S> getNonDominatedSolutions(List<S> solutionList) {
		return SolutionListUtils.getNondominatedSolutions(solutionList);
	}

	@Override public String getName() {
		return "NSGAII" ;
	}

	@Override public String getDescription() {
		return "Nondominated Sorting Genetic Algorithm version II" ;
	}

	@Override public void run() {
		List<S> offspringPopulation;
		List<S> matingPopulation;

		population = createInitialPopulation();
		population = evaluatePopulation(population);
		initProgress();
		while (!isStoppingConditionReached()) {

			//update record list
			if((iterations%this.stepIteration)==0){
				List<S> list = new ArrayList<S>();
				for(S sol: population){
					list.add((S) sol.copy());
				}
				recordSolutions.add(list);
			}

			matingPopulation = selection(population);
			offspringPopulation = reproduction(matingPopulation);
			offspringPopulation = evaluatePopulation(offspringPopulation);
			population = replacement(population, offspringPopulation);
			updateProgress();
		}
	}
	
	public List<List<S>> getRecord(){
		return this.recordSolutions;
	}
}
