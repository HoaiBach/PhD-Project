package org.uma.jmetal.algorithm.multiobjective.nsgaii;

import org.uma.jmetal.measure.Measurable;
import org.uma.jmetal.measure.MeasureManager;
import org.uma.jmetal.measure.impl.BasicMeasure;
import org.uma.jmetal.measure.impl.CountingMeasure;
import org.uma.jmetal.measure.impl.DurationMeasure;
import org.uma.jmetal.measure.impl.SimpleMeasureManager;
import org.uma.jmetal.operator.CrossoverOperator;
import org.uma.jmetal.operator.MutationOperator;
import org.uma.jmetal.operator.SelectionOperator;
import org.uma.jmetal.problem.Problem;
import org.uma.jmetal.solution.Solution;
import org.uma.jmetal.util.evaluator.SolutionListEvaluator;
import org.uma.jmetal.util.solutionattribute.Ranking;

import java.util.List;

/**
 * @author Antonio J. Nebro <antonio@lcc.uma.es>
 */
@SuppressWarnings("serial")
public class NSGAIIMeasures<S extends Solution<?>> extends NSGAII<S> implements Measurable {
  protected CountingMeasure evaluations ;
  protected DurationMeasure durationMeasure ;
  protected SimpleMeasureManager measureManager ;

  protected BasicMeasure<List<S>> solutionListMeasure ;
  protected BasicMeasure<Integer> numberOfNonDominatedSolutionsInPopulation ;

  /**
   * Constructor
   */
  public NSGAIIMeasures(Problem<S> problem, int maxIterations, int populationSize,
      CrossoverOperator<S> crossoverOperator, MutationOperator<S> mutationOperator,
      SelectionOperator<List<S>, S> selectionOperator, SolutionListEvaluator<S> evaluator) {
    super(problem, maxIterations, populationSize, crossoverOperator, mutationOperator,
        selectionOperator, evaluator) ;

    initMeasures() ;
  }

  @Override protected void initProgress() {
    evaluations.reset(getMaxPopulationSize());
  }

  @Override protected void updateProgress() {
    evaluations.increment(getMaxPopulationSize());

    solutionListMeasure.push(getPopulation());
  }

  @Override protected boolean isStoppingConditionReached() {
    return evaluations.get() >= maxIterations;
  }

  @Override
  public void run() {
    durationMeasure.reset();
    durationMeasure.start();
    super.run();
    durationMeasure.stop();
  }

  /* Measures code */
  private void initMeasures() {
    durationMeasure = new DurationMeasure() ;
    evaluations = new CountingMeasure(0) ;
    numberOfNonDominatedSolutionsInPopulation = new BasicMeasure<>() ;
    solutionListMeasure = new BasicMeasure<>() ;

    measureManager = new SimpleMeasureManager() ;
    measureManager.setPullMeasure("currentExecutionTime", durationMeasure);
    measureManager.setPullMeasure("currentEvaluation", evaluations);
    measureManager.setPullMeasure("numberOfNonDominatedSolutionsInPopulation",
        numberOfNonDominatedSolutionsInPopulation);

    measureManager.setPushMeasure("currentPopulation", solutionListMeasure);
    measureManager.setPushMeasure("currentEvaluation", evaluations);
  }

  @Override
  public MeasureManager getMeasureManager() {
    return measureManager ;
  }

  @Override protected List<S> replacement(List<S> population,
      List<S> offspringPopulation) {
    List<S> pop = super.replacement(population, offspringPopulation) ;

    Ranking<S> ranking = computeRanking(pop);
    numberOfNonDominatedSolutionsInPopulation.set(ranking.getSubfront(0).size());

    return pop;
  }

  public CountingMeasure getEvaluations() {
    return evaluations;
  }

  @Override public String getName() {
    return "NSGAIIM" ;
  }

  @Override public String getDescription() {
    return "Nondominated Sorting Genetic Algorithm version II. Version using measures" ;
  }
}
