//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU Lesser General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU Lesser General Public License for more details.
//
//  You should have received a copy of the GNU Lesser General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.

package org.uma.jmetal.algorithm.multiobjective.moead;

import java.util.ArrayList;
import java.util.List;

import org.uma.jmetal.algorithm.multiobjective.moead.util.MOEADUtils;
import org.uma.jmetal.operator.CrossoverOperator;
import org.uma.jmetal.operator.MutationOperator;
import org.uma.jmetal.operator.impl.crossover.DifferentialEvolutionCrossover;
import org.uma.jmetal.problem.Problem;
import org.uma.jmetal.solution.DoubleSolution;

/**
 * Class implementing the MOEA/D-DE algorithm described in :
 * Hui Li; Qingfu Zhang, "Multiobjective Optimization Problems With Complicated Pareto Sets,
 * MOEA/D and NSGA-II," Evolutionary Computation, IEEE Transactions on , vol.13, no.2, pp.284,302,
 * April 2009. doi: 10.1109/TEVC.2008.925798
 *
 * @author Antonio J. Nebro
 * @version 1.0
 */
@SuppressWarnings("serial")
public class MOEAD extends AbstractMOEAD<DoubleSolution> {
  private DifferentialEvolutionCrossover differentialEvolutionCrossover ;

  public MOEAD(Problem<DoubleSolution> problem,
      int populationSize,
      int resultPopulationSize,
      int maxIterations,
      MutationOperator<DoubleSolution> mutation,
      CrossoverOperator<DoubleSolution> crossover,
      FunctionType functionType,
      String dataDirectory,
      double neighborhoodSelectionProbability,
      int maximumNumberOfReplacedSolutions,
      int neighborSize) {
    super(problem, populationSize, resultPopulationSize, maxIterations, crossover, mutation, functionType,
        dataDirectory, neighborhoodSelectionProbability, maximumNumberOfReplacedSolutions,
        neighborSize);

    differentialEvolutionCrossover = (DifferentialEvolutionCrossover)crossoverOperator ;
  }

  @Override public void run() {
    initializePopulation() ;
    initializeUniformWeight();
    initializeNeighborhood();

    int iterations = 0 ;
    do {
      int[] permutation = new int[populationSize];
      MOEADUtils.randomPermutation(permutation, populationSize);

      if((iterations%this.stepIteration)==0){
    	  List<DoubleSolution> list = new ArrayList<DoubleSolution>();
    	  for(DoubleSolution sol: population){
    		  list.add((DoubleSolution) sol.copy());
    	  }
    	  recordSolutions.add(list);
      }
      
      for (int i = 0; i < populationSize; i++) {
        int subProblemId = permutation[i];

        NeighborType neighborType = chooseNeighborType() ;
        List<DoubleSolution> parents = parentSelection(subProblemId, neighborType) ;

        differentialEvolutionCrossover.setCurrentSolution(population.get(subProblemId));
        List<DoubleSolution> children = differentialEvolutionCrossover.execute(parents);

        DoubleSolution child = children.get(0);

        mutationOperator.execute(child);
        problem.evaluate(child);

        updateNeighborhood(child, subProblemId, neighborType);
      }
      updateExternalPopulation();

      iterations++;
    } while (iterations < maxIterations);

  }

  protected void initializePopulation() {
    for (int i = 0; i < populationSize; i++) {
      DoubleSolution newSolution = (DoubleSolution)problem.createSolution();

      problem.evaluate(newSolution);
      population.add(newSolution);
    }
  }

  @Override public String getName() {
    return "MOEAD" ;
  }

  @Override public String getDescription() {
    return "Multi-Objective Evolutionary Algorithm based on Decomposition" ;
  }
}
