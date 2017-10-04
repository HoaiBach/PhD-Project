package Mains;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;

import org.uma.jmetal.algorithm.Algorithm;
import org.uma.jmetal.algorithm.multiobjective.moead.AbstractMOEAD.FunctionType;
import org.uma.jmetal.algorithm.multiobjective.moead.MOEAD;
import org.uma.jmetal.algorithm.multiobjective.moead.MOEADBuilder;
import org.uma.jmetal.algorithm.multiobjective.moead.MOEADBuilder.Variant;
import org.uma.jmetal.algorithm.multiobjective.moead.MOEADDYN;
import org.uma.jmetal.algorithm.multiobjective.moead.MOEADREF;
import org.uma.jmetal.algorithm.multiobjective.nsgaii.NSGAII;
import org.uma.jmetal.algorithm.multiobjective.nsgaii.NSGAIIBuilder;
import org.uma.jmetal.algorithm.multiobjective.nsgaiii.NSGAIII;
import org.uma.jmetal.algorithm.multiobjective.nsgaiii.NSGAIIIBuilder;
import org.uma.jmetal.algorithm.multiobjective.omopso.OMOPSO;
import org.uma.jmetal.algorithm.multiobjective.omopso.OMOPSOBuilder;
import org.uma.jmetal.algorithm.multiobjective.spea2.SPEA2;
import org.uma.jmetal.algorithm.multiobjective.spea2.SPEA2Builder;
import org.uma.jmetal.operator.CrossoverOperator;
import org.uma.jmetal.operator.MutationOperator;
import org.uma.jmetal.operator.impl.crossover.DifferentialEvolutionCrossover;
import org.uma.jmetal.operator.impl.crossover.SBXCrossover;
import org.uma.jmetal.operator.impl.mutation.NonUniformMutation;
import org.uma.jmetal.operator.impl.mutation.PolynomialMutation;
import org.uma.jmetal.operator.impl.mutation.UniformMutation;
import org.uma.jmetal.operator.impl.selection.BinaryTournamentSelection;
import org.uma.jmetal.operator.impl.selection.TournamentSelection;
import org.uma.jmetal.problem.DoubleProblem;
import org.uma.jmetal.problem.Problem;
import org.uma.jmetal.problem.multiobjective.FeatureSelection.FeatureSelection;
import org.uma.jmetal.problem.multiobjective.FeatureSelection.HelpDataset;
import org.uma.jmetal.solution.DoubleSolution;
import org.uma.jmetal.util.AlgorithmRunner;
import org.uma.jmetal.util.evaluator.SolutionListEvaluator;
import org.uma.jmetal.util.evaluator.impl.SequentialSolutionListEvaluator;
import org.uma.jmetal.util.pseudorandom.JMetalRandom;

import Utility.RandomSeed;
import net.sf.javaml.core.Dataset;

public class Main {

	/**
	 * 0: type of algorithm: 1- NSGAII, 2-NSGAIII, 3-SPEA2, 4-OMOPSO, 5-MOEAD(nor), 6-MOEADREF, 7-MOEADDYN
	 * 1: run index
	 * 2: test
	 * @param args
	 * @throws FileNotFoundException
	 */
	public static void main(String[] args) throws FileNotFoundException{
		int type = Integer.parseInt(args[0]);
		int r = Integer.parseInt(args[1]);
		long seeder = (long) (Math.pow(10, 6)*(r+3));
		boolean test = Integer.parseInt(args[2]) > 0? true:false;
		
		JMetalRandom.getInstance().setSeed(seeder);
		RandomSeed.Seeder.setSeed(seeder);

		double threshold = 0.6;
		DoubleProblem problem = new FeatureSelection(RandomSeed.Create(), test);
		((FeatureSelection)problem).setThreshold(threshold);

		Algorithm<List<DoubleSolution>> algorithm = null;

		int popSize = problem.getNumberOfVariables() <= 200 ? problem.getNumberOfVariables():200;
		int maxIterations = 200;
		int T = popSize/10;
		int limit = 4;
		if(T<limit)
			T=limit;
		int nr = 1;

		switch (type) {
		case 1:
		{
			//NSGAII,
			CrossoverOperator<DoubleSolution> crossover
			= new SBXCrossover(0.9, 5);
			MutationOperator<DoubleSolution> mutation
			= new PolynomialMutation(1.0/problem.getNumberOfVariables(),
					20.0);
			TournamentSelection<DoubleSolution> selection = new BinaryTournamentSelection<DoubleSolution>();
			
			algorithm = new NSGAIIBuilder<DoubleSolution>(problem, crossover, mutation)
					.setMaxIterations(maxIterations)
					.setPopulationSize(popSize)
					.setSelectionOperator(selection)
					.build();

			break;
		}
		case 2:{
			//NSGAIII
			CrossoverOperator<DoubleSolution> crossover
			= new SBXCrossover(0.9, 5);
			MutationOperator<DoubleSolution> mutation
			= new PolynomialMutation(1.0/problem.getNumberOfVariables(),
					20.0);
			TournamentSelection<DoubleSolution> selection = new BinaryTournamentSelection<DoubleSolution>();

			algorithm = new NSGAIIIBuilder<DoubleSolution>(problem)
					.setCrossoverOperator(crossover)
					.setMutationOperator(mutation)
					.setMaxIterations(maxIterations)
					.setPopulationSize(popSize)
					.setSelectionOperator(selection)
					.build();
			break;
		}
		case 3:{
			//SPEA2
			CrossoverOperator<DoubleSolution> crossover
			= new SBXCrossover(0.9, 5);
			MutationOperator<DoubleSolution> mutation
			= new PolynomialMutation(1.0/problem.getNumberOfVariables(),
					20.0);
			TournamentSelection<DoubleSolution> selection = new BinaryTournamentSelection<DoubleSolution>();
			
			algorithm = new SPEA2Builder<DoubleSolution>(problem, crossover, mutation)
					.setMaxIterations(maxIterations)
					.setPopulationSize(popSize)
					.setSelectionOperator(selection)
					.build();
			break;
		}
		case 4:{
			//MOPSO
			SolutionListEvaluator<DoubleSolution> evaluator = new SequentialSolutionListEvaluator<DoubleSolution>();
			algorithm = new OMOPSOBuilder(problem, evaluator)
					.setArchiveSize(popSize)
					.setMaxIterations(maxIterations)
					.setSwarmSize(popSize)
					.setNonUniformMutation(new NonUniformMutation(1.0/problem.getNumberOfVariables(), 20.0, maxIterations))
					.setUniformMutation(new UniformMutation(1.0/problem.getNumberOfVariables(), 20.0))
					.build();
			break;
		}
		case 5:{
			//MOEAD-Normal
			DifferentialEvolutionCrossover crossover = new DifferentialEvolutionCrossover(0.6, 0.7, "rand/1/bin");
			MutationOperator<DoubleSolution> mutation = new PolynomialMutation(1.0/problem.getNumberOfVariables(),
					20.0);
			((FeatureSelection) problem).setCurrentFeatureRate(0.0);
			algorithm = new MOEADBuilder(problem,Variant.MOEAD)
					.setMaxIterations(maxIterations)
					.setPopulationSize(popSize)
					.setResultPopulationSize(popSize)
					.setNeighborSize(T)
					.setMaximumNumberOfReplacedSolutions(nr)
					.setCrossover(crossover)
					.setMutation(mutation)
					.setNumberOfThreads(4)
					.setNeighborhoodSelectionProbability(0.85)
					.setFunctionType(FunctionType.TCHE)
					.build();
			((MOEAD)algorithm).setReferencePoint(new double[]{0.0,0.0});
			((MOEAD)algorithm).setNadirPoint(new double[]{1.0,1.0});
			break;
		}
		case 6:{
			//MOEAD-multiRefs
			DifferentialEvolutionCrossover crossover = new DifferentialEvolutionCrossover(0.6, 0.7, "rand/1/bin");
			MutationOperator<DoubleSolution> mutation = new PolynomialMutation(1.0/problem.getNumberOfVariables(),
					20.0);
			((FeatureSelection) problem).setCurrentFeatureRate(0.0);
			algorithm = new MOEADBuilder(problem,Variant.MOEADREF)
					.setMaxIterations(maxIterations)
					.setPopulationSize(popSize)
					.setResultPopulationSize(popSize)
					.setNeighborSize(T)
					.setMaximumNumberOfReplacedSolutions(nr)
					.setCrossover(crossover)
					.setMutation(mutation)
					.setNumberOfThreads(4)
					.setNeighborhoodSelectionProbability(0.85)
					.build();
			((MOEADREF)algorithm).setWeight(1.0);
			((MOEADREF)algorithm).setReferencePoint(new double[]{0.0,0.0});
			((MOEADREF)algorithm).setNadirPoint(new double[]{1.0,1.0});
			break;
		}
		case 7:{
			//MOEAD-DYN
			double rate = 0.6;
			int noIntervals;
			if(problem.getNumberOfVariables()<20)
				noIntervals = 9;
			else
				noIntervals = 4;	
			if(rate*problem.getNumberOfVariables() < noIntervals){
				noIntervals = (int)(rate*problem.getNumberOfVariables());
			}
			//using sequential accuracy

			DifferentialEvolutionCrossover crossover = new DifferentialEvolutionCrossover(0.6, 0.7, "rand/1/bin");
			MutationOperator<DoubleSolution> mutation = new PolynomialMutation(1.0/problem.getNumberOfVariables(),
					20.0);
			((FeatureSelection) problem).setCurrentFeatureRate(0.0);
			algorithm = new MOEADBuilder(problem,Variant.MOEADDYN)
					.setMaxIterations(maxIterations)
					.setPopulationSize(popSize)
					.setResultPopulationSize(popSize)
					.setNeighborSize(T)
					.setMaximumNumberOfReplacedSolutions(nr)
					.setCrossover(crossover)
					.setMutation(mutation)
					.setNumberOfThreads(4)
					.setNeighborhoodSelectionProbability(0.85)
					.build();
			((MOEADDYN)algorithm).setWeight(1.0);
			((MOEADDYN)algorithm).initializeDynamic(rate, noIntervals);
			((MOEADDYN)algorithm).setRandom(RandomSeed.Create());
			((MOEADDYN)algorithm).setThreshold(threshold);
			((MOEADDYN)algorithm).setReferencePoint(new double[]{0.0,0.0});
			((MOEADDYN)algorithm).setNadirPoint(new double[]{1.0,1.0});
			break;
		}

		default:
			try {
				throw new Exception("Invalid algorithm index!!!");
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		run(algorithm,r,problem,type);

	}

	public static void run(Algorithm<List<DoubleSolution>> alg, int runIndex, Problem<DoubleSolution> p,int type) throws FileNotFoundException{
		AlgorithmRunner algorithmRunner = new AlgorithmRunner.Executor(alg).execute();
		List<DoubleSolution> pop = alg.getResult();
		long computingTime = algorithmRunner.getComputingTime();
		FeatureSelection fs = (FeatureSelection) p;

		PrintStream pt = new PrintStream(new File("FUN"+runIndex));
		PrintStream pt1 = new PrintStream(new File("VAR"+runIndex));
		PrintStream pt2 = new PrintStream(new File("Time"+runIndex));

		List<List<DoubleSolution>> records = null;
		switch (type) {
		case 1:
		{
			records = ((NSGAII<DoubleSolution>)alg).getRecord();
			break;
		}
		case 2:
		{
			records = ((NSGAIII<DoubleSolution>)alg).getRecord();
			break;
		}
		case 3:
		{
			records = ((SPEA2<DoubleSolution>)alg).getRecord();
			break;
		}
		case 4:
		{
			records = ((OMOPSO)alg).getRecord();
			break;
		}
		case 5:
		{
			records = ((MOEAD)alg).getRecord();
			break;
		}
		case 6:{
			records = ((MOEADREF)alg).getRecord();
			break;
		}
		case 7:{
			records = ((MOEADDYN)alg).getRecord();
			break;
		}
		default:{
			try {
				throw new Exception("Invalid algorithm index!!!");
			} catch (Exception e) {
				e.printStackTrace();
			}
			break;
		}
		}

		int index=1;
		for(List<DoubleSolution> sols: records){
			pt.println("========"+index+"=======");
			for(DoubleSolution sol: sols){
				pt.println(sol.getObjective(0)+","+sol.getObjective(1));
			}
			index++;
		}
		pt.println("========Final Solution=========");
		Map<Double,Double> nfTrAcc = new TreeMap<Double,Double>();
		Map<Double,Double> nfTeAcc = new TreeMap<Double,Double>();
		for(DoubleSolution solution: pop){
			if(!nfTrAcc.containsKey(solution.getObjective(0))){
				double[] features = fs.solutionToBits(solution);
				Dataset tempTrain = fs.getTraining().copy();
				Dataset tempTest = fs.getTesting().copy();
				tempTrain = HelpDataset.removeFeatures(tempTrain, features);
				tempTest = HelpDataset.removeFeatures(tempTest, features);
				double testError = 1- fs.getClassifier().classify(tempTrain, tempTest);
				nfTrAcc.put(solution.getObjective(0), solution.getObjective(1));
				nfTeAcc.put(solution.getObjective(0), testError);
			}
			for(int i=0;i<solution.getNumberOfVariables()-1;i++)
				pt1.print(solution.getVariableValue(i)+",");
			pt1.println(solution.getVariableValue(solution.getNumberOfVariables()-1));
		}
		
		for(Entry<Double,Double> entry: nfTrAcc.entrySet()){
			double nf = entry.getKey();
			double tr = entry.getValue();
			double te = nfTeAcc.get(nf);
			pt.println(nf+", "+tr+", "+te);
		}
		
		pt2.println(computingTime);
		pt.close();
		pt1.close();
		pt2.close();
	}
}
