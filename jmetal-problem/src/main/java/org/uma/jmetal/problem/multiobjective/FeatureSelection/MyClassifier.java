package org.uma.jmetal.problem.multiobjective.FeatureSelection;
/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */


import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import libsvm.LibSVM;
import net.sf.javaml.classification.BingKNearestNeighbors;
import net.sf.javaml.classification.Classifier;
import net.sf.javaml.classification.KNearestNeighbors;
import net.sf.javaml.classification.bayes.NaiveBayesClassifier;
import net.sf.javaml.classification.evaluation.PerformanceMeasure;
import net.sf.javaml.classification.tree.RandomTree;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.tools.weka.WekaClassifier;
import weka.classifiers.trees.REPTree;

/**
 *
 * @author xuebing
 */
public class MyClassifier {

    //private NaiveBayesClassifier classifier;
    private net.sf.javaml.classification.Classifier classifier;
    private Random random;
    /*
     */

    public MyClassifier(Random random) {
        this.random = random;
    }

    public void ClassifierKNN() {
        System.out.println("Myclassifier:  new KNearestNeighbors(5)");
        setClassifier(new KNearestNeighbors(5));
    }

    public void ClassifierBingKNN(int k) {
        System.out.println("Myclassifier:  new -- Bing-- KNearestNeighbors(5)");
        setClassifier(new BingKNearestNeighbors(k));
    }

    public void ClassifierNB() {
        setClassifier(new NaiveBayesClassifier(false, true, false));
        System.out.println("Myclassifier:  new NaiveBayesClassifier(false, true, false)");
    }

    public void ClassifierLibSVM() {
        setClassifier(new LibSVM());
        System.out.println("Myclassifier:  new LibSVM()");
    }

    public void ClassifierDT() {
        setClassifier(new WekaClassifier(new REPTree()));
        System.out.println("Myclassifier: new  DT - REPTree()");
    }

    public void ClassifierRT(int noFeatures) {
        setClassifier(new RandomTree(noFeatures, new Random(0)));
        System.out.println("Myclassifier:  new RandomTree(noFeatures, new Random(0))");
    }

    /**
     * @return the classifier
     */
    public Classifier getClassifier() {
        return classifier;
    }

    /**
     * @param classifier the classifier to set
     */
    public void setClassifier(Classifier classifier) {
        this.classifier = classifier;
    }

    public double fullclassify(Dataset training, Dataset testing) {
        double[] features = new double[training.noAttributes()];
        for (int i = 0; i < training.noAttributes(); i++) {
            features[i] = 1.0;
        }
        double acc = 0.0;
        acc = classify(training, testing);
        return acc;
    }

    public double classify(Dataset training, Dataset testing) {
//        Classifier knn = new KNearestNeighbors(5);//        KNearestNeighbors knn = new KNearestNeighbors(5);
        getClassifier().buildClassifier(training);
        Dataset dataForClassification = testing;

        Map<Object, PerformanceMeasure> out = new HashMap<Object, PerformanceMeasure>();
        for (Object o : training.classes()) {
            out.put(o, new PerformanceMeasure());
        }

        for (Instance instance : dataForClassification) {
            Object prediction = getClassifier().classify(instance);
            if (instance.classValue().equals(prediction)) {// prediction                // ==class
                for (Object o : out.keySet()) {
                    if (o.equals(instance.classValue())) {
                        out.get(o).tp++;
                    } else {
                        out.get(o).tn++;
                    }
                }
            } else {// prediction != class
                for (Object o : out.keySet()) {
                    /* prediction is positive class */
                    if (prediction.equals(o)) {
                        out.get(o).fp++;
                    } /* instance is positive class */ else if (o.equals(instance.classValue())) {
                        out.get(o).fn++;
                    } /* none is positive class */ else {
                        out.get(o).tn++;
                    }

                }
            }
        }
//                System.out.println("out====: "+out);
        double tp = 0.0, tn = 0.0;
        double Accuracy = 0.0;
        for (Object o : out.keySet()) {
//           System.out.println(o+" TP: " + p.get(o).tp);
            tp += out.get(o).tp;
            tn += out.get(o).tn;
//            System.out.println((p.get(o).tp+p.get(o).tn)/ data.size());
        }


        Accuracy = (tn + tp) / (double) (out.size() * dataForClassification.size());

//        System.out.println("dataForClassification.size()  " + dataForClassification.size());
//        System.out.println("tn " + tn + "    tp  " + tp + "   Accuracy  " + Accuracy);
        return Accuracy;
    }
}
