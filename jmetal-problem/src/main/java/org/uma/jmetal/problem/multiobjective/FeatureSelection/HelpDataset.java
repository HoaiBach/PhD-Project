package org.uma.jmetal.problem.multiobjective.FeatureSelection;

import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.DenseInstance;
import net.sf.javaml.core.Instance;

public class HelpDataset {

    public static Dataset removeFeatures(Dataset data, double[] features) {
//
        Dataset n = new DefaultDataset();
        int count = 0;
        for (double d : features) {
            if (d == 1.0) {
                count++;
            }
        }

        for (Instance instance : data) {
//            System.out.println("features.length  " + instance.noAttributes());
            double[] f = new double[count];
            int j = 0;
//            System.out.println("features.length  " + features.length);
            for (int i = 0; i < features.length; i++) {
                if (features[i] != 0) {
                    f[j++] = instance.get(i);
                }
            }

            Instance inst = new DenseInstance(f, instance.classValue());
            n.add(inst);
        }

        return n;
    }
}
