package org.uma.jmetal.problem.multiobjective.FeatureSelection;
/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */


import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.DenseInstance;
import net.sf.javaml.core.Instance;
//import java.util.*;
//import java.lang.Math;

/**
 *
 * @author xuebing
 */
public class ReadResults {

    public static String read2ndLine(String filename) throws IOException {
        String line = null;
        try {
            BufferedReader bufferedReader = new BufferedReader(new FileReader(filename));

            line = bufferedReader.readLine();
            line = bufferedReader.readLine();

            bufferedReader.close();
//            in.close();
        } catch (Exception e) {//Catch exception if any
            e.printStackTrace();
        }
        return line;
    }

    public static String[] readfromNline(String filename, int startline, int noLins) throws IOException {
        String line = null;
        String[] strings = new String[noLins];
        try {
            BufferedReader bufferedReader = new BufferedReader(new FileReader(filename));

            for (int i = 0; i < startline; i++) {
                line = bufferedReader.readLine();
            }

            int i = 0;
            while ((line = bufferedReader.readLine()) != null && i < noLins) {
                strings[i] = line;
                i++;
            }

            line = bufferedReader.readLine();

            bufferedReader.close();
//            in.close();
        } catch (Exception e) {//Catch exception if any
            e.printStackTrace();
        }
        return strings;
    }

    public static String read1Line(String filename) throws IOException {
        String line = null;
        try {
            BufferedReader bufferedReader = new BufferedReader(new FileReader(filename));

            line = bufferedReader.readLine();

            bufferedReader.close();
//            in.close();
        } catch (Exception e) {//Catch exception if any
            e.printStackTrace();
        }
        return line;
    }

    public static String[] read2array(String filename, String[] strTem) throws IOException {
        try {
            BufferedReader bufferedReader = new BufferedReader(new FileReader(filename));
            int i = 0;
            String line = null;
            while ((line = bufferedReader.readLine()) != null && i < strTem.length) {
                strTem[i] = line;
                i++;
            }
            bufferedReader.close();
//            in.close();
        } catch (Exception e) {//Catch exception if any
            e.printStackTrace();
        }
        return strTem;
    }

    public static double[] readArray2Double(String filename, int n) throws IOException {
        double[] acc = new double[n];

        String[] strTem = new String[n];

        strTem = read2array(filename, strTem);

        acc = str2doubleArray(strTem);

        return acc;
    }


    public static ArrayList<String> readAll2list(String filename) throws IOException {
        ArrayList<String> tem = new ArrayList<String>();
        try {
            BufferedReader bufferedReader = new BufferedReader(new FileReader(filename));
            int i = 0;
            String line = null;
            while ((line = bufferedReader.readLine()) != null) {
                tem.add(line);
                i++;
            }
            bufferedReader.close();
//            in.close();
        } catch (Exception e) {//Catch exception if any
            e.printStackTrace();
        }
        return tem;
    }

    public static String[] readSomLine2array(String filename, String[] strTem, String search4) throws IOException {
        try {
            BufferedReader r = new BufferedReader(new FileReader(filename));
            String line = null;
            int i = 0;

            for (int j = 0; j < 1000; j++) {
                line = r.readLine();
                if (line.equals(search4)) {
                    System.out.println("yes here !!!");
                    break;
                }
                if (j == 999) {
                    System.out.println("Something wrong, could not find the target line");
                }
            }

            while ((line = r.readLine()) != null && i < strTem.length) {
                strTem[i] = line;
                i++;
            }
            r.close();
        } catch (Exception e) {//Catch exception if any
            e.printStackTrace();
        }
        return strTem;
    }

    /*Read the lines, including the searched words (search4), start from the current line
     */
    public static String[] readCurLines2array(String filename, String[] strTem, String search4) throws IOException {
        try {
            BufferedReader r = new BufferedReader(new FileReader(filename));
            String line = null;
            int i = 0;

            for (int j = 0; j < 1000; j++) {
                line = r.readLine();
                if (line != null && line.length() > search4.length()) {
                    String linestart = line.substring(0, (search4.length()));
                    if (linestart.equals(search4)) {
//                        System.out.println("yes here !!!");
                        strTem[i] = line;
                        i++;
                        break;
                    }
                }

                if (j == 999) {
                    System.out.println("Something wrong, could not find the target line");
                }
            }

            while ((line = r.readLine()) != null && i < strTem.length) {
                strTem[i] = line;
                i++;
            }
            r.close();
        } catch (Exception e) {//Catch exception if any
            e.printStackTrace();
        }
        return strTem;
    }

    public static String readOneCertainLine(String filename, int lineNO) throws IOException {
        String line = null;
        try {
            BufferedReader bufferedReader = new BufferedReader(new FileReader(filename));


            for (int i = 0; i < lineNO - 1; i++) {
                line = bufferedReader.readLine();
            }

            line = bufferedReader.readLine();

            bufferedReader.close();
//            in.close();
        } catch (Exception e) {//Catch exception if any
            e.printStackTrace();
        }
        return line;
    }

    public static String read1stLine(String filename) throws IOException {
        String line = null;
        try {
            BufferedReader bufferedReader = new BufferedReader(new FileReader(filename));

            line = bufferedReader.readLine();

            bufferedReader.close();
//            in.close();
        } catch (Exception e) {//Catch exception if any
            e.printStackTrace();
        }
        return line;
    }

    public static double[] str2doubleArray(String[] array) {
        double[] newArray = new double[array.length];
        for (int i = 0; i < array.length; i++) {
//            System.out.println(array[i]);
            newArray[i] = Double.valueOf(array[i]);
        }
        return newArray;
    }

    public static int sizeSubset(double[] features) {
        int size = 0;
        for (int i = 0; i < features.length; i++) {
            if (features[i] >= 0.6) {
                size++;
            }  // this is suitable for both binary and continouse results, because even binay, 1>0.6,
        }


        return size;
    }

    /*
     * Removes the element from the vector that is farthest from the supplied
     * element.
     */
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

    public static Dataset[] splitTrTe(Dataset data) {
        Dataset[] folds = data.folds((10), new Random(100));
        Dataset training = new DefaultDataset();
        Dataset testing = new DefaultDataset();

        int[] tr = {0, 2, 3, 5, 6, 8, 9};
        int[] te = {1, 4, 7};   // 7, 4 and 6,5 changes
        for (int i = 0; i < tr.length; i++) {
            training.addAll(folds[tr[i]]);
        }
        for (int i = 0; i < te.length; i++) {
            testing.addAll(folds[te[i]]);
        }
        Dataset[] aa = new Dataset[2];
        aa[0] = training;
        aa[1] = testing;

        System.out.println("e.g. Zoo: Dataset[] folds = data.folds((10), new Random(100));");

        System.out.println("");
        return aa;
    }
    /*
     *separate the whole dataset to training and testing;
     */

    public static Dataset[] SeparateDataset(Dataset data) {
        Dataset[] trainTest = new Dataset[2];

        int[] tr = {0, 2, 3, 5, 6, 8, 9};
        int[] te = {1, 4, 7};   // 7, 4 and 6,5 changes
        Dataset[] folds = data.folds(10, new Random(0));
        Dataset train = new DefaultDataset();
        Dataset test = new DefaultDataset();
        for (int i = 0; i < tr.length; i++) {
            train.addAll(folds[tr[i]]);
        }
        trainTest[0] = train;
        for (int i = 0; i < te.length; i++) {
            test.addAll(folds[te[i]]);
        }
        trainTest[1] = test;

        System.out.println("e.g. Spect: Dataset[] folds = data.folds(10, new Random(0));");
        //"chess/", "connect4/", "dermatology/", "leddisplay/", "lymph/", "mushroom/", "soybeanlarge/", "spect/", "splice/", "statlog/", "waveform/"

        return trainTest;
    }

    public static Dataset[] trainTestWrapper(Dataset data) {

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

        Dataset[] trTest = {training, testing};

        return trTest;
    }

    public static double[] fullSet(int Nofeatures) {
        double[] fullSet = new double[Nofeatures];
        for (int i = 0; i < Nofeatures; i++) {
            fullSet[0] = 1.0;
        }
        return fullSet;
    }

    public static double[] str2DoubleArray(String[] array) {
        double[] newArray = new double[array.length];
        for (int i = 0; i < array.length; i++) {
//            System.out.println(array[i]);
            newArray[i] = Double.valueOf(array[i]);
        }
        return newArray;
    }

    public static String[] Double2StrArray(double[] array) {
        String[] newArray = new String[array.length];
        for (int i = 0; i < array.length; i++) {
//            System.out.println(array[i]);
            newArray[i] = String.valueOf(array[i]);
        }
        return newArray;
    }

    public static double[] index2Sub(List index, int noFea) {
        double[] features = new double[noFea];

        for (int i = 0; i < noFea; i++) {
            features[i] = 0.0;
        }
//        System.out.println("index.size()  "+index.size());
        for (int i = 0; i < index.size(); i++) {
            int ind = Integer.parseInt(index.get(i).toString());
            features[ind - 1] = 1.0;
        }

        return features;
    }

    public static double[] index02Sub(List index, int noFea) {
        double[] features = new double[noFea];

        for (int i = 0; i < noFea; i++) {
            features[i] = 0.0;
        }
//        System.out.println("index.size()  "+index.size());
        for (int i = 0; i < index.size(); i++) {
            int ind = Integer.parseInt(index.get(i).toString());
//            System.out.println("ind "+ind);
            features[ind] = 1.0;
        }

        return features;
    }

    public static double[] fullFeatures(int noFeatures) {
        double[] allFeatures = new double[noFeatures];
        for (int i = 0; i < allFeatures.length; i++) {
            allFeatures[i] = 1.0;
        }
        return allFeatures;
    }

    public static double[][] rankFea(HashMap AccFea) {
        double maxacc = Double.parseDouble(AccFea.get(1).toString());

        double[] acc = new double[AccFea.size()];
        for (int i = 0; i < AccFea.size(); i++) {
            acc[i] = Double.parseDouble(AccFea.get(i + 1).toString());
        }
        Arrays.sort(acc);

        int ind = 0;
        double[][] accIndeRank = new double[2][AccFea.size()];  // start from 1 to noFeatures

        for (int i = acc.length - 1; i > -1; i--) {
            for (int j = 1; j < AccFea.size() + 1; j++) {
//                System.out.println(j);
                if (acc[i] == Double.parseDouble(AccFea.get(j).toString())) {
                    accIndeRank[0][ind] = acc[i];
                    accIndeRank[1][ind] = j;
//                    System.out.println((int) accIndeRank[1][ind] + "    " + accIndeRank[0][ind]);
                    ind++;
//                    System.out.println(ind);
                    AccFea.put(j, 100);
                }
            }
        }
        return accIndeRank;
    }
}
//
//        int sum = 0;
//        for (int i = 0; i < position.size(); ++i) {
//            sum += position.get(i);
//        }
//        if (sum == N) {
//            for (int i = 0; i < position.size(); ++i) {
//                if (position.get(i) == 1) {
//                    Instance attributeInstance = DatasetTools.createInstanceFromAttribute(data, i);
//                    fitness += dm.measure(attributeInstance, classInstance);
//                }
//            }
//
//
//        } else {
//            fitness = getWorstFitness();  //penalty when select more than N attributs
//        }
//            }
//
//
// if (sum == N) {
//            for (int i = 0; i < position.size(); ++i) {
//                Instance attributeInstance = DatasetTools.createInstanceFromAttribute(data, i);
//                double score = position.get(i) * Math.abs(dm.measure(attributeInstance, classInstance));
//
//                if (score > fitness) {
//                   fitness = score;
//               }
//            }
//            return fitness;
//        } else {
//          return -10;

