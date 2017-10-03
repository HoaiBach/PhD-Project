/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package ExtractResults;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.Math.*;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

/**
 *
 * @author xuebing
 */
public class Helper {

    public static double[] toFm(double[] accFCOnly) {

        double[] accFm = new double[accFCOnly.length];
        DecimalFormat df = new DecimalFormat("##.##");

        accFCOnly = Helper.toPercentArray(accFCOnly);
        for (int i = 0; i < accFCOnly.length; i++) {
            accFm[i] = Double.valueOf(df.format(accFCOnly[i]));
        }
        return accFm;
    }

    public static void printArray(String[] acc, String filename) throws IOException {
        File file = new File(filename);
        BufferedWriter bw = new BufferedWriter(new FileWriter(file));

        if (!file.exists()) {
            file.createNewFile();
            System.out.println("already there-repeat: " + filename);
        }
        for (int i = 0; i < acc.length; i++) {
            bw.write(acc[i]);
            bw.newLine();
        }
        bw.close();
    }


    public static void printList(ArrayList<String> acc, String filename) throws IOException {
        File file = new File(filename);
        BufferedWriter bw = new BufferedWriter(new FileWriter(file));

        if (!file.exists()) {
            file.createNewFile();
            System.out.println("already there-repeat: " + filename);
        }
        for (int i = 0; i < acc.size(); i++) {
            bw.write(acc.get(i));
            bw.newLine();
        }
        bw.close();
    }

    public static void printArray2(double[] acc, String filename) throws IOException {
        File file = new File(filename);
        BufferedWriter bw = new BufferedWriter(new FileWriter(file));

        if (!file.exists()) {
            file.createNewFile();
            System.out.println("already there-repeat: " + filename);
        }
        for (int i = 0; i < acc.length; i++) {
            bw.write(String.valueOf(acc[i]));
            bw.newLine();
        }
        bw.close();
    }

    public static double[] sameEleArray(int n, double value) {
        double[] array = new double[n];
        for (int i = 0; i < n; i++) {
            array[i] = value;
        }
        return array;
    }

    public static double Scale(double src_min, double src_max, double value,
            double target_min, double target_max) {
        return value / ((target_max - target_min) / (src_max - src_min)) + target_min;
    }

    public static double[] AverageRunIterations(double ARI[][]) {
        double results[] = new double[ARI[0].length];
        for (int i = 0; i < ARI[0].length; ++i) {
            for (int r = 0; r < ARI.length; ++r) {
                results[i] += ARI[r][i];
            }
            results[i] = results[i] / ARI.length;
        }
        return results;
    }

// get getMean
    public static double getMean(double[] array) {
        double sum = 0.0;  // sum of all the elements
        if (array.length == 0) {
            System.out.println("This array with no elements in it !!!!");
        } else {
            for (int i = 0; i < array.length; i++) {
                sum += array[i];
            }
        }
        return sum / array.length;
    }//end method getMean
    
    public static double getMean(int[] array) {
        double sum = 0.0;  // sum of all the elements
        if (array.length == 0) {
            System.out.println("This array with no elements in it !!!!");
        } else {
            for (int i = 0; i < array.length; i++) {
                sum += array[i];
            }
        }
        return sum / array.length;
    }

    // get getMean and Standard Deviation
    public static double[] Mean_STD(double BestFitness[]) {
        double Arrays[] = BestFitness;
        double M_STD[] = new double[2];
        double sum = 0.0;
        for (int i = 0; i < BestFitness.length; ++i) {
            sum += BestFitness[i];
        }
        M_STD[0] = sum / (double) BestFitness.length;

        M_STD[1] = get_STD(Arrays, M_STD[0]);
        return M_STD;
    }

    public static double get_STD(double Arrays[], double Mean) {
        double allSquare = 0.0;
        for (Double Array : Arrays) {
            allSquare += (Array - Mean) * (Array - Mean);
        }
        
        double denominator = Arrays.length;
        return Math.sqrt(allSquare / denominator);
    }
    
    public static double getStD(int[] arrays){
    	double[] a = new double[arrays.length];
    	for(int i=0;i<arrays.length;i++){
    		a[i] = arrays[i];
    	}
    	return getStD(a);
    }

    public static double getStD(double Arrays[]) {
        double allSquare = 0.0;
        double Mean = getMean(Arrays);
        for (Double Array : Arrays) {
            allSquare += (Array - Mean) * (Array - Mean);
        }
        double denominator = Arrays.length;
        return Math.sqrt(allSquare / denominator);
    }

    public static int ModEuclidean(int D, int d) {
        int r = D % d;
        if (r < 0) {
            if (d > 0) {
                r = r + d;
            } else {

                r = r - d;
            }
        }
        return r;
    }

    public static double[][] featureranking(double distance1[], double distance2[]) {
        double[] d3 = new double[distance1.length];
        double[][] FR = new double[2][distance1.length];
        double tem;

        for (int i = distance1.length - 1; i > 0; i--) {
            for (int j = 0; j < i; j++) {
                if (distance1[j] < distance1[j + 1]) {
                    tem = distance1[j];
                    distance1[j] = distance1[j + 1];
                    distance1[j + 1] = tem;
                }

            }
        }

        for (int i = 0; i < distance2.length; i++) {
            for (int j = 0; j < distance1.length; j++) {
                if (distance1[i] == distance2[j]) {
                    d3[i] = j + 1;
                    break;
                }
            }
        }
        FR[1] = distance1;
        FR[2] = d3;
        return FR;
    }

    // prodce t different int numbers within 0-n
    public static int[] fn(int t, int n) {
        Set<Integer> set = new HashSet<Integer>();
        int length = 0;
        while ((length = set.size()) < t) {
            int raNum = (int) (RandomBing.Create().nextDouble() * n);
            set.add(raNum);
        }
        int[] array = new int[t];
        Integer[] intArray = new Integer[t];
        intArray = set.toArray(intArray);
        for (int i = 0; i < t; i++) {
            array[i] = intArray[i].intValue();
//            System.out.printArray(array[i]+ "  ");
//            System.out.println("  ");

        }
        return array;
    }

    /*Divid the datasets, select test from two sides,
    select the first fold, last fold, the third fold, and the third last fold as test set; 0 9 2,   0 9 2 7 4,  0 9 2 7 4 5;
     */
    public static int[] divData(int numTest, int numTrain) {
        int[] select = new int[numTest];
        int tem1 = 0;
        for (int i = 0; i < numTest; i = i + 2) {
            select[tem1] = i;
            if (tem1 < numTest - 1) {
                select[tem1 + 1] = numTrain + numTest - 1 - i;
                tem1 = tem1 + 2;
            } else {
                break;
            }
        }
        return select;
    }

    public static double[] toPercentArray(double[] accTest) {
        if (accTest[0] < 1.1) {
            for (int i = 0; i < accTest.length; i++) {
                accTest[i] = Double.valueOf(100 * accTest[i]);
            }
        }
        return accTest;
    }

    public static double toPercent(double accTest) {
        if (accTest < 1.1) {
            accTest = Double.valueOf(100 * accTest);
        }
        return accTest;
    }

    public static double[] toEr2Acc(double[] accEr) {
        double total = (accEr[0] > 1) ? 100 : 1;
        for (int i = 0; i < accEr.length; i++) {
            accEr[i] = total - accEr[i];
        }
        return accEr;
    }

    //        int[] select = Helper.divData(numTest, numTrain);
//        int tem = 0;
//        for (int i = 0; i < (numTrain + numTest); i++) {
//            if ((tem < numTest) && (i == select[tem])) {
//                testing.addAll(folds[i]);
//                tem++;
//            } else {
//                training.addAll(folds[i]);
//            }
//        }
    // get getMean and Standard Deviation
    
    public static double[] mean_STD(double BestFitness[]) {
        double Arrays[] = BestFitness;
        double M_STD[] = new double[2];
        double sum = 0.0;
        for (int i = 0; i < BestFitness.length; ++i) {
            sum += BestFitness[i];
        }
        M_STD[0] = sum / (double) BestFitness.length;
        M_STD[1] = get_STD(Arrays, M_STD[0]);
        return M_STD;
    }

    public static double getMaxValue(double[] numbers) {
        double maxValue = numbers[0];
        for (int i = 1; i < numbers.length; i++) {
            if (numbers[i] > maxValue) {
                maxValue = numbers[i];
//                System.out.println("maxValue"+maxValue);
            }
        }
        return maxValue;
    }

    public static int getMaxValue(int[] numbers) {
        int maxValue = numbers[0];
        for (int i = 1; i < numbers.length; i++) {
            if (numbers[i] > maxValue) {
                maxValue = numbers[i];
//                System.out.println("maxValue"+maxValue);
            }
        }
        return maxValue;
    }
    
    public static double getMinValue(double[] numbers) {
        double minValue = numbers[0];
        for (int i = 1; i < numbers.length; i++) {
            if (numbers[i] < minValue) {
                minValue = numbers[i];
            }
        }
        return minValue;
    }
    
    public static double getMinValue(int[] numbers) {
        int minValue = numbers[0];
        for (int i = 1; i < numbers.length; i++) {
            if (numbers[i] < minValue) {
                minValue = numbers[i];
            }
        }
        return minValue;
    }

    public static double[] aveValue2Dim(double mineSize[][]) {
        double aveSize[] = new double[mineSize.length];
        for (int i = 0; i < mineSize.length; ++i) {
            for (int r = 0; r < mineSize[0].length; ++r) {
                aveSize[i] += mineSize[i][r];
            }
            aveSize[i] = aveSize[i] / mineSize[0].length;
        }
        return aveSize;
    }

    public static boolean ifSameEleArray(double[] acc) {
        boolean same = true;
        double ini = acc[0];
        for (int i = 0; i < acc.length; i++) {
            if (!(ini == acc[i])) {
                same = false;
                break;
            }
        }
        return same;
    }
}
//   int n = 10;
//        int nei = 4;
//
//        for (int j = 0; j < 10; ++j) {
//            System.out.printArray(j + ":");
//
//            for (int p =-nei/2; p <= nei/2 ; ++p) {
//
//                System.out.printArray(Math.ModEuclidean(p + j, n) + " ; ");
//            }
//            System.out.println();
//        }
//        System.out.printArray((1/2)*nei);
//        System.exit(32);
//0:8 ; 9 ; 0 ; 1 ; 2 ;
//1:9 ; 0 ; 1 ; 2 ; 3 ;
//2:0 ; 1 ; 2 ; 3 ; 4 ;
//3:1 ; 2 ; 3 ; 4 ; 5 ;
//4:2 ; 3 ; 4 ; 5 ; 6 ;
//5:3 ; 4 ; 5 ; 6 ; 7 ;
//6:4 ; 5 ; 6 ; 7 ; 8 ;
//7:5 ; 6 ; 7 ; 8 ; 9 ;
//8:6 ; 7 ; 8 ; 9 ; 0 ;
//9:7 ; 8 ; 9 ; 0 ; 1 ;

