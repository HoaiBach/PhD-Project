package Utility;
/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */


//import java.util.List;
//import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

/**
 *
 * @author xuebing
 */
public class NewMath {

    public static double Scale(double src_min, double src_max, double value, double target_min, double target_max) {

        if (src_max == src_min) {
            System.out.println("src_max == src_min ");
            return (1 / 2) * (target_max - target_min) + target_min;
        } else {
            return  (target_max - target_min)*(value -src_min) / (src_max - src_min) + target_min;
        }
    }

//    public static double Averageitera(Swarm s) {
//        double result = 0.0;
//        for (int i = 0; i < s.numberOfParticles(); ++i) {
//            result += s.getParticle(i).getFitness();
//        }
//        return result / s.numberOfParticles();
//    }

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

// get mean
    public static double mean(double[] array) {
        double sum = 0.0;  // sum of all the elements
        if (array.length == 0) {
            System.out.println("This array with no elements in it !!!!");
        } else {
            for (int i = 0; i < array.length; i++) {
                sum += array[i];
            }
        }
        return sum / array.length;
    }//end method mean

    // get mean and Standard Deviation
    public static double[] Mean_STD(double BestFitness[]) {
        double Arrays[] = BestFitness;
        double M_STD[] = new double[2];
        double sum = 0.0;
        for (int i = 0; i < BestFitness.length; ++i) {
            sum += BestFitness[i];
        }
        M_STD[0] = sum / (double) BestFitness.length;

        M_STD[1] = calculateSTD(Arrays, M_STD[0]);
        return M_STD;
    }

    public static double calculateSTD(double Arrays[], double Mean) {
        double allSquare = 0.0;
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
            int raNum = (int) (RandomSeed.Create().nextDouble() * n);
            set.add(raNum);
        }
        int[] array = new int[t];
        Integer[] intArray = new Integer[t];
        intArray = set.toArray(intArray);
        for (int i = 0; i < t; i++) {
            array[i] = intArray[i].intValue();
//            System.out.print(array[i]+ "  ");
//            System.out.println("  ");

        }
        return array;
    }

    public static double round4(double value){
    	return ((int)(value*10000+0.5))/10000.0;
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
    //        int[] select = NewMath.divData(numTest, numTrain);
//        int tem = 0;
//        for (int i = 0; i < (numTrain + numTest); i++) {
//            if ((tem < numTest) && (i == select[tem])) {
//                testing.addAll(folds[i]);
//                tem++;
//            } else {
//                training.addAll(folds[i]);
//            }
//        }
}
//   int n = 10;
//        int nei = 4;
//
//        for (int j = 0; j < 10; ++j) {
//            System.out.print(j + ":");
//
//            for (int p =-nei/2; p <= nei/2 ; ++p) {
//
//                System.out.print(Math.ModEuclidean(p + j, n) + " ; ");
//            }
//            System.out.println();
//        }
//        System.out.print((1/2)*nei);
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

