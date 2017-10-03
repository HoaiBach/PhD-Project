/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package ExtractResults;

import java.util.Arrays;

import jsc.datastructures.PairedData;
import jsc.onesample.WilcoxonTest;
import jsc.tests.H1;


public class WilsonTestBing {

    /* pValue ===  the probility of being wrong when reject null hypothesis.
     * pValue === the probality of null hypothesis is right
     * pValue ---smaller the better, the samller the more change alternative hypothesis is right
     * H1 is what I want
     * threeTest is to compare accuracy, if error rate, change the strTest value + - ;
     * higher -> +, smaller -> -
     */
    public static String TestBingNew(double[] enemy, double[] mine) {
        double pValue = 0.0;
//        double sigLevel = 0.05;
        double sigLevel = 0.05;
        String strTest = null;

        Boolean same = true;

        for (int i = 0; i < enemy.length; i++) {
            if (enemy[i] != mine[i]) {
                same=false;
                break;
            }
        }

        if (same == true) {
           return "=";
        }


        if (Helper.ifSameEleArray(enemy) & Helper.ifSameEleArray(mine)) {
            if (enemy[0] == mine[0]) {
                strTest = "=";
            } else if (enemy[0] < mine[0]) {
                strTest = "+";
            } else {
                strTest = "-";
            }
        } else {
            PairedData data = new PairedData(enemy, mine);
            WilcoxonTest w0 = new WilcoxonTest(data, H1.NOT_EQUAL);

            if (w0.getSP() > sigLevel) {
                strTest = "=";
                pValue = w0.getSP();
//            System.out.println("w0.getSP()" + w0.getSP());
            } else {

                WilcoxonTest w1 = new WilcoxonTest(data, H1.LESS_THAN);
                if (w1.getSP() <= sigLevel) {
                    strTest = "+";
                    pValue = w1.getSP();
//                System.out.println("w1.getSP()" + w1.getSP());
                } else {

                    WilcoxonTest w2 = new WilcoxonTest(data, H1.GREATER_THAN);
                    if (w2.getSP() < sigLevel) {
                        strTest = "-";
                        pValue = w2.getSP();
//                    System.out.println("w2.getSP()" + w2.getSP());
                    } else {
                        strTest = "!!!!!!!";
                    }
                }
            }
        }
//        return pValue;
        return strTest;
    }

    public static String TestBingMax(int[] enemy, int[] profit){
    	double[] a = new double[enemy.length];
    	double[] b = new double[profit.length];
    	for(int i=0;i<enemy.length;i++){
    		a[i] = enemy[i];
    		b[i] = profit[i];
    	}
    	return TestBingMax(a,b);
    }

    public static String TestBingMax(double[] enemy, double[] mine) {
        double pValue = 0.0;
//        double sigLevel = 0.05;
        double sigLevel = 0.05;
        String strTest = null;

        Boolean same = true;

        for (int i = 0; i < enemy.length; i++) {
            if (enemy[i] != mine[i]) {
                same=false;
                break;
            }
        }

        if (same == true) {
           return "=";
        }


        if (Helper.ifSameEleArray(enemy) & Helper.ifSameEleArray(mine)) {
            if (enemy[0] == mine[0]) {
                strTest = "=";
            } else if (enemy[0] < mine[0]) {
                strTest = "-";
            } else {
                strTest = "+";
            }
        } else {
            PairedData data = new PairedData(enemy, mine);
            WilcoxonTest w0 = new WilcoxonTest(data, H1.NOT_EQUAL);

            if (w0.getSP() > sigLevel) {
                strTest = "=";
                pValue = w0.getSP();
//            System.out.println("w0.getSP()" + w0.getSP());
            } else {

                WilcoxonTest w1 = new WilcoxonTest(data, H1.LESS_THAN);
                if (w1.getSP() <= sigLevel) {
                    strTest = "-";
                    pValue = w1.getSP();
//                System.out.println("w1.getSP()" + w1.getSP());
                } else {

                    WilcoxonTest w2 = new WilcoxonTest(data, H1.GREATER_THAN);
                    if (w2.getSP() < sigLevel) {
                        strTest = "+";
                        pValue = w2.getSP();
//                    System.out.println("w2.getSP()" + w2.getSP());
                    } else {
                        strTest = "!!!!!!!";
                    }
                }
            }
        }
//        return pValue;
        return strTest;
    }

//    From Su
    private static double isSignificantLessStatTest(double[] s1, double[] s2) throws IllegalArgumentException {
//        if (getMean(s1) == getMean(s2)) {
//            return 1;
//        }
        PairedData data = new PairedData(s1, s2);
        WilcoxonTest w = new WilcoxonTest(data, H1.LESS_THAN);
        return w.getSP();

    }
}
