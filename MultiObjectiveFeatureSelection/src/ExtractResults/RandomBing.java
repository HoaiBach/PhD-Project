package ExtractResults;
/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */



import java.util.Random;

/**
 *
 * @author xuebing
 */
public class RandomBing {

    public static Random Seeder = new Random(0);

    public static Random Create(){
        Random result = new Random();
        result.setSeed(Seeder.nextLong());
        return result;
    }


}


//
// Deterministic
//        RandomBing.Seeder.setSeed(100);
// Non-Deterministic
//        Random non_deterministic = new Random();
//        long seeder = non_deterministic.nextLong();
////        long  seede= ;
//        RandomBing.Seeder.setSeed(7806502880085430673l);
//        System.out.println("Seed=" + seeder);
//        System.out.println("RandomBing.Create()"+RandomBing.Create());
//        System.exit(9);

//        Dataset data = FileHandler.loadDataset(new File("F:/Java/Data/UCI-small/abalone/abalone.data"), 8, ",");
//        DistanceMeasure dm = new PearsonCorrelationCoefficient();
//        Instance classInstance = DatasetTools.createInstanceFromClass(data);
//        for (int i = 0; i < 8; i++) {
//            Instance attributeInstance = DatasetTools.createInstanceFromAttribute(data, i);
//            System.out.println(dm.measure(attributeInstance, classInstance));
//        }
//        System.out.print((double) (156.00 / 178.00));
