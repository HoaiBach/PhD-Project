package Utility;
import java.util.Random;

public class RandomSeed {

    public static Random Seeder = new Random(0);

    public static Random Create(){
        Random result = new Random();
        result.setSeed(Seeder.nextLong());
        return result;
    }
}
