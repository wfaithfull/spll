import java.util.List;

/**
 * Created by Will Faithfull on 14/10/14.
 */
public interface ClusterProvider {

    List<double[][]> cluster(double[][] data, int k, int maxIterations);

}
