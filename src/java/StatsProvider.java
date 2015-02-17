import java.util.List;

public interface StatsProvider {
	public double[] featureWiseVariance(double[][] data);
	
	public double cumulativeProbability(double x, int df);
}
