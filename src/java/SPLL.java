
import java.util.ArrayList;
import java.util.List;

/**
 * Semi Parametric Log Likelihood change detection criterion for multivariate data.
 * 
 * Algorithm courtesy of Prof. Ludmila I. Kuncheva (l.i.kuncheva@bangor.ac.uk), introduced in:
 * 
 * Change detection in streaming multivariate data using likelihood detectors.
 * Knowledge and Data Engineering, IEEE Transactions on 25.5 (2013): 1175-1180.
 * 
 * @author Will Faithfull (w.faithfull@bangor.ac.uk)
 */
public class SPLL {
	
	// These are the values we tend to use.
	private final int DEFAULT_MAX_ITERATIONS = 100;
	private final int DEFAULT_N_CLUSTERS = 3;
	
	// But you can change them if you disagree.
	private int maxIterations;
	private int numClusters;
	
	// Injected clustering and CDF implementations.
	private ClusterProvider clusterer;
	private StatsProvider stats;

	public SPLL(final ClusterProvider clusterer,final StatsProvider stats) {
		this.setClusterer(clusterer);
		this.setStatsProvider(stats);
		
		setMaxIterations(DEFAULT_MAX_ITERATIONS);
		setNumClusters(DEFAULT_N_CLUSTERS);
	}
	
	public SPLL() {
		// We use K-Means and Chi Square.
		this(new ApacheKMeansAdapter(), new ApacheStatsAdapter());
	}
	
	private List<double[]> getClusterVariance(List<double[][]> clusters) {
		List<double[]> clusterVariance = new ArrayList<double[]>();
		
		for(int i=0; i<clusters.size(); i++) {
			double[] variance = stats.featureWiseVariance(clusters.get(i));
			clusterVariance.add(variance);
		}
		
		return clusterVariance;
	}
	
	private List<double[]> getClusterMeans(List<double[][]> clusters) {
		List<double[]> clusterMeans = new ArrayList<double[]>();
		
		for(int i=0;i<clusters.size(); i++){
			double[][] cluster = clusters.get(i);
			int nObsv		= cluster.length;
			int nFeatures 	= cluster[0].length;
			
			double[] center = new double[nFeatures];
			for(int j=0;j<nObsv;j++) {
				for(int k=0; k<nFeatures; k++) {
					center[k] += cluster[j][k];
				}
				
			}
			
			for(int k=0; k<nFeatures; k++) {
				center[k] /= nObsv;
			}
			
			clusterMeans.add(center);
		}
		return clusterMeans;
	}

	@SuppressWarnings("unused")
	private static double mahalanobis(double[] xx, double[] mk, double[] reciprocals) {
        double dist = 0;
        for(int j=0;j<xx.length;j++) {
            dist += Math.pow(xx[j] - mk[j], 2) * reciprocals[j];
        }
        return dist;
    }
	
	public LikelihoodResult testChange(double[][] w1, double[][] w2) {
		
		// Detector needs to be run forwards and backwards
		// to guarantee a fair result.
		LikelihoodResult forwards 	= logLL(w1, w2);
		LikelihoodResult backwards 	= logLL(w2, w1);
		
		boolean winner = forwards.rawStat > backwards.rawStat;
		
		return winner ? forwards : backwards; 
	}
	
	public LikelihoodResult logLL(double[][] w1, double[][] w2)
	{
		List<double[][]> clusters = getClusterer().cluster(w1, numClusters, maxIterations);
		
    	int totalObservations   = w1.length;
        int nFeatures           = w1[0].length;	
        int nClusters			= clusters.size();
        
        double[] classCount 	= new double[nClusters];
        double[] classPriors 	= new double[nClusters];

        // Calculate class priors by counting cluster membership
        for(int k=0;k<nClusters;k++) {
        	double[][] cluster 	= clusters.get(k);
			classCount[k] 		= cluster.length;
			classPriors[k] 		= classCount[k] / totalObservations;
		}
        
        List<double[]> clusterMeans = getClusterMeans(clusters);
        List<double[]> clusterVariance = getClusterVariance(clusters);

        /* Combine cluster variances into the final covariance matrix, weighted by priors.
        ~ One covariance matrix to rule them all,
        ~ One covariance matrix to find them.
        ~ One covariance matrix to bring them all,
        ~ And in the darkness bind them.
        ~
        ~ ... Except not. We are lazy, and only calculate the diagonal.
        */
        double[] featureVariance = new double[nFeatures];
        double minVariance = Double.MAX_VALUE;
        
        for(int j=0;j<nFeatures;j++) {
	        double cov = 0;
	        
	        // Sum over clusters. Weight by priors.
	        for(int k=0;k<clusters.size();k++) {
	        	cov += (clusterVariance.get(k)[j] * classPriors[k]);
	        }

	        if(cov != 0 && cov < minVariance)
                minVariance = cov;

	        featureVariance[j] = cov; // We can cheat and only do the diagonal
        }
        
        double[] reciprocalVariance = new double[nFeatures];

        for(int j=0;j<nFeatures;j++) {
            if(featureVariance[j] == 0)
                featureVariance[j] = minVariance; // Guard against 0 variance
            
            reciprocalVariance[j] = 1 / featureVariance[j]; // Precalculate reciprocals
        }
        
		
        double logLikelihoodTerm = 0;
        for(int i=0;i<totalObservations;i++) {
            double minDist = Double.MAX_VALUE;
            for (int k = 0; k < nClusters; k++) {
            	
            	// This is a transliteration of the actual MATLAB code
            	double[] distanceToMean = new double[nFeatures];
            	double dst = 0;
            	for(int j=0;j<nFeatures;j++)
            	{
            		double[] clusterMean = clusterMeans.get(k);
            		double[] xx = w2[i];
            		distanceToMean[j] = (clusterMean[j] - xx[j]);
            		dst += (distanceToMean[j] * reciprocalVariance[j]) * distanceToMean[j];
            	}
            	// </transliteration>
            	
            	// This is the suggested measure. They produce quite different results.
                //double dst = mahalanobis(w2.get(i).toDoubleArray(), clusterMeans.get(k), reciprocalVariance);
            	
                if (dst < minDist) {
                    minDist = dst;
                }
            }

            logLikelihoodTerm += minDist;
        }
        
        double meanLL 	= logLikelihoodTerm / totalObservations; // Mean Log-Likelihood term
        
        double a 		= getStatsProvider().cumulativeProbability(meanLL, nFeatures);
        double b 		= 1-a;

        double chi2Stat 	= a < b ? a : b;
        boolean change = chi2Stat < 0.05;
        
        return new LikelihoodResult(change, chi2Stat, meanLL);
	}

    public int getMaxIterations() {
		return maxIterations;
	}

	public void setMaxIterations(int maxIterations) {
		this.maxIterations = maxIterations;
	}

	public int getNumClusters() {
		return numClusters;
	}

	public void setNumClusters(int numClusters) {
		this.numClusters = numClusters;
	}
	
	public ClusterProvider getClusterer() {
		return clusterer;
	}

	public void setClusterer(ClusterProvider clusterer) {
		this.clusterer = clusterer;
	}
	
	public StatsProvider getStatsProvider() {
		return stats;
	}
	
	public void setStatsProvider(StatsProvider provider)
	{
		this.stats = provider;
	}

    public class LikelihoodResult {
    	
        public final boolean change;
        public final double chi2Stat;
        public final double rawStat;
        
        public LikelihoodResult(final boolean change, final double chi2Stat, final double rawStat)
        {
        	this.change = change;
        	this.chi2Stat = chi2Stat;
        	this.rawStat = rawStat;
        }
        
    }
}
