from dataclasses import dataclass

from shgp.data.dataset import *


@dataclass
class MetricsMetaDataset:
    """
        A dataset utilities class specifically for metrics experiments.

        # Shared
        :param num_cycles: The number of times to train a model and average results over.
        :param M: The number of inducing points to use.
        # SVGP
        :param svgp_iters: The number of iterations to train the SVGP model for.
        # PGPR
        :param inner_iters: The number of iterations of the inner optimisation loop.
        :param opt_iters: The number of iterations of gradient-based optimisation of the kernel hyperparameters.
        :param ci_iters: The number of iterations of update for the local variational parameters.
    """
    num_cycles: int
    M: int
    svgp_iters: int
    inner_iters: int
    opt_iters: int
    ci_iters: int


# TODO: Get metrics results for PGPR greedy var (non-hetero) and HGV gradient optim
#       To run on twonorm and ringnorm (joint)
#   This is the best way to phrase it (grad optim, intitialsed at hgv)
# TODO: Get metrics results for PGPR uniform, k_means on Magic and Electricity?
# TODO: Check that the Ms used below correspond to the convergent M achieved in sparsity plots!


"""
SVGP Distribution: (kmeans++, no grad-optim)
ELBO - max: -89.519647, min: -103.250001, median: -98.954888, mean: -98.629271, std: 3.787107.
ACC  - max: 0.975000, min: 0.825000, median: 0.912500, mean: 0.902500, std: 0.039449.
NLL  - max: 0.538161, min: 0.061780, median: 0.202625, mean: 0.222576, std: 0.123889.

SVGP Distribution: (kmeans++, *with* grad-optim)
ELBO - max: -88.777886, min: -166.742860, median: -98.449648, mean: -104.836131, std: 20.970514.
ACC  - max: 0.975000, min: 0.825000, median: 0.900000, mean: 0.902500, std: 0.042500.
NLL  - max: 0.497468, min: 0.067487, median: 0.196544, mean: 0.229954, std: 0.122943.

PGPR Distribution: (hetero greedy var, no grad-optim)
ELBO - max: -103.146880, min: -115.827680, median: -110.754296, mean: -110.913423, std: 3.235203.
ACC  - max: 1.000000, min: 0.825000, median: 0.925000, mean: 0.912500, std: 0.044791.
NLL  - max: 0.457047, min: 0.069402, median: 0.215834, mean: 0.216745, std: 0.097789.

PGPR Distribution: (uniform subsample, no grad-optim)
ELBO - max: -107.424944, min: -121.194819, median: -113.858812, mean: -114.140355, std: 3.902623.
ACC  - max: 1.000000, min: 0.825000, median: 0.912500, mean: 0.910000, std: 0.056125.
NLL  - max: 0.439950, min: 0.054075, median: 0.204322, mean: 0.210819, std: 0.106394.

PGPR Distribution: (k_means, no grad-optim)
ELBO - max: -104.259884, min: -117.645103, median: -111.919963, mean: -111.793438, std: 3.539366.
ACC  - max: 1.000000, min: 0.825000, median: 0.912500, mean: 0.910000, std: 0.056125.
NLL  - max: 0.433336, min: 0.052908, median: 0.202742, mean: 0.208474, std: 0.105845.

PGPR Distribution: (greedy var, no grad-optim)
ELBO - max: -105.114435, min: -115.935184, median: -110.190969, mean: -110.535456, std: 3.686483.
ACC  - max: 1.000000, min: 0.825000, median: 0.900000, mean: 0.907500, std: 0.055958.
NLL  - max: 0.453835, min: 0.069092, median: 0.229810, mean: 0.238702, std: 0.103961.

PGPR Distribution: (hetero greedy var, *with* grad-optim)
ELBO - max: -105.842257, min: -115.629812, median: -112.541827, mean: -111.658866, std: 3.045818.
ACC  - max: 1.000000, min: 0.825000, median: 0.925000, mean: 0.920000, std: 0.049749.
NLL  - max: 0.379085, min: 0.068993, median: 0.159590, mean: 0.188525, std: 0.093011.
"""


class BananaMetricsMetaDataset(BananaDataset, MetricsMetaDataset):
    def __init__(self):
        BananaDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 40, 250, 10, 250, 10)


"""
SVGP Distribution: (kmeans++, no grad-optim)
ELBO - max: -22.581493, min: -54.522554, median: -29.843940, mean: -32.031536, std: 8.377061.
ACC  - max: 1.000000, min: 1.000000, median: 1.000000, mean: 1.000000, std: 0.000000.
NLL  - max: 0.104137, min: 0.001727, median: 0.021329, mean: 0.029044, std: 0.028902.

SVGP Distribution: (kmeans++, *with* grad-optim)
ELBO - max: -36.455656, min: -44.474626, median: -42.407652, mean: -41.618363, std: 2.521321.
ACC  - max: 1.000000, min: 1.000000, median: 1.000000, mean: 1.000000, std: 0.000000.
NLL  - max: 0.096502, min: 0.031684, median: 0.056493, mean: 0.058082, std: 0.021037.

PGPR Distribution: (hetero greedy var, no grad-optim)
ELBO - max: -29.459306, min: -30.159630, median: -29.961047, mean: -29.885057, std: 0.197002.
ACC  - max: 1.000000, min: 1.000000, median: 1.000000, mean: 1.000000, std: 0.000000.
NLL  - max: 0.027977, min: 0.002512, median: 0.004539, mean: 0.010042, std: 0.008899.

PGPR Distribution: (uniform subsample, no grad-optim)
ELBO - max: -30.102250, min: -30.634909, median: -30.341378, mean: -30.322743, std: 0.160593.
ACC  - max: 1.000000, min: 1.000000, median: 1.000000, mean: 1.000000, std: 0.000000.
NLL  - max: 0.028861, min: 0.003745, median: 0.008912, mean: 0.012525, std: 0.008983.

PGPR Distribution: (k_means, no grad-optim)
ELBO - max: -30.011641, min: -30.558309, median: -30.275018, mean: -30.292264, std: 0.165048.
ACC  - max: 1.000000, min: 1.000000, median: 1.000000, mean: 1.000000, std: 0.000000.
NLL  - max: 0.028432, min: 0.003339, median: 0.009115, mean: 0.012525, std: 0.009005.

PGPR Distribution: (greedy var, no grad-optim)
ELBO - max: -29.632323, min: -30.141687, median: -29.922309, mean: -29.918207, std: 0.144355.
ACC  - max: 1.000000, min: 1.000000, median: 1.000000, mean: 1.000000, std: 0.000000.
NLL  - max: 0.016573, min: 0.002571, median: 0.007889, mean: 0.008127, std: 0.004350.

PGPR Distribution: (hetero greedy var, *with* grad-optim)
ELBO - max: -29.525481, min: -30.109749, median: -29.854573, mean: -29.864040, std: 0.150605.
ACC  - max: 1.000000, min: 1.000000, median: 1.000000, mean: 1.000000, std: 0.000000.
NLL  - max: 0.017193, min: 0.002502, median: 0.007596, mean: 0.007940, std: 0.004309.
"""


class CrabsMetricsMetaDataset(CrabsDataset, MetricsMetaDataset):
    # Bernoulli requires large number of training iters for this dataset
    def __init__(self):
        CrabsDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 10, 1000, 10, 250, 10)


"""
SVGP Distribution: (kmeans++, no grad-optim)
ELBO - max: -102.457024, min: -147.645443, median: -105.172674, mean: -116.764816, std: 19.141206.
ACC  - max: 0.888889, min: 0.740741, median: 0.814815, mean: 0.825926, std: 0.037222.
NLL  - max: 0.541615, min: 0.234183, median: 0.394648, mean: 0.394743, std: 0.095801.

SVGP Distribution: (kmeans++, *with* grad-optim)
ELBO - max: -98.888180, min: -148.291108, median: -102.323782, mean: -106.737562, std: 14.027806.
ACC  - max: 0.925926, min: 0.740741, median: 0.814815, mean: 0.814815, std: 0.064150.
NLL  - max: 0.557633, min: 0.215903, median: 0.430545, mean: 0.423089, std: 0.103536.

PGPR Distribution: (hetero greedy var, no grad-optim)
ELBO - max: -105.712986, min: -109.829346, median: -107.190899, mean: -107.372922, std: 1.172509.
ACC  - max: 0.888889, min: 0.814815, median: 0.851852, mean: 0.844444, std: 0.027716.
NLL  - max: 0.424670, min: 0.240716, median: 0.347886, mean: 0.347235, std: 0.050955.

PGPR Distribution: (uniform subsample, no grad-optim)
ELBO - max: -98.576805, min: -112.172565, median: -106.693834, mean: -105.828913, std: 4.080949.
ACC  - max: 0.962963, min: 0.666667, median: 0.851852, mean: 0.833333, std: 0.086464.
NLL  - max: 0.705459, min: 0.166910, median: 0.377869, mean: 0.408043, std: 0.164833.

PGPR Distribution: (k_means, no grad-optim)
ELBO - max: -98.474002, min: -112.002983, median: -106.549697, mean: -105.677156, std: 4.063422.
ACC  - max: 0.962963, min: 0.666667, median: 0.851852, mean: 0.833333, std: 0.086464.
NLL  - max: 0.705542, min: 0.166731, median: 0.377724, mean: 0.408234, std: 0.164898.

PGPR Distribution: (greedy var, no grad-optim)
ELBO - max: -100.655929, min: -108.262102, median: -106.784653, mean: -106.207520, std: 2.123873.
ACC  - max: 0.925926, min: 0.703704, median: 0.870370, mean: 0.855556, std: 0.058443.
NLL  - max: 0.621502, min: 0.311186, median: 0.371856, mean: 0.389818, std: 0.087791.

PGPR Distribution: (hetero greedy var, *with* grad-optim)
ELBO - max: -100.313782, min: -107.880601, median: -106.391733, mean: -105.825597, std: 2.108760.
ACC  - max: 0.925926, min: 0.703704, median: 0.851852, mean: 0.851852, std: 0.057378.
NLL  - max: 0.620721, min: 0.311570, median: 0.372340, mean: 0.390028, std: 0.087604.
"""


class HeartMetricsMetaDataset(HeartDataset, MetricsMetaDataset):
    def __init__(self):
        HeartDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 35, 250, 10, 250, 10)


"""
SVGP Distribution: (kmeans++, no grad-optim)
ELBO - max: -102.320602, min: -111.255612, median: -107.341194, mean: -107.392428, std: 2.878623.
ACC  - max: 0.972222, min: 0.861111, median: 0.888889, mean: 0.911111, std: 0.042673.
NLL  - max: 0.539788, min: 0.087637, median: 0.269310, mean: 0.276174, std: 0.127492.

SVGP Distribution: (kmeans++, *with* grad-optim)
ELBO - max: -93.940266, min: -101.288614, median: -98.460579, mean: -97.753783, std: 2.390541.
ACC  - max: 1.000000, min: 0.888889, median: 0.944444, mean: 0.936111, std: 0.035246.
NLL  - max: 0.345818, min: 0.061877, median: 0.186034, mean: 0.205206, std: 0.089058.

PGPR Distribution: (hetero greedy var, no grad-optim)
ELBO - max: -116.037903, min: -125.215633, median: -120.372931, mean: -120.675359, std: 2.760629.
ACC  - max: 0.972222, min: 0.833333, median: 0.861111, mean: 0.886111, std: 0.042035.
NLL  - max: 0.502648, min: 0.121410, median: 0.301423, mean: 0.307200, std: 0.112117.

PGPR Distribution: (uniform subsample, no grad-optim)
ELBO - max: -116.084780, min: -122.249368, median: -119.974776, mean: -119.350842, std: 2.225078.
ACC  - max: 0.888889, min: 0.805556, median: 0.861111, mean: 0.866667, std: 0.024216.
NLL  - max: 0.530426, min: 0.244046, median: 0.335632, mean: 0.358985, std: 0.092933.

PGPR Distribution: (k_means, no grad-optim)
ELBO - max: -115.889836, min: -122.028524, median: -119.628350, mean: -119.111147, std: 2.186207.
ACC  - max: 0.888889, min: 0.805556, median: 0.861111, mean: 0.866667, std: 0.024216.
NLL  - max: 0.530372, min: 0.244066, median: 0.331379, mean: 0.356909, std: 0.093422.

PGPR Distribution: (greedy var, no grad-optim)
ELBO - max: -114.372871, min: -124.591728, median: -119.594699, mean: -119.101956, std: 3.255026.
ACC  - max: 0.916667, min: 0.833333, median: 0.875000, mean: 0.875000, std: 0.033449.
NLL  - max: 0.585661, min: 0.151937, median: 0.326092, mean: 0.370183, std: 0.145032.

PGPR Distribution: (hetero greedy var, *with* grad-optim)
ELBO - max: -113.749178, min: -123.980814, median: -119.198269, mean: -118.599297, std: 3.242994.
ACC  - max: 0.944444, min: 0.833333, median: 0.875000, mean: 0.877778, std: 0.037680.
NLL  - max: 0.633198, min: 0.138230, median: 0.328473, mean: 0.376487, std: 0.157450.
"""


class IonosphereMetricsMetaDataset(IonosphereDataset, MetricsMetaDataset):
    def __init__(self):
        IonosphereDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 50, 500, 20, 500, 20)


"""
SVGP Distribution: (kmeans++, no grad-optim)
ELBO - max: -51.035899, min: -260.193332, median: -79.258723, mean: -128.164729, std: 87.821823.
ACC  - max: 1.000000, min: 0.877193, median: 0.956140, mean: 0.954386, std: 0.034379.
NLL  - max: 0.438799, min: 0.035690, median: 0.147612, mean: 0.198864, std: 0.145042.

SVGP Distribution: (kmeans++, *with* grad-optim)
ELBO - max: -49.448369, min: -60.149434, median: -55.705711, mean: -55.182518, std: 2.716737.
ACC  - max: 1.000000, min: 0.947368, median: 0.982456, mean: 0.977193, std: 0.015789.
NLL  - max: 0.226910, min: 0.014528, median: 0.074933, mean: 0.093823, std: 0.061250.

PGPR Distribution: (hetero greedy var, no grad-optim)
ELBO - max: -65.263036, min: -71.668893, median: -69.381602, mean: -69.548874, std: 1.810755.
ACC  - max: 1.000000, min: 0.947368, median: 0.982456, mean: 0.980702, std: 0.014573.
NLL  - max: 0.156879, min: 0.037506, median: 0.085697, mean: 0.082185, std: 0.033648.

PGPR Distribution: (uniform subsample, no grad-optim)
ELBO - max: -63.706091, min: -73.053283, median: -69.586676, mean: -68.642236, std: 3.426317.
ACC  - max: 1.000000, min: 0.929825, median: 0.964912, mean: 0.970175, std: 0.024873.
NLL  - max: 0.244724, min: 0.032159, median: 0.091112, mean: 0.120495, std: 0.079303.

PGPR Distribution: (k_means, no grad-optim)
ELBO - max: -63.483362, min: -72.851351, median: -69.530363, mean: -68.509234, std: 3.428112.
ACC  - max: 1.000000, min: 0.929825, median: 0.964912, mean: 0.970175, std: 0.024873.
NLL  - max: 0.243820, min: 0.032183, median: 0.091140, mean: 0.120344, std: 0.079160.

PGPR Distribution: (greedy var, no grad-optim)
ELBO - max: -65.145384, min: -72.811842, median: -71.469509, mean: -70.436993, std: 2.435500.
ACC  - max: 1.000000, min: 0.964912, median: 0.982456, mean: 0.984211, std: 0.014573.
NLL  - max: 0.174146, min: 0.024167, median: 0.047871, mean: 0.068991, std: 0.048040.

PGPR Distribution: (hetero greedy var, *with* grad-optim)
ELBO - max: -64.919496, min: -72.588883, median: -71.218075, mean: -70.205327, std: 2.433261.
ACC  - max: 1.000000, min: 0.964912, median: 0.982456, mean: 0.984211, std: 0.014573.
NLL  - max: 0.173942, min: 0.023864, median: 0.048304, mean: 0.068886, std: 0.048026.
"""


class BreastCancerMetricsMetaDataset(BreastCancerDataset, MetricsMetaDataset):
    def __init__(self):
        BreastCancerDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 50, 500, 10, 250, 10)


"""
SVGP Distribution: (kmeans++, no grad-optim)
ELBO - max: -335.880773, min: -346.814701, median: -342.072790, mean: -341.638486, std: 3.397007.
ACC  - max: 0.844156, min: 0.714286, median: 0.785714, mean: 0.780519, std: 0.035065.
NLL  - max: 0.522449, min: 0.364078, median: 0.443357, mean: 0.441000, std: 0.047642.

SVGP Distribution: (kmeans++, *with* grad-optim)
ELBO - max: -335.544731, min: -345.226318, median: -339.586144, mean: -340.116380, std: 2.973971.
ACC  - max: 0.870130, min: 0.753247, median: 0.798701, mean: 0.798701, std: 0.035448.
NLL  - max: 0.493747, min: 0.359438, median: 0.442572, mean: 0.431360, std: 0.040611.

PGPR Distribution: (hetero greedy var, no grad-optim)
ELBO - max: -341.612109, min: -350.626261, median: -345.751315, mean: -346.000145, std: 2.916892.
ACC  - max: 0.857143, min: 0.766234, median: 0.785714, mean: 0.797403, std: 0.030291.
NLL  - max: 0.489695, min: 0.371073, median: 0.440371, mean: 0.434471, std: 0.038771.

PGPR Distribution: (uniform subsample, no grad-optim)
ELBO - max: -339.664353, min: -351.340364, median: -343.859690, mean: -344.068017, std: 2.994499.
ACC  - max: 0.857143, min: 0.714286, median: 0.759740, mean: 0.770130, std: 0.042699.
NLL  - max: 0.540152, min: 0.371490, median: 0.473276, mean: 0.471043, std: 0.043061.

PGPR Distribution: (k_means, no grad-optim)
ELBO - max: -339.135498, min: -350.284142, median: -343.076570, mean: -343.219577, std: 2.871558.
ACC  - max: 0.844156, min: 0.714286, median: 0.753247, mean: 0.767532, std: 0.042462.
NLL  - max: 0.537636, min: 0.369869, median: 0.471164, mean: 0.469568, std: 0.042902.

PGPR Distribution: (greedy var, no grad-optim)
ELBO - max: -333.930295, min: -349.471636, median: -343.712130, mean: -343.782834, std: 4.828198.
ACC  - max: 0.870130, min: 0.714286, median: 0.766234, mean: 0.788312, std: 0.056325.
NLL  - max: 0.613338, min: 0.387170, median: 0.461009, mean: 0.466531, std: 0.069375.

PGPR Distribution: (hetero greedy var, *with* grad-optim)
ELBO - max: -333.042295, min: -347.209922, median: -342.095503, mean: -342.048507, std: 4.495234.
ACC  - max: 0.857143, min: 0.701299, median: 0.772727, mean: 0.779221, std: 0.055708.
NLL  - max: 0.598265, min: 0.393203, median: 0.459524, mean: 0.463622, std: 0.064727.
"""


class PimaMetricsMetaDataset(PimaDataset, MetricsMetaDataset):
    def __init__(self):
        PimaDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 60, 500, 10, 250, 10)


"""
SVGP Distribution: (kmeans++, no grad-optim, M=300)
ELBO - max: -440.252890, min: -4480.603436, median: -4454.929089, mean: -3581.029505, std: 1314.972465.
ACC  - max: 0.982432, min: 0.810811, median: 0.968243, mean: 0.943649, std: 0.057986.
NLL  - max: 0.667848, min: 0.054345, median: 0.663046, mean: 0.516713, std: 0.204524.

SVGP Distribution: (kmeans++, *with* grad-optim, M=300)
ELBO - max: -3706.290435, min: -4527.708744, median: -4466.294742, mean: -4362.042284, std: 230.266477.
ACC  - max: 0.968919, min: 0.689189, median: 0.922297, mean: 0.893108, std: 0.081363.
NLL  - max: 0.673092, min: 0.489449, median: 0.665749, mean: 0.631629, std: 0.055333.

PGPR Distribution: (hetero greedy var, no grad-optim, M=300)
ELBO - max: -441.419710, min: -464.018613, median: -452.083285, mean: -451.609530, std: 7.613344.
ACC  - max: 0.986486, min: 0.972973, median: 0.979730, mean: 0.979324, std: 0.004799.
NLL  - max: 0.075884, min: 0.043305, median: 0.052011, mean: 0.054395, std: 0.010713.

PGPR Distribution: (uniform subsample, no grad-optim, M=300)
ELBO - max: -431.669276, min: -467.890859, median: -455.823611, mean: -455.121204, std: 9.190612.
ACC  - max: 0.986486, min: 0.967568, median: 0.977703, mean: 0.978243, std: 0.005036.
NLL  - max: 0.091890, min: 0.044199, median: 0.060317, mean: 0.060964, std: 0.012172.

PGPR Distribution: (k_means, no grad-optim, M=300)
ELBO - max: -431.688747, min: -467.903019, median: -455.840175, mean: -455.136209, std: 9.188429.
ACC  - max: 0.986486, min: 0.967568, median: 0.977703, mean: 0.978243, std: 0.005036.
NLL  - max: 0.091879, min: 0.044205, median: 0.060318, mean: 0.060964, std: 0.012168.

PGPR Distribution: (greedy var, no grad-optim, M=300)
ELBO - max: -444.943399, min: -465.351050, median: -458.610668, mean: -457.373139, std: 6.188278.
ACC  - max: 0.983784, min: 0.972973, median: 0.978378, mean: 0.977703, std: 0.003928.
NLL  - max: 0.073826, min: 0.045847, median: 0.054184, mean: 0.056054, std: 0.008707.

PGPR Distribution: (hetero greedy var, *with* grad-optim, M=300)
ELBO - max: -444.979294, min: -464.877358, median: -458.634332, mean: -457.330837, std: 6.124120.
ACC  - max: 0.983784, min: 0.972973, median: 0.978378, mean: 0.977703, std: 0.003928.
NLL  - max: 0.073811, min: 0.045877, median: 0.054185, mean: 0.056056, std: 0.008700.
"""


class TwonormMetricsMetaDataset(TwonormDataset, MetricsMetaDataset):
    def __init__(self):
        TwonormDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 300, 500, 10, 250, 10)


"""
SVGP Distribution: (kmeans++, no grad-optim, M=300)
ELBO - max: -768.285139, min: -4332.227797, median: -2597.246881, mean: -2260.877601, std: 1301.823368.
ACC  - max: 0.979730, min: 0.495946, median: 0.871622, mean: 0.803243, std: 0.183625.
NLL  - max: 0.646989, min: 0.043059, median: 0.315875, mean: 0.290692, std: 0.209444.

SVGP Distribution: (kmeans++, *with* grad-optim, M=300)
ELBO - max: -745.050897, min: -4013.076913, median: -768.661123, mean: -1675.260796, std: 1401.709263.
ACC  - max: 0.987838, min: 0.493243, median: 0.975676, mean: 0.840946, std: 0.213598.
NLL  - max: 0.590866, min: 0.041503, median: 0.070809, mean: 0.207506, std: 0.229643.

PGPR Distribution: (hetero greedy var, no grad-optim, M=300)
ELBO - max: -933.852979, min: -967.672191, median: -952.890675, mean: -953.277250, std: 9.537135.
ACC  - max: 0.989189, min: 0.964865, median: 0.980405, mean: 0.976622, std: 0.007023.
NLL  - max: 0.097374, min: 0.037707, median: 0.054798, mean: 0.062037, std: 0.015969.

PGPR Distribution: (uniform subsample, no grad-optim, M=300)
ELBO - max: -932.467899, min: -980.560829, median: -960.690189, mean: -959.429487, std: 15.226527.
ACC  - max: 0.987838, min: 0.967568, median: 0.975676, mean: 0.977162, std: 0.005935.
NLL  - max: 0.082750, min: 0.048434, median: 0.070451, mean: 0.066714, std: 0.012016.

PGPR Distribution: (k_means, no grad-optim, M=300)
ELBO - max: -933.976157, min: -971.332302, median: -954.930721, mean: -956.432142, std: 10.902029.
ACC  - max: 0.987838, min: 0.968919, median: 0.975676, mean: 0.977162, std: 0.005873.
NLL  - max: 0.082839, min: 0.048921, median: 0.070721, mean: 0.067049, std: 0.011971.

PGPR Distribution: (greedy var, no grad-optim, M=300)
ELBO - max: -926.199076, min: -4616.360228, median: -965.481081, mean: -1685.342074, std: 1465.640732.
ACC  - max: 0.981081, min: 0.795946, median: 0.975000, mean: 0.947703, std: 0.058375.
NLL  - max: 0.693147, min: 0.050673, median: 0.071076, mean: 0.191241, std: 0.251122.

PGPR Distribution: (hetero greedy var, *with* grad-optim, M=300)
ELBO - max: -904.414564, min: -4616.360228, median: -922.584130, mean: -1657.752554, std: 1479.318193.
ACC  - max: 0.979730, min: 0.498649, median: 0.975000, mean: 0.916892, std: 0.143329.
NLL  - max: 0.693147, min: 0.049834, median: 0.069067, mean: 0.189993, std: 0.251775.
"""


class RingnormMetricsMetaDataset(RingnormDataset, MetricsMetaDataset):
    def __init__(self):
        RingnormDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 300, 500, 10, 250, 10)


"""
SVGP Distribution: (kmeans++, no grad-optim, M=300)
ELBO - max: -5697.229763, min: -5781.106789, median: -5725.451467, mean: -5731.254334, std: 22.866931.
ACC  - max: 0.885910, min: 0.862776, median: 0.870925, mean: 0.871293, std: 0.005947.
NLL  - max: 0.331892, min: 0.293679, median: 0.316215, mean: 0.316249, std: 0.010425.

SVGP Distribution: (kmeans++, *with* grad-optim, M=300)


PGPR Distribution: (hetero greedy var, no grad-optim, M=300)
ELBO - max: -5807.941402, min: -5868.467288, median: -5828.071268, mean: -5834.611281, std: 19.699746.
ACC  - max: 0.882229, min: 0.860147, median: 0.868822, mean: 0.869401, std: 0.006055.
NLL  - max: 0.368359, min: 0.300778, median: 0.324061, mean: 0.324389, std: 0.017964.

PGPR Distribution: (uniform subsample, no grad-optim, M=300)

PGPR Distribution: (k_means, no grad-optim, M=300)

PGPR Distribution: (greedy var, no grad-optim, M=300)

PGPR Distribution: (hetero greedy var, *with* grad-optim, M=300)

"""


# TODO: Run experiment on Colab
# TODO: Perhaps performance is worse for HGV than k-means as M isn't large enough
class MagicMetricsMetaDataset(MagicDataset, MetricsMetaDataset):
    def __init__(self):
        MagicDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 300, 500, 10, 250, 10)


"""
SVGP Distribution: (kmeans++, no grad-optim, M=300)

SVGP Distribution: (kmeans++, *with* grad-optim, M=300)

PGPR Distribution: (hetero greedy var, no grad-optim, M=300)

PGPR Distribution: (uniform subsample, no grad-optim, M=300)

PGPR Distribution: (k_means, no grad-optim, M=300)

PGPR Distribution: (greedy var, no grad-optim, M=300)

PGPR Distribution: (hetero greedy var, *with* grad-optim, M=300)

"""

# TODO: Compute results
# 10, 300, 500, 10, 250, 10
# Beginning cycle 1...
# PGPR ELBO failed to converge: prev -18415.408450099414, next -18396.388646198277.
# SVGP: ELBO = -18101.636126, ACC = 0.810238, NLL = 0.420085.
# PGPR: ELBO = -18383.393121, ACC = 0.803398, NLL = 0.427922.
# Beginning cycle 2...
# PGPR ELBO failed to converge: prev -18358.49135184391, next -18339.628066297373.
# SVGP: ELBO = -18085.934464, ACC = 0.800309, NLL = 0.429165.
# PGPR: ELBO = -18339.628066, ACC = 0.793689, NLL = 0.437207.
# Beginning cycle 3...
# PGPR ELBO failed to converge: prev -18308.17050435062, next -18311.187940974338.
# SVGP: ELBO = -18039.079090, ACC = 0.801412, NLL = 0.435286.
# PGPR: ELBO = -18308.170504, ACC = 0.799206, NLL = 0.451750.
# Beginning cycle 4...
# Cholesky failed with maximum jitter.
# Cholesky error caught, retrying...
# PGPR ELBO failed to converge: prev -18408.913979693327, next -18378.57118006317.
# SVGP: ELBO = -18141.160896, ACC = 0.808252, NLL = 0.423720.
# PGPR: ELBO = -18373.837635, ACC = 0.806267, NLL = 0.432042.
# Beginning cycle 5...
# PGPR ELBO failed to converge: prev -18360.466179625975, next -18341.354457487796.
# SVGP: ELBO = -18152.600065, ACC = 0.813548, NLL = 0.416427.
# PGPR: ELBO = -18341.354457, ACC = 0.805605, NLL = 0.425906.
# Beginning cycle 6...
# PGPR ELBO failed to converge: prev -18394.74792047306, next -18408.51579684986.
# SVGP: ELBO = -18117.216560, ACC = 0.805605, NLL = 0.418614.
# PGPR: ELBO = -18394.747920, ACC = 0.804501, NLL = 0.425104.
# Beginning cycle 7...

# Beginning cycle 1...
# Cholesky failed with maximum jitter.
# Cholesky error caught, retrying...
# PGPR ELBO failed to converge: prev -18379.659842897516, next -18330.932710341793.
# SVGP: ELBO = -18100.133043, ACC = 0.800750, NLL = 0.428685.
# PGPR: ELBO = -18330.932710, ACC = 0.796558, NLL = 0.435063.
# Beginning cycle 2...
# PGPR ELBO failed to converge: prev -18365.08189566786, next -18335.47831834503.
# SVGP: ELBO = -18164.863439, ACC = 0.805384, NLL = 0.422066.
# PGPR: ELBO = -18335.478318, ACC = 0.803839, NLL = 0.427654.
# Beginning cycle 3...
# PGPR ELBO failed to converge: prev -18344.000485800905, next -18338.904702654185.
# SVGP: ELBO = -18113.853773, ACC = 0.803398, NLL = 0.424619.
# PGPR: ELBO = -18338.904703, ACC = 0.801412, NLL = 0.430044.
# Beginning cycle 4...
# PGPR ELBO failed to converge: prev -18325.330732839262, next -18296.95940539742.
# SVGP: ELBO = -18082.869820, ACC = 0.799426, NLL = 0.434670.
# PGPR: ELBO = -18292.692873, ACC = 0.798985, NLL = 0.440920.
# Beginning cycle 5...
# PGPR ELBO failed to converge: prev -18408.03723792868, next -18383.60226338658.
# SVGP: ELBO = -18150.090232, ACC = 0.801412, NLL = 0.421257.
# PGPR: ELBO = -18383.602263, ACC = 0.802957, NLL = 0.424809.
# Beginning cycle 6...
# PGPR ELBO failed to converge: prev -18398.355014462602, next -18392.530243766036.
# SVGP: ELBO = -18111.560910, ACC = 0.808914, NLL = 0.419617.
# PGPR: ELBO = -18387.196726, ACC = 0.801192, NLL = 0.429616.
# Beginning cycle 7...
# PGPR ELBO failed to converge: prev -18380.56610272094, next -18377.507000804017.
# SVGP: ELBO = -18134.107411, ACC = 0.808032, NLL = 0.422314.
# PGPR: ELBO = -18354.383904, ACC = 0.800750, NLL = 0.430153.
# Beginning cycle 8...
# PGPR ELBO failed to converge: prev -18386.52170452088, next -18399.379634296027.
# SVGP: ELBO = -18100.638620, ACC = 0.809356, NLL = 0.419212.
# PGPR: ELBO = -18355.680738, ACC = 0.804943, NLL = 0.427674.
# Beginning cycle 9...
# PGPR ELBO failed to converge: prev -18303.571028784576, next -18283.04250305434.
# SVGP: ELBO = -18092.365786, ACC = 0.790159, NLL = 0.439643.
# PGPR: ELBO = -18283.042503, ACC = 0.791041, NLL = 0.445399.
# Beginning cycle 10...


# TODO: Run experiment on Colab
# TODO: Perhaps performance is worse for HGV than k-means as M isn't large enough
class ElectricityMetricsMetaDataset(ElectricityDataset, MetricsMetaDataset):
    def __init__(self):
        ElectricityDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 300, 500, 10, 250, 10)
