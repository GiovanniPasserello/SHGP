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


"""
SVGP Distribution: (k_means, no grad-optim)
ELBO - max: -89.519647, min: -103.250001, median: -98.954888, mean: -98.629271, std: 3.787107.
ACC  - max: 0.975000, min: 0.825000, median: 0.912500, mean: 0.902500, std: 0.039449.
NLL  - max: 0.538161, min: 0.061780, median: 0.202625, mean: 0.222576, std: 0.123889.

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

###################################
#####  Gradient Optimisation  #####
###################################

SVGP Distribution: (k_means, *with* grad-optim)
ELBO - max: -91.609100, min: -102.746290, median: -98.136628, mean: -98.025583, std: 3.906769.
ACC - max: 0.975000, min: 0.825000, median: 0.950000, mean: 0.917500, std: 0.057064.
NLL - max: 0.401933, min: 0.065484, median: 0.220048, mean: 0.217418, std: 0.123041.

PGPR Distribution: (k_means, *with* grad-optim)
ELBO - max: -105.626915, min: -115.629807, median: -111.116798, mean: -111.170208, std: 3.393407.
ACC - max: 1.000000, min: 0.825000, median: 0.925000, mean: 0.910000, std: 0.059372.
NLL - max: 0.377189, min: 0.069321, median: 0.218586, mean: 0.213890, std: 0.104379.
"""


class BananaMetricsMetaDataset(BananaDataset, MetricsMetaDataset):
    def __init__(self):
        BananaDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 40, 250, 10, 250, 10)


"""
SVGP Distribution: (k_means, no grad-optim)
ELBO - max: -22.581493, min: -54.522554, median: -29.843940, mean: -32.031536, std: 8.377061.
ACC  - max: 1.000000, min: 1.000000, median: 1.000000, mean: 1.000000, std: 0.000000.
NLL  - max: 0.104137, min: 0.001727, median: 0.021329, mean: 0.029044, std: 0.028902.

PGPR Distribution: (hetero greedy var, no grad-optim)
ELBO - max: -29.459306, min: -30.109630, median: -29.961047, mean: -29.885057, std: 0.197002.
ACC  - max: 1.000000, min: 1.000000, median: 1.000000, mean: 1.000000, std: 0.000000.
NLL  - max: 0.027977, min: 0.002512, median: 0.004539, mean: 0.007942, std: 0.008899.

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

###################################
#####  Gradient Optimisation  #####
###################################

SVGP Distribution: (k_means, *with* grad-optim)
ELBO - max: -35.761997, min: -47.524147, median: -41.124035, mean: -41.127233, std: 2.928483.
ACC  - max: 1.000000, min: 0.950000, median: 1.000000, mean: 0.995000, std: 0.015000.
NLL  - max: 0.098970, min: 0.032850, median: 0.038808, mean: 0.056910, std: 0.024781.

PGPR Distribution: (k_means, *with* grad-optim)
ELBO - max: -33.368825, min: -43.575429, median: -37.193093, mean: -37.483213, std: 2.911335.
ACC  - max: 1.000000, min: 1.000000, median: 1.000000, mean: 1.000000, std: 0.000000.
NLL  - max: 0.073733, min: 0.007476, median: 0.036561, mean: 0.037793, std: 0.019290.
"""


class CrabsMetricsMetaDataset(CrabsDataset, MetricsMetaDataset):
    # Bernoulli requires large number of training iters for this dataset
    def __init__(self):
        CrabsDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 10, 1000, 10, 250, 10)


"""
SVGP Distribution: (k_means, no grad-optim)
ELBO - max: -102.457024, min: -147.645443, median: -105.172674, mean: -116.764816, std: 19.141206.
ACC  - max: 0.888889, min: 0.740741, median: 0.814815, mean: 0.825926, std: 0.037222.
NLL  - max: 0.541615, min: 0.234183, median: 0.394648, mean: 0.394743, std: 0.095801.

PGPR Distribution: (hetero greedy var, no grad-optim)
ELBO - max: -105.712986, min: -109.829346, median: -107.190899, mean: -107.372922, std: 1.172509.
ACC  - max: 0.888889, min: 0.814815, median: 0.851852, mean: 0.844444, std: 0.027716.
NLL  - max: 0.424670, min: 0.240716, median: 0.347886, mean: 0.347235, std: 0.050955.

PGPR Distribution: (uniform subsample, no grad-optim)
ELBO - max: -102.169119, min: -111.343948, median: -105.268686, mean: -106.147500, std: 2.976103.
ACC  - max: 0.925926, min: 0.777778, median: 0.796296, mean: 0.825926, std: 0.057497.
NLL  - max: 0.546463, min: 0.190650, median: 0.435317, mean: 0.395260, std: 0.114083.

PGPR Distribution: (k_means, no grad-optim)
ELBO - max: -102.079243, min: -111.201320, median: -105.027989, mean: -105.979876, std: 2.972753.
ACC  - max: 0.925926, min: 0.777778, median: 0.796296, mean: 0.825926, std: 0.057497.
NLL  - max: 0.546634, min: 0.190929, median: 0.434555, mean: 0.394946, std: 0.113989.

PGPR Distribution: (greedy var, no grad-optim)
ELBO - max: -100.655929, min: -108.262102, median: -106.784653, mean: -106.207520, std: 2.123873.
ACC  - max: 0.925926, min: 0.703704, median: 0.870370, mean: 0.855556, std: 0.058443.
NLL  - max: 0.621502, min: 0.311186, median: 0.371856, mean: 0.389818, std: 0.087791.

###################################
#####  Gradient Optimisation  #####
###################################

SVGP Distribution: (k_means, *with* grad-optim)
ELBO - max: -98.528924, min: -146.187904, median: -103.060142, mean: -113.977464, std: 18.506504.
ACC  - max: 0.925926, min: 0.740741, median: 0.814815, mean: 0.811111, std: 0.056047.
NLL  - max: 0.551764, min: 0.224742, median: 0.448642, mean: 0.434787, std: 0.096945.

PGPR Distribution: (k_means, *with* grad-optim)
ELBO - max: -102.257948, min: -109.816611, median: -105.613186, mean: -105.632307, std: 1.968729.
ACC  - max: 0.925926, min: 0.740741, median: 0.833333, mean: 0.822222, std: 0.059259.
NLL  - max: 0.525609, min: 0.235252, median: 0.401161, mean: 0.395334, std: 0.075484.
"""


class HeartMetricsMetaDataset(HeartDataset, MetricsMetaDataset):
    def __init__(self):
        HeartDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 35, 250, 10, 250, 10)


"""
SVGP Distribution: (k_means, no grad-optim)
ELBO - max: -103.563473, min: -112.047841, median: -107.209806, mean: -106.996660, std: 2.520888.
ACC  - max: 0.944444, min: 0.833333, median: 0.902778, mean: 0.891667, std: 0.033907.
NLL  - max: 0.444872, min: 0.204699, median: 0.290666, mean: 0.298987, std: 0.081673.

PGPR Distribution: (hetero greedy var, no grad-optim)
ELBO - max: -116.201476, min: -123.996954, median: -120.696971, mean: -120.516805, std: 2.243119.
ACC  - max: 0.944444, min: 0.861111, median: 0.902778, mean: 0.894444, std: 0.029918.
NLL  - max: 0.443399, min: 0.226932, median: 0.299854, mean: 0.312216, std: 0.076521.

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

###################################
#####  Gradient Optimisation  #####
###################################

SVGP Distribution: (k_means, *with* grad-optim)
ELBO - max: -93.733839, min: -99.595373, median: -97.140276, mean: -96.838896, std: 1.625935.
ACC  - max: 0.972222, min: 0.888889, median: 0.930556, mean: 0.936111, std: 0.027916.
NLL  - max: 0.353865, min: 0.072665, median: 0.187099, mean: 0.191554, std: 0.085631.

PGPR Distribution: (k_means, *with* grad-optim)
ELBO - max: -115.060499, min: -122.053566, median: -119.387124, mean: -119.255531, std: 1.717693.
ACC  - max: 0.972222, min: 0.861111, median: 0.916667, mean: 0.913889, std: 0.033907.
NLL  - max: 0.440378, min: 0.154491, median: 0.228702, mean: 0.251751, std: 0.080864.
"""


class IonosphereMetricsMetaDataset(IonosphereDataset, MetricsMetaDataset):
    def __init__(self):
        IonosphereDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 50, 500, 20, 500, 20)


"""
SVGP Distribution: (k_means, no grad-optim)
ELBO - max: -51.035899, min: -260.193332, median: -79.258723, mean: -128.164729, std: 87.821823.
ACC  - max: 1.000000, min: 0.877193, median: 0.956140, mean: 0.954386, std: 0.034379.
NLL  - max: 0.438799, min: 0.035690, median: 0.147612, mean: 0.198864, std: 0.145042.

PGPR Distribution: (hetero greedy var, no grad-optim)
ELBO - max: -65.063036, min: -71.668893, median: -69.381602, mean: -69.548874, std: 1.810755.
ACC  - max: 1.000000, min: 0.947368, median: 0.982456, mean: 0.980702, std: 0.014573.
NLL  - max: 0.156879, min: 0.022692, median: 0.085697, mean: 0.082185, std: 0.033648.

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

###################################
#####  Gradient Optimisation  #####
###################################

SVGP Distribution: (k_means, *with* grad-optim)
ELBO - max: -49.448369, min: -60.149434, median: -55.705711, mean: -55.182518, std: 2.716737.
ACC  - max: 1.000000, min: 0.947368, median: 0.982456, mean: 0.977193, std: 0.015789.
NLL  - max: 0.226910, min: 0.014528, median: 0.074933, mean: 0.093823, std: 0.061250.

PGPR Distribution: (k_means, *with* grad-optim)
ELBO - max: -70.212348, min: -73.540194, median: -71.780424, mean: -71.690173, std: 1.057898.
ACC  - max: 1.000000, min: 0.982456, median: 0.982456, mean: 0.989474, std: 0.008595.
NLL  - max: 0.089640, min: 0.030987, median: 0.065878, mean: 0.060539, std: 0.016552.
"""


class BreastCancerMetricsMetaDataset(BreastCancerDataset, MetricsMetaDataset):
    def __init__(self):
        BreastCancerDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 50, 500, 10, 250, 10)


"""
SVGP Distribution: (k_means, no grad-optim)
ELBO - max: -335.880773, min: -346.814701, median: -342.072790, mean: -341.638486, std: 3.397007.
ACC  - max: 0.844156, min: 0.714286, median: 0.785714, mean: 0.780519, std: 0.035065.
NLL  - max: 0.522449, min: 0.364078, median: 0.443357, mean: 0.441000, std: 0.047642.

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

###################################
#####  Gradient Optimisation  #####
###################################

SVGP Distribution: (k_means, *with* grad-optim)
ELBO - max: -333.573561, min: -346.306158, median: -338.796763, mean: -339.442382, std: 3.397075.
ACC  - max: 0.844156, min: 0.701299, median: 0.779221, mean: 0.784416, std: 0.038613.
NLL  - max: 0.513809, min: 0.356909, median: 0.447968, mean: 0.439688, std: 0.041391.

PGPR Distribution: (k_means, *with* grad-optim)
ELBO - max: -338.361598, min: -350.558833, median: -342.947328, mean: -343.674984, std: 3.292003.
ACC  - max: 0.844156, min: 0.701299, median: 0.785714, mean: 0.785714, std: 0.037752.
NLL  - max: 0.507673, min: 0.357157, median: 0.450626, mean: 0.440636, std: 0.040233.
"""


class PimaMetricsMetaDataset(PimaDataset, MetricsMetaDataset):
    def __init__(self):
        PimaDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 60, 500, 10, 250, 10)


"""
SVGP Distribution: (k_means, no grad-optim)
ELBO - max: -440.252890, min: -4480.603436, median: -4454.929089, mean: -3581.029505, std: 1314.972465.
ACC  - max: 0.982432, min: 0.810811, median: 0.968243, mean: 0.943649, std: 0.057986.
NLL  - max: 0.667848, min: 0.054345, median: 0.663046, mean: 0.516713, std: 0.204524.

PGPR Distribution: (hetero greedy var, no grad-optim)
ELBO - max: -441.419710, min: -464.018613, median: -452.083285, mean: -451.609530, std: 7.613344.
ACC  - max: 0.986486, min: 0.972973, median: 0.979730, mean: 0.979324, std: 0.004799.
NLL  - max: 0.075884, min: 0.043305, median: 0.052011, mean: 0.054395, std: 0.010713.

PGPR Distribution: (uniform subsample, no grad-optim)
ELBO - max: -431.669276, min: -467.890859, median: -455.823611, mean: -455.121204, std: 9.190612.
ACC  - max: 0.986486, min: 0.967568, median: 0.977703, mean: 0.978243, std: 0.005036.
NLL  - max: 0.091890, min: 0.044199, median: 0.060317, mean: 0.060964, std: 0.012172.

PGPR Distribution: (k_means, no grad-optim)
ELBO - max: -431.688747, min: -467.903019, median: -455.840175, mean: -455.136209, std: 9.188429.
ACC  - max: 0.986486, min: 0.967568, median: 0.977703, mean: 0.978243, std: 0.005036.
NLL  - max: 0.091879, min: 0.044205, median: 0.060318, mean: 0.060964, std: 0.012168.

PGPR Distribution: (greedy var, no grad-optim)
ELBO - max: -444.943399, min: -465.351050, median: -458.610668, mean: -457.373139, std: 6.188278.
ACC  - max: 0.983784, min: 0.972973, median: 0.978378, mean: 0.977703, std: 0.003928.
NLL  - max: 0.073826, min: 0.045847, median: 0.054184, mean: 0.056054, std: 0.008707.

###################################
#####  Gradient Optimisation  #####
###################################

SVGP Distribution: (k_means, *with* grad-optim)
ELBO - max: -4064.002824, min: -4501.340313, median: -4471.074616, mean: -4402.973363, std: 151.044773.
ACC  - max: 0.974324, min: 0.860811, median: 0.966216, mean: 0.948784, std: 0.034882.
NLL  - max: 0.671367, min: 0.486228, median: 0.666746, mean: 0.634811, std: 0.065158.

PGPR Distribution: (k_means, *with* grad-optim)
ELBO - max: -436.050463, min: -471.809093, median: -459.235116, mean: -456.351689, std: 12.267362.
ACC  - max: 0.989189, min: 0.968919, median: 0.977703, mean: 0.977838, std: 0.005834.
NLL  - max: 0.087823, min: 0.040188, median: 0.057173, mean: 0.060929, std: 0.016044.
"""


class TwonormMetricsMetaDataset(TwonormDataset, MetricsMetaDataset):
    def __init__(self):
        TwonormDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 300, 500, 10, 250, 10)


"""
SVGP Distribution: (k_means, no grad-optim)
ELBO - max: -768.285139, min: -4332.227797, median: -2597.246881, mean: -2260.877601, std: 1301.823368.
ACC  - max: 0.979730, min: 0.495946, median: 0.871622, mean: 0.803243, std: 0.183625.
NLL  - max: 0.646989, min: 0.043059, median: 0.315875, mean: 0.290692, std: 0.209444.

PGPR Distribution: (hetero greedy var, no grad-optim)
ELBO - max: -933.852979, min: -967.672191, median: -952.890675, mean: -953.277250, std: 9.537135.
ACC  - max: 0.989189, min: 0.964865, median: 0.980405, mean: 0.976622, std: 0.007023.
NLL  - max: 0.097374, min: 0.037707, median: 0.054798, mean: 0.062037, std: 0.015969.

PGPR Distribution: (uniform subsample, no grad-optim)
ELBO - max: -932.467899, min: -980.560829, median: -960.690189, mean: -959.429487, std: 15.226527.
ACC  - max: 0.987838, min: 0.967568, median: 0.975676, mean: 0.977162, std: 0.005935.
NLL  - max: 0.082345, min: 0.048434, median: 0.070451, mean: 0.066714, std: 0.012016.

PGPR Distribution: (k_means, no grad-optim)
ELBO - max: -933.976157, min: -971.332302, median: -954.930721, mean: -956.432142, std: 10.902029.
ACC  - max: 0.987838, min: 0.968919, median: 0.975676, mean: 0.977162, std: 0.005873.
NLL  - max: 0.082839, min: 0.048921, median: 0.070721, mean: 0.067049, std: 0.011971.

PGPR Distribution: (greedy var, no grad-optim)
ELBO - max: -926.199076, min: -987.939713, median: -965.576319, mean: -958.984452, std: 19.976217.
ACC  - max: 0.983784, min: 0.970270, median: 0.973649, mean: 0.975676, std: 0.004563.
NLL  - max: 0.088039, min: 0.052046, median: 0.066332, mean: 0.067730, std: 0.011997.

###################################
#####  Gradient Optimisation  #####
###################################

SVGP Distribution: (k_means, *with* grad-optim)
ELBO - max: -717.257938, min: -3711.059627, median: -739.043615, mean: -1321.611076, std: 1179.373818.
ACC  - max: 0.985135, min: 0.554054, median: 0.977703, mean: 0.898784, std: 0.162212.
NLL  - max: 0.518785, min: 0.045849, median: 0.059915, mean: 0.148665, std: 0.184027.

PGPR Distribution: (k_means, *with* grad-optim)
ELBO - max: -1045.652322, min: -1133.138370, median: -1119.397630, mean: -1101.003295, std: 31.811725.
ACC  - max: 0.987838, min: 0.970270, median: 0.977027, mean: 0.978649, std: 0.005399.
NLL  - max: 0.080699, min: 0.060410, median: 0.072017, mean: 0.069876, std: 0.006767.
"""


class RingnormMetricsMetaDataset(RingnormDataset, MetricsMetaDataset):
    def __init__(self):
        RingnormDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 300, 500, 10, 250, 10)


"""
SVGP Distribution: (k_means, no grad-optim)
ELBO - max: -5699.377250, min: -5777.605244, median: -5737.342847, mean: -5736.178598, std: 24.040027.
ACC  - max: 0.879075, min: 0.856467, median: 0.871188, mean: 0.869874, std: 0.006380.
NLL  - max: 0.335494, min: 0.296170, median: 0.316794, mean: 0.319020, std: 0.010461.

PGPR Distribution: (hetero greedy var, no grad-optim)
ELBO - max: -5801.452425, min: -5868.467288, median: -5828.071268, mean: -5831.008173, std: 19.699746.
ACC  - max: 0.882229, min: 0.862250, median: 0.873800, mean: 0.872347, std: 0.005634.
NLL  - max: 0.353511, min: 0.300778, median: 0.324061, mean: 0.324389, std: 0.013510.

PGPR Distribution: (uniform subsample, no grad-optim)
ELBO - max: -5907.587163, min: -5980.157311, median: -5935.344158, mean: -5936.948171, std: 24.630645.
ACC  - max: 0.876972, min: 0.860147, median: 0.867508, mean: 0.867666, std: 0.005520.
NLL  - max: 0.339632, min: 0.300164, median: 0.330803, mean: 0.326025, std: 0.013762.

PGPR Distribution: (k_means, no grad-optim)
ELBO - max: -5835.694997, min: -5895.958561, median: -5845.246655, mean: -5854.735864, std: 21.176794.
ACC  - max: 0.876972, min: 0.856993, median: 0.869348, mean: 0.868980, std: 0.005971.
NLL  - max: 0.335551, min: 0.299321, median: 0.326909, mean: 0.322658, std: 0.012951.

PGPR Distribution: (greedy var, no grad-optim)
ELBO - max: -5839.504091, min: -5894.213351, median: -5859.701456, mean: -5864.279534, std: 17.867179.
ACC  - max: 0.877497, min: 0.864879, median: 0.869085, mean: 0.870137, std: 0.003948.
NLL  - max: 0.346354, min: 0.309627, median: 0.331679, mean: 0.328185, std: 0.011733.

###################################
#####  Gradient Optimisation  #####
###################################

SVGP Distribution: (k_means, *with* grad-optim)
...

PGPR Distribution: (k_means, *with* grad-optim)
...
"""


class MagicMetricsMetaDataset(MagicDataset, MetricsMetaDataset):
    def __init__(self):
        MagicDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 300, 500, 10, 250, 10)


"""
SVGP Distribution: (k_means, no grad-optim)
ELBO - max: -18063.550467, min: -18178.405298, median: -18109.695571, mean: -18114.706985, std: 37.360415.
ACC  - max: 0.813327, min: 0.793910, median: 0.805053, mean: 0.803928, std: 0.005759.
NLL  - max: 0.440693, min: 0.413341, median: 0.422907, mean: 0.424236, std: 0.007474.

PGPR Distribution: (hetero greedy var, no grad-optim)
ELBO - max: -18308.170504, min: -18394.747920, median: -18357.596046, mean: -18356.855284, std: 29.821048.
ACC  - max: 0.806267, min: 0.793689, median: 0.803949, mean: 0.802111, std: 0.004401.
NLL  - max: 0.451750, min: 0.425104, median: 0.429982, mean: 0.433322, std: 0.009202.

PGPR Distribution: (uniform subsample, no grad-optim)
ELBO - max: -18284.592037, min: -18401.895847, median: -18336.620948, mean: -18341.835006, std: 38.983403.
ACC  - max: 0.805605, min: 0.796558, median: 0.799426, mean: 0.800044, std: 0.002973.
NLL  - max: 0.444850, min: 0.424833, median: 0.435564, mean: 0.434925, std: 0.005099.

PGPR Distribution: (k_means, no grad-optim)
ELBO - max: -18149.154007, min: -18284.339341, median: -18208.751162, mean: -18208.193579, std: 36.173418.
ACC  - max: 0.809576, min: 0.795455, median: 0.800971, mean: 0.801324, std: 0.004084.
NLL  - max: 0.442121, min: 0.423484, median: 0.433032, mean: 0.431940, std: 0.004992.

PGPR Distribution: (greedy var, no grad-optim)
ELBO - max: -18282.290839, min: -18422.084934, median: -18389.020222, mean: -18372.674480, std: 44.600077.
ACC  - max: 0.803177, min: 0.785525, median: 0.793358, mean: 0.794219, std: 0.004900.
NLL  - max: 0.455802, min: 0.434493, median: 0.441399, mean: 0.442115, std: 0.006935.

###################################
#####  Gradient Optimisation  #####
###################################

SVGP Distribution: (k_means, *with* grad-optim)
ELBO - max: -17440.357161, min: -17586.207270, median: -17523.674322, mean: -17514.699011, std: 47.354226.
ACC  - max: 0.817741, min: 0.803619, median: 0.813548, mean: 0.811751, std: 0.004558.
NLL  - max: 0.422247, min: 0.399283, median: 0.406280, mean: 0.408225, std: 0.007383.

PGPR Distribution: (k_means, *with* grad-optim)
ELBO - max: -17620.673601, min: -17736.029735, median: -17679.623742, mean: -17685.187954, std: 38.838911.
ACC  - max: 0.819285, min: 0.801192, median: 0.813548, mean: 0.811815, std: 0.004433.
NLL  - max: 0.421578, min: 0.399762, median: 0.405035, mean: 0.407326, std: 0.006516.
"""


class ElectricityMetricsMetaDataset(ElectricityDataset, MetricsMetaDataset):
    def __init__(self):
        ElectricityDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 300, 500, 10, 250, 10)
