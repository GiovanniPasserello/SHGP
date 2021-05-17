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
SVGP Distribution: (kmeans++, no grad-optim, with unconstrained/default)
ELBO - max: -89.519647, min: -103.250001, median: -98.954888, mean: -98.629271, std: 3.787107.
ACC  - max: 0.975000, min: 0.825000, median: 0.912500, mean: 0.902500, std: 0.039449.
NLL  - max: 0.538161, min: 0.061780, median: 0.202625, mean: 0.222576, std: 0.123889.

PGPR Distribution: (hetero greedy var, no grad-optim, with unconstrained/default)
ELBO - max: -103.146880, min: -115.827680, median: -110.754296, mean: -110.913423, std: 3.235203.
ACC  - max: 1.000000, min: 0.825000, median: 0.925000, mean: 0.907500, std: 0.044791.
NLL  - max: 0.457047, min: 0.069402, median: 0.215834, mean: 0.216745, std: 0.097789.
"""


class BananaMetricsMetaDataset(BananaDataset, MetricsMetaDataset):
    def __init__(self):
        BananaDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 40, 250, 10, 250, 10)


"""
SVGP Distribution: (kmeans++, no grad-optim, with unconstrained/default)
ELBO - max: -22.581493, min: -54.522554, median: -29.843940, mean: -32.031536, std: 8.377061.
ACC  - max: 1.000000, min: 1.000000, median: 1.000000, mean: 1.000000, std: 0.000000.
NLL  - max: 0.104137, min: 0.001727, median: 0.021329, mean: 0.029044, std: 0.028902.

PGPR Distribution: (hetero greedy var, no grad-optim, with unconstrained/default)
ELBO - max: -29.895069, min: -30.130667, median: -29.984135, mean: -30.000083, std: 0.078844.
ACC  - max: 1.000000, min: 1.000000, median: 1.000000, mean: 1.000000, std: 0.000000.
NLL  - max: 0.030897, min: 0.003004, median: 0.008187, mean: 0.011842, std: 0.009417.
"""


# TODO: Sparsity experiment
class CrabsMetricsMetaDataset(CrabsDataset, MetricsMetaDataset):
    # Bernoulli requires large number of training iters for this dataset
    def __init__(self):
        CrabsDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 10, 1000, 10, 250, 10)


"""
SVGP Distribution: (kmeans++, no grad-optim, with unconstrained/default)
ELBO - max: -102.457024, min: -147.645443, median: -105.172674, mean: -116.764816, std: 19.141206.
ACC  - max: 0.888889, min: 0.740741, median: 0.814815, mean: 0.825926, std: 0.037222.
NLL  - max: 0.541615, min: 0.234183, median: 0.394648, mean: 0.394743, std: 0.095801.

PGPR Distribution: (hetero greedy var, no grad-optim, with unconstrained/default)
ELBO - max: -105.712986, min: -109.829346, median: -107.190899, mean: -107.372922, std: 1.172509.
ACC  - max: 0.888889, min: 0.814815, median: 0.851852, mean: 0.844444, std: 0.027716.
NLL  - max: 0.424670, min: 0.240716, median: 0.347886, mean: 0.347235, std: 0.050955.
"""


class HeartMetricsMetaDataset(HeartDataset, MetricsMetaDataset):
    def __init__(self):
        HeartDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 35, 250, 10, 250, 10)


""" Most likely use M=30 to show the benefits of sparse PGPR
SVGP Distribution: (kmeans++, no grad-optim, with unconstrained/default, M=30)
ELBO - max: -107.877620, min: -187.498914, median: -115.158968, mean: -121.753272, std: 22.224797.
ACC  - max: 0.972222, min: 0.611111, median: 0.875000, mean: 0.847222, std: 0.096425.
NLL  - max: 0.581576, min: 0.175762, median: 0.285526, mean: 0.334639, std: 0.137892.

SVGP Distribution: (kmeans++, no grad-optim, with unconstrained/default, M=50)
ELBO - max: -102.320602, min: -111.255612, median: -107.341194, mean: -107.392428, std: 2.878623.
ACC  - max: 0.972222, min: 0.861111, median: 0.888889, mean: 0.911111, std: 0.042673.
NLL  - max: 0.539788, min: 0.087637, median: 0.269310, mean: 0.276174, std: 0.127492.

PGPR Distribution: (hetero greedy var, no grad-optim, with unconstrained/default, M=30)
ELBO - max: -119.951628, min: -131.741912, median: -126.352503, mean: -125.994329, std: 3.639924.
ACC  - max: 0.972222, min: 0.750000, median: 0.888889, mean: 0.883333, std: 0.059317.
NLL  - max: 0.493626, min: 0.176245, median: 0.302106, mean: 0.317360, std: 0.106654.

PGPR Distribution: (hetero greedy var, no grad-optim, with unconstrained/default, M=50)
ELBO - max: -116.037903, min: -125.215633, median: -120.372931, mean: -120.675359, std: 2.760629.
ACC  - max: 0.972222, min: 0.833333, median: 0.861111, mean: 0.886111, std: 0.042035.
NLL  - max: 0.502648, min: 0.121410, median: 0.301423, mean: 0.307200, std: 0.112117.
"""


# TODO: Sparsity experiment
class IonosphereMetricsMetaDataset(IonosphereDataset, MetricsMetaDataset):
    def __init__(self):
        IonosphereDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 30, 500, 20, 500, 20)


"""
SVGP Distribution: (kmeans++, no grad-optim, with unconstrained/default)
ELBO - max: -51.035899, min: -260.193332, median: -79.258723, mean: -128.164729, std: 87.821823.
ACC  - max: 1.000000, min: 0.877193, median: 0.956140, mean: 0.954386, std: 0.034379.
NLL  - max: 0.438799, min: 0.035690, median: 0.147612, mean: 0.198864, std: 0.145042.

SVGP Distribution: (kmeans++, with grad-optim, with unconstrained/default)
ELBO - max: -49.402746, min: -263.220134, median: -55.021090, mean: -75.629275, std: 62.565196.
ACC  - max: 1.000000, min: 0.807018, median: 0.982456, mean: 0.963158, std: 0.053473.
NLL  - max: 0.436694, min: 0.038113, median: 0.088166, mean: 0.121176, std: 0.110232.

PGPR Distribution: (hetero greedy var, no grad-optim, with unconstrained/default)
ELBO - max: -65.263036, min: -71.668893, median: -69.381602, mean: -69.548874, std: 1.810755.
ACC  - max: 1.000000, min: 0.947368, median: 0.982456, mean: 0.980702, std: 0.014573.
NLL  - max: 0.156879, min: 0.037506, median: 0.085697, mean: 0.082185, std: 0.033648.
"""


class BreastCancerMetricsMetaDataset(BreastCancerDataset, MetricsMetaDataset):
    def __init__(self):
        BreastCancerDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 50, 500, 10, 250, 10)


"""
SVGP Distribution: (kmeans++, no grad-optim, with unconstrained/default)
ELBO - max: -335.880773, min: -346.814701, median: -342.072790, mean: -341.638486, std: 3.397007.
ACC  - max: 0.844156, min: 0.714286, median: 0.785714, mean: 0.780519, std: 0.035065.
NLL  - max: 0.522449, min: 0.364078, median: 0.443357, mean: 0.441000, std: 0.047642.

PGPR Distribution: (hetero greedy var, no grad-optim, with unconstrained/default)
ELBO - max: -339.673734, min: -350.845679, median: -345.761043, mean: -345.529780, std: 3.339615.
ACC  - max: 0.857143, min: 0.727273, median: 0.785714, mean: 0.789610, std: 0.032233.
NLL  - max: 0.518303, min: 0.370971, median: 0.440707, mean: 0.441000, std: 0.044599.
"""


# TODO: Sparsity experiment
class PimaMetricsMetaDataset(PimaDataset, MetricsMetaDataset):
    def __init__(self):
        PimaDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 60, 500, 10, 250, 10)


"""
SVGP Distribution: (kmeans++, no grad-optim, with unconstrained/default, M=100)
ELBO - max: -435.307312, min: -4550.765233, median: -4132.715446, mean: -2949.644103, std: 1793.864148.
ACC  - max: 0.979730, min: 0.956757, median: 0.970946, mean: 0.969324, std: 0.007452.
NLL  - max: 0.679808, min: 0.062394, median: 0.526168, mean: 0.398681, std: 0.269309.

PGPR Distribution: (hetero greedy var, no grad-optim, with unconstrained/default, M=100)
ELBO - max: -449.189194, min: -4616.360228, median: -4616.360228, mean: -3784.142286, std: 1664.438106.
ACC  - max: 0.985135, min: 0.902703, median: 0.933108, mean: 0.938243, std: 0.024764.
NLL  - max: 0.693147, min: 0.050044, median: 0.693147, mean: 0.566307, std: 0.253713.
"""

# (10, 100, 500, 10, 250, 10)
# SVGP: ELBO = -520.542760, ACC = 0.975676, NLL = 0.062394.  # 1
# PGPR: ELBO = -4616.360228, ACC = 0.927027, NLL = 0.693147.
# SVGP: ELBO = -437.738622, ACC = 0.979730, NLL = 0.067435.  # 2
# PGPR: ELBO = -449.189194, ACC = 0.979730, NLL = 0.067844.
# SVGP: ELBO = -4548.879012, ACC = 0.977027, NLL = 0.679808. # 3
# PGPR: ELBO = -461.351842, ACC = 0.985135, NLL = 0.050044.
# SVGP: ELBO = -1843.197768, ACC = 0.971622, NLL = 0.102737. # 4
# PGPR: ELBO = -4616.360228, ACC = 0.902703, NLL = 0.693147.
# SVGP: ELBO = -4219.859002, ACC = 0.959459, NLL = 0.540904. # 5
# PGPR: ELBO = -4616.360228, ACC = 0.940541, NLL = 0.693147.


# TODO: Need to finish - maybe try on Colab?
# TODO: Try 200 - keep experimenting
# TODO: Try 100 again, but with (10, 100, 500, 20, 500, 20)!!!!! -> important one to try!
# M=150 works quite well
# With a large number of inducing points, we are prone to inversion/cholesky errors
# This is something to look into in future work.
class TwonormMetricsMetaDataset(TwonormDataset, MetricsMetaDataset):
    def __init__(self):
        TwonormDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 150, 500, 10, 250, 10)


"""
SVGP Distribution: (kmeans++, no grad-optim, with unconstrained/default, M=100)
ELBO - max: -1343.326952, min: -3853.552475, median: -1663.819734, mean: -2132.205899, std: 897.843589.
ACC  - max: 0.959459, min: 0.508108, median: 0.946622, mean: 0.852703, std: 0.168002.
NLL  - max: 0.567328, min: 0.111391, median: 0.154397, mean: 0.244232, std: 0.156691.

PGPR Distribution: (hetero greedy var, no grad-optim, with unconstrained/default, M=100)
ELBO - max: -1915.354700, min: -4616.360228, median: -2110.011878, mean: -2342.593651, std: 763.174001.
ACC  - max: 0.968919, min: 0.822973, median: 0.943919, mean: 0.933243, std: 0.038046.
NLL  - max: 0.693147, min: 0.117649, median: 0.173071, mean: 0.221537, std: 0.159302.
"""

# (10, 100, 500, 10, 250, 10) -> ~4mins per iteration
# SVGP: ELBO = -1343.326952, ACC = 0.956757, NLL = 0.111391.
# PGPR: ELBO = -1944.491960, ACC = 0.966216, NLL = 0.125356.

# (10, 100, 1000, 20, 500, 20) -> ~8mins per iteration
# We observe that SVGP is prone to catastrophic failure, whereas PGPR is much more stable.
# SVGP: ELBO = -1343.326952, ACC = 0.956757, NLL = 0.111391. # 1
# PGPR: ELBO = -1915.354700, ACC = 0.968919, NLL = 0.117649.
# SVGP: ELBO = -1398.368509, ACC = 0.944595, NLL = 0.147709. # 2
# PGPR: ELBO = -2069.113872, ACC = 0.944595, NLL = 0.168655.
# SVGP: ELBO = -1344.080831, ACC = 0.948649, NLL = 0.128455. # 3
# PGPR: ELBO = -1981.906180, ACC = 0.943243, NLL = 0.154874.
# SVGP: ELBO = -1470.956913, ACC = 0.954054, NLL = 0.138897. # 4
# PGPR: ELBO = -2160.896554, ACC = 0.936486, NLL = 0.206544.
# SVGP: ELBO = -2238.879201, ACC = 0.906757, NLL = 0.235299. # 5
# PGPR: ELBO = -2058.936326, ACC = 0.954054, NLL = 0.145316.
# SVGP: ELBO = -3853.552475, ACC = 0.508108, NLL = 0.567328. # 6
# PGPR: ELBO = -2073.280321, ACC = 0.944595, NLL = 0.156893.
# SVGP: ELBO = -3486.709797, ACC = 0.535135, NLL = 0.485103. # 7
# PGPR: ELBO = -2164.046119, ACC = 0.935135, NLL = 0.198798.
# SVGP: ELBO = -1481.393108, ACC = 0.959459, NLL = 0.123016. # 8
# PGPR: ELBO = -4616.360228, ACC = 0.822973, NLL = 0.693147.
# SVGP: ELBO = -1846.246360, ACC = 0.950000, NLL = 0.161084. # 9
# PGPR: ELBO = -2146.743435, ACC = 0.948649, NLL = 0.177487.

# (10, 100, 1000, 20, 500, 20)
# ... (running overnight)


# TODO: Investigate various M here (try M=200, or higher if using Colab?)
# TODO: Try 500 for a few iterations to see if there's any significant improvement?
# TODO: Try standard greedy variance with (10, 100, 500, 20, 500, 20)
# TODO: Best is (10, 100, 500, 20, 500, 20)
class RingnormMetricsMetaDataset(RingnormDataset, MetricsMetaDataset):
    def __init__(self):
        RingnormDataset.__init__(self)  # TODO: Change back to M=100?
        MetricsMetaDataset.__init__(self, 10, 200, 500, 20, 500, 20)
