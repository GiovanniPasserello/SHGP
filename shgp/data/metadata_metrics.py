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
SVGP Distribution: (kmeans++, no grad-optim, with unconstrained/default, M=300)
ELBO - max: -3525.264918, min: -4500.681519, median: -4460.407540, mean: -4194.907578, std: 425.883111.
ACC  - max: 0.974324, min: 0.936486, median: 0.961486, mean: 0.959459, std: 0.012834.
NLL  - max: 0.671031, min: 0.397189, median: 0.662346, mean: 0.595052, std: 0.108684.

PGPR Distribution: (hetero greedy var, no grad-optim, with unconstrained/default, M=300)
ELBO - max: -441.419710, min: -464.018613, median: -452.083285, mean: -451.609530, std: 7.613344.
ACC  - max: 0.985135, min: 0.972973, median: 0.975676, mean: 0.977838, std: 0.004196.
NLL  - max: 0.077174, min: 0.046878, median: 0.062856, mean: 0.063841, std: 0.010713.
"""

# (10, 300, 500, 10, 250, 10)
# SVGP: ELBO = -4434.778278, ACC = 0.972973, NLL = 0.657662. # 1
# PGPR: ELBO = -452.927107, ACC = 0.974324, NLL = 0.062034.
# SVGP: ELBO = -3579.396447, ACC = 0.936486, NLL = 0.397189. # 2
# PGPR: ELBO = -458.741920, ACC = 0.982432, NLL = 0.053706.
# SVGP: ELBO = -4463.158650, ACC = 0.964865, NLL = 0.663630. # 3
# PGPR: ELBO = -444.264951, ACC = 0.974324, NLL = 0.075936.
# SVGP: ELBO = -4473.513536, ACC = 0.958108, NLL = 0.666257. # 4
# PGPR: ELBO = -459.084314, ACC = 0.982432, NLL = 0.053327.
# SVGP: ELBO = -4482.821112, ACC = 0.956757, NLL = 0.669868. # 5
# PGPR: ELBO = -441.419710, ACC = 0.975676, NLL = 0.077174.
# SVGP: ELBO = -4500.681519, ACC = 0.937838, NLL = 0.671031. # 6
# PGPR: ELBO = -445.391294, ACC = 0.975676, NLL = 0.073069.
# SVGP: ELBO = -4457.656429, ACC = 0.974324, NLL = 0.661061. # 7
# PGPR: ELBO = -456.977911, ACC = 0.985135, NLL = 0.055768.
# SVGP: ELBO = -4500.596467, ACC = 0.955405, NLL = 0.670082. # 8
# PGPR: ELBO = -442.030012, ACC = 0.974324, NLL = 0.076842.
# SVGP: ELBO = -3531.208425, ACC = 0.966216, NLL = 0.445102. # 9
# PGPR: ELBO = -464.018613, ACC = 0.981081, NLL = 0.046878.
# SVGP: ELBO = -3525.264918, ACC = 0.971622, NLL = 0.448636. # 10
# PGPR: ELBO = -451.239463, ACC = 0.972973, NLL = 0.063677.


class TwonormMetricsMetaDataset(TwonormDataset, MetricsMetaDataset):
    def __init__(self):
        TwonormDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 300, 500, 10, 250, 10)


"""
SVGP Distribution: (kmeans++, no grad-optim, with unconstrained/default, M=300)
ELBO - max: -760.377360, min: -3527.675610, median: -2422.280729, mean: -2111.355175, std: 1037.510762.
ACC  - max: 0.983784, min: 0.493243, median: 0.912162, mean: 0.857703, std: 0.156352.
NLL  - max: 0.512994, min: 0.046621, median: 0.238337, mean: 0.239820, std: 0.168667.

PGPR Distribution: (hetero greedy var, no grad-optim, with unconstrained/default, M=300)
ELBO - max: -948.141417, min: -4616.360228, median: -959.493653, mean: -1688.341512, std: 1464.016603.
ACC  - max: 0.985135, min: 0.877027, median: 0.980405, mean: 0.959595, std: 0.041208.
NLL  - max: 0.693147, min: 0.046114, median: 0.054798, mean: 0.183520, std: 0.254989.
"""

# (10, 300, 500, 10, 250, 10)
# SVGP: ELBO = -2701.769081, ACC = 0.874324, NLL = 0.315759. # 1
# PGPR: ELBO = -963.796712, ACC = 0.981081, NLL = 0.048220.
# SVGP: ELBO = -3527.675610, ACC = 0.493243, NLL = 0.512994. # 2
# PGPR: ELBO = -4616.360228, ACC = 0.878378, NLL = 0.693147.
# SVGP: ELBO = -1325.857785, ACC = 0.981081, NLL = 0.077672. # 3
# PGPR: ELBO = -960.825776, ACC = 0.983784, NLL = 0.046114.
# SVGP: ELBO = -801.330255, ACC = 0.983784, NLL = 0.046621.  # 4
# PGPR: ELBO = -960.507233, ACC = 0.985135, NLL = 0.047909.
# SVGP: ELBO = -2142.792377, ACC = 0.950000, NLL = 0.160916. # 5
# PGPR: ELBO = -953.951669, ACC = 0.985135, NLL = 0.053338.
# SVGP: ELBO = -3120.459927, ACC = 0.660811, NLL = 0.440570. # 6
# PGPR: ELBO = -4616.360228, ACC = 0.877027, NLL = 0.693147.
# SVGP: ELBO = -3039.894458, ACC = 0.801351, NLL = 0.366579. # 7
# PGPR: ELBO = -954.944517, ACC = 0.982432, NLL = 0.056259.
# SVGP: ELBO = -760.377360, ACC = 0.974324, NLL = 0.073831.  # 8
# PGPR: ELBO = -950.047262, ACC = 0.971622, NLL = 0.075348.
# SVGP: ELBO = -2902.601762, ACC = 0.874324, NLL = 0.355600. # 9
# PGPR: ELBO = -948.141417, ACC = 0.971622, NLL = 0.071920.
# SVGP: ELBO = -790.793133, ACC = 0.983784, NLL = 0.047664.  # 10
# PGPR: ELBO = -958.480074, ACC = 0.979730, NLL = 0.049802.


class RingnormMetricsMetaDataset(RingnormDataset, MetricsMetaDataset):
    def __init__(self):
        RingnormDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 300, 500, 10, 250, 10)


"""
SVGP Distribution: (kmeans++, no grad-optim, with unconstrained/default, M=300)

PGPR Distribution: (hetero greedy var, no grad-optim, with unconstrained/default, M=300)
"""

# (10, 200, 500, 20, 500, 20)


# TODO: Needs to run on Colab
class MagicMetricsMetaDataset(MagicDataset, MetricsMetaDataset):
    def __init__(self):
        MagicDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 200, 500, 20, 500, 20)


"""
SVGP Distribution: (kmeans++, no grad-optim, with unconstrained/default, M=300)

PGPR Distribution: (hetero greedy var, no grad-optim, with unconstrained/default, M=300)
"""

# (10, 200, 500, 20, 500, 20)
# ...


# TODO: Needs to run on Colab
class ElectricityMetricsMetaDataset(ElectricityDataset, MetricsMetaDataset):
    def __init__(self):
        ElectricityDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 200, 500, 20, 500, 20)
