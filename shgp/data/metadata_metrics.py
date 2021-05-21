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


# TODO: Fertility experiment?


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


class PimaMetricsMetaDataset(PimaDataset, MetricsMetaDataset):
    def __init__(self):
        PimaDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 60, 500, 10, 250, 10)


"""
SVGP Distribution: (kmeans++, no grad-optim, with unconstrained/default, M=300)
ELBO - max: -440.252890, min: -4480.603436, median: -4454.929089, mean: -3581.029505, std: 1314.972465.
ACC  - max: 0.982432, min: 0.810811, median: 0.968243, mean: 0.943649, std: 0.057986.
NLL  - max: 0.667848, min: 0.054345, median: 0.663046, mean: 0.516713, std: 0.204524.

PGPR Distribution: (hetero greedy var, no grad-optim, with unconstrained/default, M=300)
ELBO - max: -441.419710, min: -464.018613, median: -452.083285, mean: -451.609530, std: 7.613344.
ACC  - max: 0.986486, min: 0.972973, median: 0.979730, mean: 0.979324, std: 0.004799.
NLL  - max: 0.075884, min: 0.043305, median: 0.052011, mean: 0.054395, std: 0.010713.
"""


class TwonormMetricsMetaDataset(TwonormDataset, MetricsMetaDataset):
    def __init__(self):
        TwonormDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 300, 500, 10, 250, 10)


"""
SVGP Distribution: (kmeans++, no grad-optim, with unconstrained/default, M=300)
ELBO - max: -768.285139, min: -4332.227797, median: -2597.246881, mean: -2260.877601, std: 1301.823368.
ACC  - max: 0.979730, min: 0.495946, median: 0.871622, mean: 0.803243, std: 0.183625.
NLL  - max: 0.646989, min: 0.043059, median: 0.315875, mean: 0.290692, std: 0.209444.

PGPR Distribution: (hetero greedy var, no grad-optim, with unconstrained/default, M=300)
ELBO - max: -933.852979, min: -967.672191, median: -952.890675, mean: -953.277250, std: 9.537135.
ACC  - max: 0.989189, min: 0.964865, median: 0.980405, mean: 0.976622, std: 0.007023.
NLL  - max: 0.097374, min: 0.037707, median: 0.054798, mean: 0.062037, std: 0.015969.
"""


class RingnormMetricsMetaDataset(RingnormDataset, MetricsMetaDataset):
    def __init__(self):
        RingnormDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 300, 500, 10, 250, 10)


"""
SVGP Distribution: (kmeans++, no grad-optim, with unconstrained/default, M=300)

PGPR Distribution: (hetero greedy var, no grad-optim, with unconstrained/default, M=300)
"""


# TODO: Doesn't fit in memory -> crashes
# TODO: Try fix memory errors (maybe I can run PGPR, but not svgp?)
#       This is why it won't fit in memory???
# This experiment was run on a GPU so is not reproducable on CPU.
class MagicMetricsMetaDataset(MagicDataset, MetricsMetaDataset):
    def __init__(self):
        MagicDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 200, 500, 20, 500, 20)


"""
SVGP Distribution: (kmeans++, no grad-optim, with unconstrained/default, M=300)

PGPR Distribution: (hetero greedy var, no grad-optim, with unconstrained/default, M=300)
"""


# TODO: Doesn't fit in memory -> crashes
# TODO: Try fix memory errors (maybe I can run PGPR, but not svgp?)
#       This is why it won't fit in memory???
# This experiment was run on a GPU so is not reproducable on CPU.
class ElectricityMetricsMetaDataset(ElectricityDataset, MetricsMetaDataset):
    def __init__(self):
        ElectricityDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 300, 500, 10, 250, 10)
