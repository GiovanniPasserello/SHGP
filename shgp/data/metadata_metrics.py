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


""" with ConstrainedExpSEKernel
SVGP Distribution: (kmeans++, no grad-optim)
ELBO - max: -96.172701, min: -142.895417, median: -99.064924, mean: -103.964749, std: 13.172938.
ACC  - max: 0.975000, min: 0.775000, median: 0.912500, mean: 0.895000, std: 0.052202.
NLL  - max: 0.472376, min: 0.061772, median: 0.202291, mean: 0.215964, std: 0.107715.

SVGP Distribution: (kmeans++, no grad-optim, with unconstrained)
ELBO - max: -89.519647, min: -103.250001, median: -98.954888, mean: -98.629271, std: 3.787107.
ACC  - max: 0.975000, min: 0.825000, median: 0.912500, mean: 0.902500, std: 0.039449.
NLL  - max: 0.538161, min: 0.061780, median: 0.202625, mean: 0.222576, std: 0.123889.

PGPR Distribution: (heteroscedastic greedy variance, no grad-optim)
ELBO - max: -103.146688, min: -115.827573, median: -110.754288, mean: -110.907491, std: 3.230506.
ACC  - max: 1.000000, min: 0.825000, median: 0.925000, mean: 0.907500, std: 0.044791.
NLL  - max: 0.457501, min: 0.069479, median: 0.215852, mean: 0.216817, std: 0.097865.

PGPR Distribution: (heteroscedastic greedy variance, no grad-optim, with unconstrained)
ELBO - max: -103.146880, min: -115.827680, median: -110.754296, mean: -110.913423, std: 3.235203.
ACC  - max: 1.000000, min: 0.825000, median: 0.925000, mean: 0.907500, std: 0.044791.
NLL  - max: 0.457047, min: 0.069402, median: 0.215834, mean: 0.216745, std: 0.097789.
"""


class BananaMetricsMetaDataset(BananaDataset, MetricsMetaDataset):
    def __init__(self):
        BananaDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 40, 250, 10, 250, 10)


"""  with ConstrainedExpSEKernel
SVGP Distribution: (kmeans++, no grad-optim)
ELBO - max: -99.340152, min: -145.892476, median: -128.431883, mean: -123.210936, std: 19.711216.
ACC  - max: 0.888889, min: 0.740741, median: 0.814815, mean: 0.814815, std: 0.052378.
NLL  - max: 0.523986, min: 0.403181, median: 0.477456, mean: 0.464399, std: 0.037347.

SVGP Distribution: (kmeans++, no grad-optim, with unconstrained)
ELBO - max: -97.981874, min: -141.906360, median: -103.374677, mean: -112.786775, std: 17.097140.
ACC  - max: 0.925926, min: 0.703704, median: 0.796296, mean: 0.803704, std: 0.074167.
NLL  - max: 0.599461, min: 0.251868, median: 0.443676, mean: 0.450312, std: 0.107118.

PGPR Distribution: (heteroscedastic greedy variance, no grad-optim)
ELBO - max: -102.888512, min: -111.538384, median: -105.964342, mean: -106.061365, std: 2.483365.
ACC  - max: 1.000000, min: 0.740741, median: 0.833333, mean: 0.848148, std: 0.071146.
NLL  - max: 0.510061, min: 0.176194, median: 0.395165, mean: 0.385094, std: 0.095510.

PGPR Distribution: (heteroscedastic greedy variance, no grad-optim, with unconstrained)
ELBO - max: -101.877519, min: -110.116645, median: -105.968008, mean: -105.697286, std: 2.681292.
ACC  - max: 0.925926, min: 0.740741, median: 0.814815, mean: 0.829630, std: 0.068693.
NLL  - max: 0.569416, min: 0.235813, median: 0.391251, mean: 0.403495, std: 0.106145.
"""


# TODO: Sparsity experiment to decide M
#       Try lower M
class HeartMetricsMetaDataset(HeartDataset, MetricsMetaDataset):
    def __init__(self):
        HeartDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 60, 250, 10, 250, 10)


"""  with ConstrainedExpSEKernel
SVGP Distribution: (kmeans++, no grad-optim)
ELBO - max: -59.207973, min: -204.349690, median: -190.682037, mean: -164.279048, std: 51.465942.
ACC  - max: 0.982456, min: 0.842105, median: 0.877193, mean: 0.898246, std: 0.050115.
NLL  - max: 0.436276, min: 0.045218, median: 0.331891, mean: 0.293182, std: 0.123159.

SVGP Distribution: (kmeans++, with grad-optim)
ELBO - max: -51.490389, min: -256.267176, median: -91.386792, mean: -133.586293, std: 83.223318.
ACC  - max: 1.000000, min: 0.771930, median: 0.964912, mean: 0.922807, std: 0.079317.
NLL  - max: 0.466121, min: 0.037010, median: 0.168088, mean: 0.218202, std: 0.155015.

SVGP Distribution: (kmeans++, with unconstrained)
ELBO - max: -51.035899, min: -260.193332, median: -79.258723, mean: -128.164729, std: 87.821823.
ACC  - max: 1.000000, min: 0.877193, median: 0.956140, mean: 0.954386, std: 0.034379.
NLL  - max: 0.438799, min: 0.035690, median: 0.147612, mean: 0.198864, std: 0.145042.

SVGP Distribution: (kmeans++, with grad-optim and unconstrained)
ELBO - max: -49.402746, min: -263.220134, median: -55.021090, mean: -75.629275, std: 62.565196.
ACC  - max: 1.000000, min: 0.807018, median: 0.982456, mean: 0.963158, std: 0.053473.
NLL  - max: 0.436694, min: 0.038113, median: 0.088166, mean: 0.121176, std: 0.110232.

PGPR Distribution: (heteroscedastic greedy variance, no grad-optim)
ELBO - max: -63.981061, min: -71.668601, median: -69.323396, mean: -68.677415, std: 2.596238.
ACC  - max: 1.000000, min: 0.929825, median: 0.982456, mean: 0.971930, std: 0.022467.
NLL  - max: 0.191799, min: 0.037462, median: 0.087556, mean: 0.105514, std: 0.055949.

PGPR Distribution: (heteroscedastic greedy variance, no grad-optim, with unconstrained)
ELBO - max: -65.263036, min: -71.668893, median: -69.381602, mean: -69.548874, std: 1.810755.
ACC  - max: 1.000000, min: 0.947368, median: 0.982456, mean: 0.980702, std: 0.014573.
NLL  - max: 0.156879, min: 0.037506, median: 0.085697, mean: 0.082185, std: 0.033648.
"""


class BreastCancerMetaDataset(BreastCancerDataset, MetricsMetaDataset):
    def __init__(self):
        BreastCancerDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 50, 500, 10, 250, 10)
