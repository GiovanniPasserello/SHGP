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


# SVGP Distribution: (kmeans++, no grad-optim)
# ELBO - max: -93.853361, min: -182.765859, median: -95.308010, mean: -112.722297, std: 32.829257.
# ACC  - max: 0.975000, min: 0.700000, median: 0.862500, mean: 0.865000, std: 0.069101.
# NLL  - max: 0.624052, min: 0.060643, median: 0.304361, mean: 0.331887, std: 0.151419.
#
# PGPR Distribution: (heteroscedastic greedy variance, no grad-optim)
# ELBO - max: -107.566929, min: -115.670121, median: -108.165849, mean: -109.333836, std: 2.417689.
# ACC  - max: 1.000000, min: 0.800000, median: 0.862500, mean: 0.882500, std: 0.058149.
# NLL  - max: 0.517790, min: 0.069080, median: 0.284605, mean: 0.274303, std: 0.104619.

class BananaMetricsMetaDataset(BananaDataset, MetricsMetaDataset):
    def __init__(self):
        BananaDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 50, 250, 10, 250, 10)
