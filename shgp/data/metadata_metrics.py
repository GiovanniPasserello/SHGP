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


# SVGP Distribution:
# ELBO - max: -106.792589, min: -196.696657, median: -107.981071, mean: -116.731394, std: 26.660928.
# ACC  - max: 0.940000, min: 0.850000, median: 0.935000, mean: 0.926750, std: 0.025813.
# NLL  - max: 0.345786, min: 0.157273, median: 0.160449, mean: 0.178562, std: 0.055760.
#
# PGPR Distribution:
# ELBO - max: -119.815090, min: -119.815090, median: -119.815090, mean: -119.815090, std: 0.000000.
# ACC  - max: 0.935000, min: 0.935000, median: 0.935000, mean: 0.935000, std: 0.000000.
# NLL  - max: 0.168204, min: 0.168201, median: 0.168201, mean: 0.168202, std: 0.000001.

class BananaMetricsMetaDataset(BananaDataset, MetricsMetaDataset):
    def __init__(self):
        BananaDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 50, 250, 10, 250, 10)
