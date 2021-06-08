from dataclasses import dataclass

from shgp.data.dataset import BananaDataset, BreastCancerDataset, CrabsDataset, HeartDataset, IonosphereDataset, PimaDataset


@dataclass
class ConvergenceMetaDataset:
    """
    A dataset utilities class specifically for convergence experiments.
    Please ensure that elbo_period is an integer divisor of (inner_iters * opt_iters).

    :param M: The number of inducing points.
    :param inner_iters: The number of iterations of the inner optimisation loop.
    :param opt_iters: The number of iterations of gradient-based optimisation of the kernel hyperparameters.
    :param ci_iters: The number of iterations of update for the local variational parameters.
    """
    M: int
    inner_iters: int = 10
    opt_iters: int = 20
    ci_iters: int = 10


class BananaConvergenceMetaDataset(BananaDataset, ConvergenceMetaDataset):
    def __init__(self):
        BananaDataset.__init__(self)
        ConvergenceMetaDataset.__init__(self, 40, 5, 5, 10)


class CrabsConvergenceMetaDataset(CrabsDataset, ConvergenceMetaDataset):
    def __init__(self):
        CrabsDataset.__init__(self)
        ConvergenceMetaDataset.__init__(self, 10, 10, 10, 10)


class HeartConvergenceMetaDataset(HeartDataset, ConvergenceMetaDataset):
    def __init__(self):
        HeartDataset.__init__(self)
        ConvergenceMetaDataset.__init__(self, 35, 3, 10, 10)


class IonosphereConvergenceMetaDataset(IonosphereDataset, ConvergenceMetaDataset):
    def __init__(self):
        IonosphereDataset.__init__(self)
        ConvergenceMetaDataset.__init__(self, 50, 5, 10, 10)


class BreastCancerConvergenceMetaDataset(BreastCancerDataset, ConvergenceMetaDataset):
    def __init__(self):
        BreastCancerDataset.__init__(self)
        ConvergenceMetaDataset.__init__(self, 50, 5, 10, 10)


class PimaConvergenceMetaDataset(PimaDataset, ConvergenceMetaDataset):
    def __init__(self):
        PimaDataset.__init__(self)
        ConvergenceMetaDataset.__init__(self, 60, 4, 5, 10)
