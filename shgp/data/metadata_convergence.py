from dataclasses import dataclass

from shgp.data.dataset import BananaDataset, BreastCancerDataset, CrabsDataset, HeartDataset, IonosphereDataset, PimaDataset


@dataclass
class ConvergenceMetaDataset:
    """
    A dataset utilities class specifically for convergence experiments.
    Please ensure that elbo_period is an integer divisor of (inner_iters * opt_iters).

    :param M: The number of inducing points.
    :param opt_iters: The number of iterations of gradient-based optimisation.
    """
    M: int
    opt_iters: int = 100


class BananaConvergenceMetaDataset(BananaDataset, ConvergenceMetaDataset):
    def __init__(self):
        BananaDataset.__init__(self)
        ConvergenceMetaDataset.__init__(self, 40, 150)


class CrabsConvergenceMetaDataset(CrabsDataset, ConvergenceMetaDataset):
    def __init__(self):
        CrabsDataset.__init__(self)
        ConvergenceMetaDataset.__init__(self, 10, 200)


class HeartConvergenceMetaDataset(HeartDataset, ConvergenceMetaDataset):
    def __init__(self):
        HeartDataset.__init__(self)
        ConvergenceMetaDataset.__init__(self, 35, 150)


class IonosphereConvergenceMetaDataset(IonosphereDataset, ConvergenceMetaDataset):
    def __init__(self):
        IonosphereDataset.__init__(self)
        ConvergenceMetaDataset.__init__(self, 80, 250)


class BreastCancerConvergenceMetaDataset(BreastCancerDataset, ConvergenceMetaDataset):
    def __init__(self):
        BreastCancerDataset.__init__(self)
        ConvergenceMetaDataset.__init__(self, 50, 150)


class PimaConvergenceMetaDataset(PimaDataset, ConvergenceMetaDataset):
    def __init__(self):
        PimaDataset.__init__(self)
        ConvergenceMetaDataset.__init__(self, 60, 150)
