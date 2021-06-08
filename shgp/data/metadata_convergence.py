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


""" Banana with Exp kernel:
results_hgv_bfgs = -120.12304219158608
results_hgv_adam = -120.74396144822651
results_kmeans_bfgs = -120.92400722500008
results_kmeans_adam = -121.4123171648204
optimal = -120.06609320829034
"""


class BananaConvergenceMetaDataset(BananaDataset, ConvergenceMetaDataset):
    def __init__(self):
        BananaDataset.__init__(self)
        ConvergenceMetaDataset.__init__(self, 40, 5, 5, 10)


""" Crabs with Exp kernel:
results_hgv_bfgs = -30.83096017301088
results_hgv_adam = -30.91271745579118
results_kmeans_bfgs = -30.893175906038067
results_kmeans_adam = -30.592522689905365
optimal = -30.176934247928045
"""


class CrabsConvergenceMetaDataset(CrabsDataset, ConvergenceMetaDataset):
    def __init__(self):
        CrabsDataset.__init__(self)
        ConvergenceMetaDataset.__init__(self, 10, 10, 10, 10)


""" Heart with Exp kernel:
results_hgv_bfgs = -116.87896126803733
results_hgv_adam = -120.50981894487643
results_kmeans_bfgs = -117.49246260411502
results_kmeans_adam = -120.1986022919203
optimal = -116.40517503711263
"""


class HeartConvergenceMetaDataset(HeartDataset, ConvergenceMetaDataset):
    def __init__(self):
        HeartDataset.__init__(self)
        ConvergenceMetaDataset.__init__(self, 35, 3, 10, 10)


""" Ionosphere with Exp kernel:
results_hgv_bfgs = -136.5843232435876
results_hgv_adam = -132.4442299062226
results_kmeans_bfgs = -134.31545532969926
results_kmeans_adam = -133.37755384748468
optimal = -127.33302466735934
"""


class IonosphereConvergenceMetaDataset(IonosphereDataset, ConvergenceMetaDataset):
    def __init__(self):
        IonosphereDataset.__init__(self)
        ConvergenceMetaDataset.__init__(self, 50, 5, 10, 10)


""" Breast Cancer with Exp kernel:
results_hgv_bfgs = -80.05559903201078
results_hgv_adam = -88.1640662775219
results_kmeans_bfgs = -87.11531148182786
results_kmeans_adam = -85.66347679514195
optimal = -80.14058112260301
"""


class BreastCancerConvergenceMetaDataset(BreastCancerDataset, ConvergenceMetaDataset):
    def __init__(self):
        BreastCancerDataset.__init__(self)
        ConvergenceMetaDataset.__init__(self, 50, 5, 10, 10)


""" Pima Diabetes with Exp kernel:
results_hgv_bfgs = -379.6071485735707
results_hgv_adam = -386.11236963470503
results_kmeans_bfgs = -378.93046287023844
results_kmeans_adam = -383.08885975956173
optimal = -377.604760027815
"""


class PimaConvergenceMetaDataset(PimaDataset, ConvergenceMetaDataset):
    def __init__(self):
        PimaDataset.__init__(self)
        ConvergenceMetaDataset.__init__(self, 60, 5, 5, 10)
