from dataclasses import dataclass

import numpy as np

from shgp.data.dataset import BreastCancerDataset, FertilityDataset, MagicDataset
from shgp.data.metadata_reinit import ReinitMetaDataset


@dataclass
class SparsityMetaDataset:
    """
        A dataset utilities class specifically for sparsity experiments.
        The training hyperparameters typically need to be smaller for sparsity
        experiments. For example, we use a smaller number of optimisation iterations
        otherwise the experiments are computationally infeasible.

        :param num_cycles: The number of times to train a model and average results over.
        :param inner_iters: The number of iterations of the inner optimisation loop.
        :param opt_iters: The number of iterations of gradient-based optimisation of the kernel hyperparameters.
        :param ci_iters: The number of iterations of update for the local variational parameters.
        :param M_array: An array containing the number of inducing points to test.
        :param reinit_metadata: A dataclass containing training hyperparameters for reinitialisation schemes.
    """
    num_cycles: int
    inner_iters: int
    opt_iters: int
    ci_iters: int
    M_array: np.ndarray
    reinit_metadata: ReinitMetaDataset = ReinitMetaDataset()  # for sparsity experiment, we use default class


""" Fertility with Exp kernel - np.arange(1, 31):
results_gv = [-42.35065142 -39.35464259 -39.35464426 -39.35464209 -39.35464269
 -42.35065107 -39.3546428  -39.35464179 -39.35464195 -39.35464215
 -39.35464229 -39.35464224 -39.35464236 -39.35464212 -39.35464417
 -39.35464205 -39.35464209 -39.35464192 -39.35464212 -39.35464186
 -39.35464263 -39.35464141 -39.35464136 -39.35464229 -39.35464584
 -39.35464628 -39.35464198 -39.3546423  -39.35464183 -39.3546425 ]
results_hgv = [-45.34665907 -42.35065092 -42.35065219 -42.35064986 -39.35464339
 -42.35065014 -39.35464253 -39.35464164 -39.35464411 -39.35464301
 -39.35464248 -39.3546429  -39.35464322 -39.35464222 -39.35464381
 -39.35464179 -39.35464291 -39.35464257 -39.35464213 -39.35464161
 -39.35464247 -39.3546422  -39.35464183 -39.35464139 -39.35464259
 -39.3546423  -39.35464138 -39.35464166 -39.35464215 -39.35464218]
optimal = -39.35464385423624
"""


class FertilitySparsityMetaDataset(FertilityDataset, SparsityMetaDataset):
    def __init__(self):
        FertilityDataset.__init__(self)
        SparsityMetaDataset.__init__(self, 10, 10, 250, 10, np.arange(1, 31))


""" Breast Cancer with Exp kernel - np.arange(5, 101, 5):
results_gv = [-130.22956909  -98.45589806  -86.72260874  -80.46782369  -76.716973
  -75.55670327  -75.31185702  -75.10336947  -74.97549006  -74.90682751
  -74.84245045  -74.80137592  -74.76943013  -74.75166719  -74.72767932
  -74.7136058   -74.70275634  -74.69287453  -74.6853426   -74.67759243]
results_hgv = [-139.56282149  -99.58917371  -86.86939456  -80.67553816  -76.16629887
  -75.40867314  -75.11433029  -74.95148554  -74.87154145  -74.79757262
  -74.7759053   -74.74274939  -74.73493773  -74.71050661  -74.69445391
  -74.68535592  -74.6780934   -74.67292411  -74.66851262  -74.66450918]
optimal = -75.029016277001
"""


class BreastCancerSparsityMetaDataset(BreastCancerDataset, SparsityMetaDataset):
    def __init__(self):
        BreastCancerDataset.__init__(self)
        SparsityMetaDataset.__init__(self, 5, 10, 100, 10, np.arange(5, 101, 5))


""" MAGIC with Sigmoid kernel - np.arange(5, 306, 10):
results_gv = [-2609.01627019, -2201.40247108, -2109.2050934,  -2013.48576156,
              -1901.21945807, -1854.57465388, -1825.84648597, -1812.65802267,
              -1797.36711529, -1786.26852406, -1778.03356322, -1773.46910692,
              -1766.3075105,  -1761.9136597,  -1757.24984294, -1753.25090633,
              -1748.17801301, -1746.31409115, -1743.27589385, -1741.22624362,
              -1738.10009734, -1736.18195694, -1733.6701451,  -1731.57258114,
              -1729.72409474, -1727.88304081, -1726.35915071, -1724.42671345,
              -1723.71076765, -1721.85358176, -1720.22779382]
results_hgv = [-2551.32677851, -2200.63281494, -2038.35653607, -1932.38857638,
               -1873.80833915, -1833.6790594,  -1805.48248071, -1788.88192242,
               -1779.27184977, -1770.83345528, -1764.07466289, -1756.49798976,
               -1752.3457539,  -1749.01948999, -1745.69713547, -1742.08328673,
               -1740.06159459, -1736.85540609, -1733.98299389, -1730.60367442,
               -1728.2396689,  -1726.68616977, -1725.01996404, -1723.29015993,
               -1721.98628245, -1720.66932513, -1719.0206125,  -1718.0954,
               -1716.75620072, -1715.80900772, -1714.86194047]
optimal = -1705.2172688375176
"""

"""  MAGIC with Exp kernel - np.arange(5, 306, 10):
results_gv = [-2561.31334758 -2165.37069417 -2134.67602761 -1992.87756039
 -1885.44702783 -1794.06725993 -1769.79365389 -1763.24322235
 -1758.49365415 -1755.42048816 -1753.1743076  -1750.86115964
 -1748.19611735 -1746.44331787 -1744.32813507 -1741.95022722
 -1739.80930487 -1737.55566551 -1735.44098962 -1733.64412236
 -1731.47234623 -1730.00651265 -1727.85975375 -1726.76136829
 -1725.00488299 -1723.30045485 -1721.12463406 -1720.04372473
 -1719.30756427 -1718.2254801  -1717.054943  ]
results_hgv = [-2734.057357   -2165.30727176 -2054.25396881 -1950.02161732
 -1844.38156099 -1783.39589196 -1761.67624335 -1756.59869345
 -1753.99358497 -1750.23157204 -1749.04992228 -1746.05259104
 -1743.84685888 -1741.1791646  -1737.91263266 -1735.98522248
 -1732.95190218 -1730.46783778 -1728.70211972 -1726.65195255
 -1724.42077484 -1722.63382612 -1721.72428039 -1720.06013066
 -1717.60317692 -1716.19707043 -1715.13982877 -1714.1739867
 -1712.47813301 -1711.82166615 -1711.07908877]
optimal = -1702.8659218791713
"""


class MagicSparsityMetaDataset(MagicDataset, SparsityMetaDataset):
    def __init__(self):
        MagicDataset.__init__(self)
        SparsityMetaDataset.__init__(self, 3, 5, 50, 10, np.arange(5, 306, 10))

    # Prune the dataset - a full sparsity experiment is computationally infeasible.
    # This is 4755/19020 datapoints of the total dataset.
    def load_data(self):
        X, Y = super().load_data()
        N = len(X)
        random_subset = np.random.choice(N, N // 4)
        return X[random_subset], Y[random_subset]
