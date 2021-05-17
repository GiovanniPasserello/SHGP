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

SVGP Distribution: (kmeans++, no grad-optim, with unconstrained/default, M=200)
ELBO - max: -2267.790177, min: -4488.839930, median: -4376.347131, mean: -4157.906175, std: 635.116272.
ACC  - max: 0.979730, min: 0.779730, median: 0.967568, mean: 0.949324, std: 0.056867.
NLL  - max: 0.667522, min: 0.256185, median: 0.646418, mean: 0.605678, std: 0.117353.

SVGP Distribution: (kmeans++, no grad-optim, with unconstrained/default, M=300)
ELBO - max: -3525.264918, min: -4500.681519, median: -4460.407540, mean: -4194.907578, std: 425.883111.
ACC  - max: 0.974324, min: 0.936486, median: 0.961486, mean: 0.959459, std: 0.012834.
NLL  - max: 0.671031, min: 0.397189, median: 0.662346, mean: 0.595052, std: 0.108684.

PGPR Distribution: (hetero greedy var, no grad-optim, with unconstrained/default, M=100)
ELBO - max: -449.189194, min: -4616.360228, median: -4616.360228, mean: -3784.142286, std: 1664.438106.
ACC  - max: 0.985135, min: 0.902703, median: 0.933108, mean: 0.938243, std: 0.024764.
NLL  - max: 0.693147, min: 0.050044, median: 0.693147, mean: 0.566307, std: 0.253713.

PGPR Distribution: (hetero greedy var, no grad-optim, with unconstrained/default, M=200)
ELBO - max: -456.871311, min: -4616.360228, median: -4616.360228, mean: -3785.067699, std: 1662.585607.
ACC  - max: 0.979730, min: 0.902703, median: 0.954054, mean: 0.952568, std: 0.022077.
NLL  - max: 0.693147, min: 0.048832, median: 0.693147, mean: 0.565048, std: 0.256204.

PGPR Distribution: (hetero greedy var, no grad-optim, with unconstrained/default, M=300)
ELBO - max: -441.419710, min: -464.018613, median: -452.083285, mean: -451.609530, std: 7.613344.
ACC  - max: 0.985135, min: 0.972973, median: 0.975676, mean: 0.977838, std: 0.004196.
NLL  - max: 0.077174, min: 0.046878, median: 0.062856, mean: 0.063841, std: 0.010713.
"""

# Too few for stability
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


# TODO: Try larger M on Colab (M=500)?
# With a large number of inducing points (>= 200), we are prone to inversion/cholesky errors?
# This means that we have to restart many times - a solution to this would be valuable in future work.
class TwonormMetricsMetaDataset(TwonormDataset, MetricsMetaDataset):
    def __init__(self):
        TwonormDataset.__init__(self)
        MetricsMetaDataset.__init__(self, 10, 300, 500, 10, 250, 10)


"""
SVGP Distribution: (kmeans++, no grad-optim, with unconstrained/default, M=100)
ELBO - max: -1343.326952, min: -3853.552475, median: -1663.819734, mean: -2132.205899, std: 897.843589.
ACC  - max: 0.959459, min: 0.508108, median: 0.946622, mean: 0.852703, std: 0.168002.
NLL  - max: 0.567328, min: 0.111391, median: 0.154397, mean: 0.244232, std: 0.156691.

SVGP Distribution: (kmeans++, no grad-optim, with unconstrained/default, M=200)
ELBO - max: -827.505858, min: -4229.019174, median: -1354.343196, mean: -1952.737348, std: 1263.648744.
ACC  - max: 0.981081, min: 0.535135, median: 0.946622, mean: 0.861351, std: 0.166338.
NLL  - max: 0.630038, min: 0.051680, median: 0.142081, mean: 0.238596, std: 0.203371.

SVGP Distribution: (kmeans++, no grad-optim, with unconstrained/default, M=350)
# ELBO - max: -752.252068, min: -3933.220987, median: -2438.577702, mean: -2129.962850, std: 1167.147312.
# ACC  - max: 0.983784, min: 0.539189, median: 0.898649, mean: 0.839189, std: 0.166699.
# NLL  - max: 0.571916, min: 0.047898, median: 0.281699, mean: 0.258076, std: 0.184207.

PGPR Distribution: (hetero greedy var, no grad-optim, with unconstrained/default, M=100)
ELBO - max: -1915.354700, min: -4616.360228, median: -2110.011878, mean: -2342.593651, std: 763.174001.
ACC  - max: 0.968919, min: 0.822973, median: 0.943919, mean: 0.933243, std: 0.038046.
NLL  - max: 0.693147, min: 0.117649, median: 0.173071, mean: 0.221537, std: 0.159302.

PGPR Distribution: (hetero greedy var, no grad-optim, with unconstrained/default, M=200)
ELBO - max: -1195.692441, min: -4616.360228, median: -1218.478131, mean: -1559.946067, std: 1018.956643.
ACC  - max: 0.983784, min: 0.709459, median: 0.971622, mean: 0.947568, std: 0.079626.
NLL  - max: 0.693147, min: 0.060996, median: 0.081233, mean: 0.139155, std: 0.184928.

PGPR Distribution: (hetero greedy var, no grad-optim, with unconstrained/default, M=350)
ELBO - max: -923.197476, min: -4616.360228, median: -939.158414, mean: -1671.387251, std: 1472.501752.
ACC  - max: 0.981081, min: 0.487838, median: 0.976351, mean: 0.907027, std: 0.152161.
NLL  - max: 0.693147, min: 0.047588, median: 0.065627, mean: 0.189958, std: 0.251713.
"""

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

# (10, 200, 1000, 20, 500, 20)
# SVGP: ELBO = -4229.019174, ACC = 0.535135, NLL = 0.630038. # 1
# PGPR: ELBO = -1228.709653, ACC = 0.983784, NLL = 0.060996.
# SVGP: ELBO = -827.505858, ACC = 0.960811, NLL = 0.089892.  # 2
# PGPR: ELBO = -1211.444851, ACC = 0.963514, NLL = 0.092852.
# SVGP: ELBO = -838.526551, ACC = 0.967568, NLL = 0.078273.  # 3
# PGPR: ELBO = -1208.594288, ACC = 0.967568, NLL = 0.085665.
# SVGP: ELBO = -2518.517455, ACC = 0.886486, NLL = 0.307559. # 4
# PGPR: ELBO = -1203.384664, ACC = 0.968919, NLL = 0.084999.
# SVGP: ELBO = -2876.921200, ACC = 0.858108, NLL = 0.343834. # 5
# PGPR: ELBO = -1210.708639, ACC = 0.978378, NLL = 0.070667.
# SVGP: ELBO = -861.552785, ACC = 0.981081, NLL = 0.051680.  # 6
# PGPR: ELBO = -1242.803776, ACC = 0.983784, NLL = 0.061210.
# SVGP: ELBO = -1847.133607, ACC = 0.932432, NLL = 0.194271. # 7
# PGPR: ELBO = -1195.692441, ACC = 0.971622, NLL = 0.081010.
# SVGP: ELBO = -860.419242, ACC = 0.977027, NLL = 0.063239.  # 8
# PGPR: ELBO = -4616.360228, ACC = 0.709459, NLL = 0.693147.
# SVGP: ELBO = -847.409672, ACC = 0.974324, NLL = 0.070873.  # 9
# PGPR: ELBO = -1256.250717, ACC = 0.977027, NLL = 0.081456.
# SVGP: ELBO = -3820.367932, ACC = 0.540541, NLL = 0.556305. # 10
# PGPR: ELBO = -1225.511412, ACC = 0.971622, NLL = 0.079550.

# (10, 350, 500, 10, 250, 10)
# SVGP: ELBO = -3590.345572, ACC = 0.551351, NLL = 0.484845. # 1
# PGPR: ELBO = -946.302094, ACC = 0.981081, NLL = 0.047588.
# SVGP: ELBO = -3933.220987, ACC = 0.539189, NLL = 0.571916. # 2
# PGPR: ELBO = -923.197476, ACC = 0.978378, NLL = 0.065232.
# SVGP: ELBO = -780.125658, ACC = 0.983784, NLL = 0.047898.  # 3
# PGPR: ELBO = -4616.360228, ACC = 0.774324, NLL = 0.693147.
# SVGP: ELBO = -3078.699040, ACC = 0.701351, NLL = 0.432548. # 4
# PGPR: ELBO = -936.401064, ACC = 0.978378, NLL = 0.066022.
# SVGP: ELBO = -965.233208, ACC = 0.970270, NLL = 0.061761.  # 5
# PGPR: ELBO = -938.161495, ACC = 0.970270, NLL = 0.059254.
# SVGP: ELBO = -758.443778, ACC = 0.972973, NLL = 0.075304.  # 6
# PGPR: ELBO = -928.896593, ACC = 0.968919, NLL = 0.078204.
# SVGP: ELBO = -2416.712015, ACC = 0.900000, NLL = 0.281091. # 7
# PGPR: ELBO = -926.775732, ACC = 0.978378, NLL = 0.073006.
# SVGP: ELBO = -2460.443388, ACC = 0.895946, NLL = 0.282307. # 8
# PGPR: ELBO = -4616.360228, ACC = 0.487838, NLL = 0.693147.   #  TODO: ???
# SVGP: ELBO = -2564.152789, ACC = 0.897297, NLL = 0.287191. # 9
# PGPR: ELBO = -940.155333, ACC = 0.975676, NLL = 0.063902.
# SVGP: ELBO = -752.252068, ACC = 0.979730, NLL = 0.055897. # 10
# PGPR: ELBO = -941.262270, ACC = 0.977027, NLL = 0.060075.

# (10, 500, 1000, 20, 500, 20)
# SVGP: ELBO = -2512.950163, ACC = 0.878378, NLL = 0.286941. # 1
# PGPR: ELBO = -929.724989, ACC = 0.979730, NLL = 0.047482.


# TODO: Try 250 (with 20, 500, 20), 400! -> UNLUCKY 48% at 350
# TODO: Investigate higher M here for stability (M=500 on Colab?)
class RingnormMetricsMetaDataset(RingnormDataset, MetricsMetaDataset):
    def __init__(self):
        RingnormDataset.__init__(self)  # TODO: Reset to M=200?
        MetricsMetaDataset.__init__(self, 10, 350, 500, 10, 250, 10)
