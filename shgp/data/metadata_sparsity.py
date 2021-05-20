from dataclasses import dataclass

import numpy as np

from shgp.data.dataset import BananaDataset, BreastCancerDataset, CrabsDataset, FertilityDataset, HeartDataset, \
    IonosphereDataset, MagicDataset, PimaDataset, RingnormDataset, TwonormDataset


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
    """
    num_cycles: int
    inner_iters: int
    opt_iters: int
    ci_iters: int
    M_array: np.ndarray


# TODO: When plotting these, show the curve as it clearly approaches convergence
#       We care about how the methods achieve convergence, not the extreme cases

""" Banana with Exp kernel - np.arange(5, 51, 5):
results_uniform = [-252.13643516, -208.20541972, -157.02072046, -140.62518881, -131.01991315, -126.45785361,
                   -125.58773707, -123.02937557, -122.18254487, -121.02846546]
results_kmeans = [-261.4396643, -148.19500976, -134.64923173, -128.44973186, -123.8891034, -122.41569379,
                  -121.3741855, -120.59062521, -120.25771766, -120.03329343]
results_gv = [-263.96390801, -193.47278328, -146.06442775, -131.69559018, -125.21248857, -122.07170094,
              -120.68262996, -120.08218722, -119.87603769, -119.79144935]
results_hgv = [-265.21812865, -218.44034193, -146.41359466, -133.27414978, -124.94452836, -122.51418155,
               -120.67969943, -120.04857661, -119.9010278,  -119.80383267]
optimal = -119.74156303725914
"""

""" Banana with Exp kernel - np.arange(5, 51, 5):
results_hgv = [-265.21812865, -218.44034193, -146.41359466, -133.27414978, -124.94452836, -122.51418155,
               -120.67969943, -120.04857661, -119.9010278,  -119.80383267]
results_hgv_then_optimise = [-187.90945912, -143.35572839, -129.54256206, -122.85893017, -120.83210659
                             -120.07338803, -119.83879177, -119.76479951, -119.74784784, -119.74348298]
optimal = -119.74156303725914
"""


# TODO: Plot
class BananaSparsityMetaDataset(BananaDataset, SparsityMetaDataset):
    def __init__(self):
        BananaDataset.__init__(self)
        SparsityMetaDataset.__init__(self, 5, 10, 250, 10, np.arange(5, 51, 5))


""" Fertility with Exp kernel - np.arange(1, 31):
results_uniform = [-45.34666016, -39.35464394, -39.35464209, -39.35464442, -39.35464312,
                   -39.35464272, -39.35464202, -39.35464229, -39.35464274, -39.35464186,
                   -39.35464128, -39.35464191, -39.35464173, -39.35464241, -39.35464358,
                   -39.3546416,  -39.35464298, -39.35464156, -39.35464549, -39.3546475,
                   -39.35464692, -39.35464528, -39.35464175, -39.35464522, -39.35464456,
                   -39.35464439, -39.35464208, -39.35464906, -39.35464219, -39.35465084]
results_kmeans = [-39.35464118, -39.35464133, -39.35464219, -39.35464229, -39.35464157,
                  -39.35464161, -39.35464197, -39.35464189, -39.35464274, -39.35464231,
                  -39.35464227, -39.35464261, -39.35464233, -39.35464184, -39.35464243,
                  -39.35464234, -39.35464168, -39.35464155, -39.35464159, -39.3546415,
                  -39.35464151, -39.35464227, -39.3546419,  -39.35464207, -39.35464247,
                  -39.35464173, -39.35464147, -39.35464228, -39.35464153, -39.3546423]
results_gv = [-42.35065142, -39.35464259, -39.35464426, -39.35464209, -39.35464269,
              -42.35065107, -39.3546428,  -39.35464179, -39.35464195, -39.35464215,
              -39.35464229, -39.35464224, -39.35464236, -39.35464212, -39.35464417,
              -39.35464205, -39.35464209, -39.35464192, -39.35464212, -39.35464186,
              -39.35464263, -39.35464141, -39.35464136, -39.35464229, -39.35464584,
              -39.35464628, -39.35464198, -39.3546423,  -39.35464183, -39.3546425]
results_hgv = [-45.34665907, -42.35065092, -42.35065219, -42.35064986, -39.35464339,
               -42.35065014, -39.35464253, -39.35464164, -39.35464411, -39.35464301,
               -39.35464248, -39.3546429,  -39.35464322, -39.35464222, -39.35464381,
               -39.35464179, -39.35464291, -39.35464257, -39.35464213, -39.35464161,
               -39.35464247, -39.3546422,  -39.35464183, -39.35464139, -39.35464259,
               -39.3546423,  -39.35464138, -39.35464166, -39.35464215, -39.35464218]
optimal = -39.35464385423624
"""

""" Fertility with Exp kernel - np.arange(1, 31):
results_hgv = [-51.33867501 -45.34666055 -39.35464361 -42.3506516  -39.35464283
 -39.35464206 -39.35464241 -39.35464186 -39.35464319 -39.35464326
 -39.35464316 -39.35464293 -39.3546419  -39.35464249 -39.35464234
 -39.35464189 -39.35464184 -39.35464278 -39.3546421  -39.35464181
 -39.35464253 -39.3546418  -39.35464226 -39.35464168 -39.35464247
 -39.35464186 -39.35464306 -39.35464413 -39.35464235 -39.35464188]
results_hgv_then_optimise = [-42.3506507  -54.33468666 -39.35464292 -39.35464409 -42.35064979
 -39.35464194 -39.35464376 -39.35464256 -39.35464417 -39.35464362
 -39.35464186 -39.35464234 -39.35464259 -39.35464232 -39.35464194
 -39.35464245 -39.35464207 -39.35464162 -39.35464251 -39.35464197
 -39.35464186 -39.35464162 -39.3546426  -39.35464362 -39.35464131
 -39.35464152 -39.35464332 -39.35464181 -39.35464562 -39.35464186]
optimal = -39.354643854237516
"""


# TODO: Plot (very unstable / noisy for all four)
class FertilitySparsityMetaDataset(FertilityDataset, SparsityMetaDataset):
    def __init__(self):
        FertilityDataset.__init__(self)
        SparsityMetaDataset.__init__(self, 10, 10, 250, 10, np.arange(1, 31))


""" Crabs with Exp kernel - np.arange(5, 20):
results_uniform = [-80.61627643, -47.09542691, -31.46513565, -30.84902902, -30.44167335,
                   -30.38516442, -30.23776116, -30.23074342, -30.19937426, -30.19039235,
                   -30.18762122, -30.18395277, -30.17936203, -30.1790386,  -30.17803817]
results_kmeans = [-37.76849839, -36.08477977, -32.28005576, -31.36771645, -30.64517772,
                  -30.26398711, -30.21539706, -30.19207023, -30.18986539, -30.18215712,
                  -30.18018349, -30.17911427, -30.17776444, -30.1778678,  -30.17749545]
results_gv = [-37.91640816, -32.60721861, -31.05576051, -30.27227717, -30.19792298,
              -30.13459039, -30.1305231,  -30.12853005, -30.12531174, -30.12233256,
              -30.12177307, -30.12097562, -30.12067711, -30.12046462, -30.12032768]
results_hgv = [-77.020314,   -33.02012411, -30.70187662, -30.26146468, -30.19302943,
               -30.16403339, -30.13561464, -30.13122342, -30.12628466, -30.12465734,
               -30.12163524, -30.12093994, -30.12056074, -30.12043881, -30.12021612]
optimal = -30.12021581
"""

""" Crabs with Exp kernel - np.arange(5, 20):
results_hgv = [-77.020314,  -33.02012411, -30.70187662, -30.26146468, -30.19302943,
               -30.16403339, -30.13561464, -30.13122342, -30.12628466, -30.12465734,
               -30.12163524, -30.12093994, -30.12056074, -30.12043881, -30.12021612]
results_hgv_then_optimise = [-36.00822127 -31.25506725 -30.33460073 -30.14617903 -30.12271089
 -30.12136622 -30.12073425 -30.12034384 -30.12022219 -30.12011015
 -30.12005339 -30.1200338  -30.12003275 -30.1200168  -30.11999865]
optimal = -30.17694684298644
"""


# TODO: Plot (start plot from M=6 or M=7 - or set y-axis limits to -37.5)
class CrabsSparsityMetaDataset(CrabsDataset, SparsityMetaDataset):
    def __init__(self):
        CrabsDataset.__init__(self)
        SparsityMetaDataset.__init__(self, 10, 10, 250, 10, np.arange(5, 20))


""" Heart with Exp kernel - np.arange(5, 51, 5):
results_uniform = [-149.42498625, -127.93138126, -118.61004383, -117.26833717, -117.08529289,
                   -117.32575073, -116.87486456, -116.81209015, -116.76194136, -116.72075093]
results_kmeans = [-131.03841683, -122.5709699,  -117.7041781,  -117.04362786, -116.8887037,
                  -116.79497328, -116.72721457, -116.67900941, -116.62119305, -116.60006627]
results_gv = [-149.46760587, -125.11375784, -117.37715944, -117.22393201, -116.99396058,
              -116.89471727, -116.82395893, -116.75403832, -116.69246211, -116.63350006]
results_hgv = [-149.12100835, -126.93577783, -117.56235171, -117.23099965, -117.01637655,
               -116.87625325, -116.79155455, -116.7388337,  -116.68191999, -116.61718728]
optimal = -116.39046838357692
"""

""" Heart with Exp kernel - np.arange(5, 51, 5):
results_hgv = [-149.12100835 -126.93577783 -117.56235171 -117.23099965 -117.01637655
 -116.87625325 -116.79155455 -116.7388337  -116.68191999 -116.61718728]
results_hgv_then_optimise = [-143.75599    -120.08909116 -116.57998049 -116.46777726 -116.4453033
 -123.50099838 -116.41999216 -116.41201695 -116.40704368 -116.40335913]
optimal = -116.39046838357692
"""


# TODO: Plot
class HeartSparsityMetaDataset(HeartDataset, SparsityMetaDataset):
    def __init__(self):
        HeartDataset.__init__(self)
        SparsityMetaDataset.__init__(self, 10, 10, 250, 10, np.arange(5, 51, 5))


""" Ionosphere with Exp kernel - np.arange(5, 101, 5):
results_uniform = [-195.99568983, -183.73768501, -170.40518643, -163.2257885,  -150.83160473,
                   -145.56911293, -133.50300213, -131.54295144, -131.31397919, -131.10743971,
                   -130.91596957, -130.80930761, -130.70859252, -130.59523248, -130.61580109,
                   -130.52012504, -130.44194978, -130.41369647, -130.41175474, -130.30733303]
results_kmeans = [-167.05182502, -159.25789831, -151.62947209, -145.36073602, -141.29886399,
                  -137.49455579, -132.63846187, -131.56577962, -131.14418307, -131.00202243,
                  -130.66893227, -130.58952932, -130.48620511, -130.4270217,  -130.35708797,
                  -130.30755513, -130.20001139, -130.09886699, -130.07788179, -129.98892198]
results_gv = [-196.65469498, -175.97484797, -165.9095813,  -151.71188434, -145.7723689,
              -137.99585018, -131.53087214, -131.33485169, -131.16074613, -130.97737169,
              -130.83737541, -130.68515049, -130.56488604, -130.39868595, -130.27334016,
              -130.21817046, -130.0857033,  -129.94984779, -129.75846762, -129.60681498]
results_hgv = [-203.80311205, -180.12051104, -167.71280856, -154.60128442, -147.11327949,
               -136.90363315, -131.61687268, -131.34020331, -131.13195079, -130.96671821,
               -130.82673061, -130.7042614,  -130.54060093, -130.42575658, -130.28609701,
               -130.23969488, -130.15821464, -130.04294839, -129.95596404, -129.83239931]
optimal = -127.00552361477861
"""

""" Ionosphere with Exp kernel - np.arange(5, 101, 5):
results_hgv = [-206.09356244919604, -183.24536775439262, -167.14560471998274, -154.73159392969552, 
-147.1059980225197, -137.07158782736937, -131.61687263486573, -131.34020360796418, -131.1526319167218, 
-130.93655018833942, -130.8330417215593, -130.69758539868462, -130.5985301510862, -130.42939913109558, 
-130.34080884755872, -130.27729582881605, -130.11691308944117, -129.9998350770992, -129.9794852102401, 
-129.87016236446476]
results_hgv_then_optimise = [-153.73386594882237, -143.87917141067973, -139.5352554664604, -136.33786916900803, 
-133.79081164982009, -131.2309264156396, -129.59001708784606, -129.276073055637, -129.0732634622664, 
-128.88357852294703, -128.72630150779162, -128.50254430015048, -128.3291856770311, -128.13987618189543, 
-127.96852417887922, -127.80981361806997, -127.68314589407956, -127.58164336658196, -127.45071692472926, 
-127.35130477028518]
optimal = -127.00553397076634
"""


# TODO: Plot
class IonosphereSparsityMetaDataset(IonosphereDataset, SparsityMetaDataset):
    def __init__(self):
        IonosphereDataset.__init__(self)
        SparsityMetaDataset.__init__(self, 10, 10, 250, 10, np.arange(5, 101, 5))


""" Breast Cancer with Exp kernel - np.arange(5, 101, 5):
results_uniform = [-143.29751761, -100.16207851, -87.94939344, -80.76551685, -77.80310778,
                   -76.19368973,  -75.59881276,  -75.44682893, -75.35277861, -75.23948104,
                   -75.21717796,  -75.21656606,  -75.1417328,  -75.11478398, -75.13099418,
                   -75.1117179,   -75.10418052,  -75.08718824, -75.08370164, -75.0643957]
results_kmeans = [-115.65709975, -90.10761774, -83.64096084, -80.2528042,  -77.18323436,
                  -75.96563409,  -75.59557702, -75.32452344, -75.19829685, -75.12990161,
                  -75.08124084,  -75.07916469, -75.05162263, -75.0424329,  -75.03501528,
                  -75.03824124,  -75.03082868, -75.02377655, -75.02299504, -75.0240564]
results_gv = [-130.22956909, -98.45589806, -86.72260874, -80.46782369, -76.716973,
              -75.55670327,  -75.31185702, -75.10336947, -74.97549006, -74.90682751,
              -74.84245045,  -74.80137592, -74.76943013, -74.75166719, -74.72767932,
              -74.7136058,   -74.70275634, -74.69287453, -74.6853426,  -74.67759243]
results_hgv = [-139.56282149, -99.58917371, -86.86939456, -80.67553816, -76.16629887,
               -75.40867314,  -75.11433029, -74.95148554, -74.87154145, -74.79757262,
               -74.7759053,   -74.74274939, -74.73493773, -74.71050661, -74.69445391,
               -74.68535592,  -74.6780934,  -74.67292411, -74.66851262, -74.66450918]
optimal = -74.65128212
"""

""" Breast Cancer with Exp kernel - np.arange(5, 101, 5):
results_hgv = [-128.54272442, -98.5290426,  -87.82357273, -80.77856873, -76.16624748,
               -75.41824255,  -75.11827172, -74.95152676, -74.86525487, -74.79757264,
               -74.77744846,  -74.74552347, -74.73493794, -74.71050664, -74.69445384,
               -74.68535608,  -74.67809315, -74.67292411, -74.66851115, -74.66450929]
results_hgv_then_optimise = [-97.32619739 -82.72792927 -76.78488875 -75.19711647 -74.78423776
 -74.69878274 -74.67597054 -74.66463683 -74.65998954 -74.65666712
 -74.65461688 -74.65336598 -74.65256175 -74.65220575 -74.65171341
 -74.65135656 -74.65113877 -74.65078521 -74.65077963 -74.65079426]
optimal = -75.02901627637823
"""


# TODO: Plot (start plot from M=20 for stability at convergence)
class BreastCancerSparsityMetaDataset(BreastCancerDataset, SparsityMetaDataset):
    def __init__(self):
        BreastCancerDataset.__init__(self)
        SparsityMetaDataset.__init__(self, 5, 10, 100, 10, np.arange(5, 101, 5))


""" Pima Diabetes with Exp kernel - np.arange(5, 101, 5):
results_uniform = [-444.77996138, -386.91316907, -386.02783327, -385.31543353, -384.54316826,
                   -383.67183261, -382.85791003, -382.46500407, -381.84566833, -380.99304483,
                   -380.63460644, -380.55328098, -380.27110464, -379.96738077, -379.8475552,
                   -379.90802159, -379.40674155, -379.26510242, -379.15252496, -379.1945228]
results_kmeans = [-412.01661822, -385.87567976, -385.15533115, -384.31392658, -383.22495125,
                  -382.52105175, -381.99257373, -380.89674274, -380.75191388, -380.17271875,
                  -379.39160705, -379.47724897, -379.35385998, -379.1131317,  -378.93623123,
                  -378.8262073,  -378.64420163, -378.6724664,  -378.52242826, -378.45192916]
results_gv = [-423.64588235, -386.59501815, -386.17788556, -385.68507796, -384.89533349,
              -383.84076949, -382.54249711, -381.6088387,  -381.27863342, -380.61205978,
              -380.16222904, -379.75945861, -379.48135565, -379.35240091, -378.95491056,
              -378.80356969, -378.65048198, -378.5303823,  -378.40855262, -378.3538156]
results_hgv = [-420.21889021, -386.66910472, -386.27469259, -385.7520532,  -385.11243869,
               -383.90052552, -382.9113214,  -382.01660468, -381.17262278, -380.69372039,
               -380.22351617, -379.7882304,  -379.40044747, -379.2088569,  -379.02888323,
               -378.88890559, -378.65256517, -378.48233193, -378.40623597, -378.34645726]
optimal = -377.60474770202654
"""

""" Pima Diabetes with Exp kernel - np.arange(5, 101, 5):
results_hgv = [-428.40331381, -386.66910472, -386.26175908, -385.68283928, -385.15406953,
               -384.15858353, -382.72303009, -381.98085024, -381.21443913, -380.69221024,
               -380.28591426, -379.9466808,  -379.46890005, -379.23909147, -379.00873003,
               -378.79243527, -378.63612655, -378.57201862, -378.42718324, -378.30245032]
results_hgv_then_optimise = [-416.0128765  -383.12674192 -381.24907292 -380.23462236 -379.5407066
 -379.01157183 -378.63276174 -378.38281912 -378.18601216 -378.05279903
 -377.94867806 -377.87153446 -377.81553453 -377.77530287 -377.73936246
 -377.71547247 -377.69614057 -377.67934838 -377.66691497 -377.65551202]
optimal = -377.6047477020269
"""


# TODO: Plot (start plot from M=10)
class PimaSparsityMetaDataset(PimaDataset, SparsityMetaDataset):
    def __init__(self):
        PimaDataset.__init__(self)
        SparsityMetaDataset.__init__(self, 5, 10, 100, 10, np.arange(5, 101, 5))


""" MAGIC with Exp kernel - np.arange(5, 306, 10):
results_uniform = [-2582.99173976, -2104.80811312, -1958.83503055, -1881.14354052,
                   -1854.33932865, -1840.50499331, -1821.01402584, -1808.7078278,
                   -1799.20339048, -1788.37600696, -1784.02246444, -1785.58873987,
                   -1775.91781631, -1766.60142097, -1769.52043129, -1770.07095783,
                   -1765.34210231, -1761.62787456, -1757.23640484, -1753.30464709,
                   -1754.24995716, -1755.63574753, -1751.34110749, -1747.01796176,
                   -1746.89917371, -1746.50369033, -1744.82567686, -1744.84720835,
                   -1740.74121855, -1741.57467377, -1741.08639596]
results_kmeans = [-2313.16800662, -2041.80009975, -1918.94840119, -1875.28742534,
                  -1827.87728808, -1798.85417887, -1779.77868459, -1771.28671117,
                  -1762.8463923,  -1757.30426553, -1755.01539015, -1749.82217907,
                  -1746.25761289, -1743.13321865, -1741.20122444, -1736.80472491,
                  -1735.91522426, -1733.84242707, -1732.03534237, -1730.60244738,
                  -1730.01434255, -1727.79461215, -1725.82638431, -1726.02914834,
                  -1724.21537818, -1722.29997238, -1723.99172757, -1722.36707742,
                  -1722.03890138, -1721.87861348, -1720.53102835]
results_gv = [-2561.31334758, -2165.37069417, -2134.67602761, -1992.87756039,
              -1885.44702783, -1794.06725993, -1769.79365389, -1763.24322235,
              -1758.49365415, -1755.42048816, -1753.1743076,  -1750.86115964,
              -1748.19611735, -1746.44331787, -1744.32813507, -1741.95022722,
              -1739.80930487, -1737.55566551, -1735.44098962, -1733.64412236,
              -1731.47234623, -1730.00651265, -1727.85975375, -1726.76136829,
              -1725.00488299, -1723.30045485, -1721.12463406, -1720.04372473,
              -1719.30756427, -1718.2254801,  -1717.054943]
results_hgv = [-2622.88920078, -2165.29989543, -2030.89551171, -1935.82762473,
               -1845.58860804, -1778.75346152, -1761.10452395, -1756.82255539,
               -1753.64038911, -1751.31769235, -1748.77866817, -1745.81058575,
               -1743.88910324, -1741.05621014, -1737.86539163, -1736.05263271,
               -1732.48842931, -1730.87240599, -1728.49708456, -1726.61598566,
               -1724.54669587, -1722.89507483, -1721.96259604, -1719.88484892,
               -1718.15797201, -1716.33362112, -1715.11453179, -1714.16480832,
               -1712.64782855, -1711.71690642, -1711.33342417]
optimal = -1702.8659218764612
"""

""" MAGIC with Exp kernel - np.arange(5, 306, 10):
results_hgv = [-2622.88920078 -2165.29989543 -2030.89551171 -1935.82762473
 -1845.58860804 -1778.75346152 -1761.10452395 -1756.82255539
 -1753.64038911 -1751.31769235 -1748.77866817 -1745.81058575
 -1743.88910324 -1741.05621014 -1737.86539163 -1736.05263271
 -1732.48842931 -1730.87240599 -1728.49708456 -1726.61598566
 -1724.54669587 -1722.89507483 -1721.96259604 -1719.88484892
 -1718.15797201 -1716.33362112 -1715.11453179 -1714.16480832
 -1712.64782855 -1711.71690642 -1711.33342417]
results_hgv_then_optimise = [-2012.34005544 -1870.35952549 -1814.63618322 -1781.81814063
 -1759.79239295 -1744.64846209 -1735.52586658 -1729.87674608
 -1725.10174498 -1721.70709836 -1718.36769472 -1715.75304107
 -1713.44155478 -1711.49321582 -1709.97927022 -1708.11360013
 -1707.04489305 -1705.98416789 -1705.01974408 -1704.47648932
 -1703.74840548 -1703.19588689 -1702.76273437 -1702.36497604
 -1701.93507812 -1701.61450529 -1701.30799658 -1701.01032498
 -1700.7762311  -1700.50261561 -1700.36405207]
optimal = -1702.8659218764612
"""


# TODO: Plot (start from M=15 and make xticks sparser - e.g., [50, 100, 150, 200, 250, 300])
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


""" Twonorm with Exp kernel - np.arange(5, 51, 5):
results_uniform = [-1396.70583573, -1017.94869122, -771.22872136, -605.30156586,
                   -510.48140843,  -510.19789597,  -510.03721765, -511.07506492,
                   -511.69896158,  -511.57398715]
results_kmeans = [-806.35163434, -756.38506029, -690.23999253, -576.88703554, -512.15358069,
                  -511.84802163, -511.69774876, -511.622588,   -511.60590338, -511.56933536]
results_gv = [-1014.26413527, -818.68460218, -701.12421969, -567.82354756,
              -499.54325414,  -499.43971312, -499.37827332, -499.30453848,
              -499.2595869,   -499.22006871]
results_hgv = [-1352.40125219, -872.77578924, -776.43254085, -587.7131583,
               -499.49391003,  -499.39084051, -499.32350562, -499.28439395,
               -499.245308,    -499.21059677]
optimal = -498.60123103 (infeasible, so we take the max value achieved from any iteration)
"""

# TODO: Run experiment on Colab
""" Twonorm with Exp kernel - np.arange(5, 51, 5):
results_hgv = ...
results_hgv_then_optimise = ...
optimal = ...
"""


# TODO: Plot (start at M=10)
class TwonormSparsityMetaDataset(TwonormDataset, SparsityMetaDataset):
    # Converges at a surprisingly sparse M=25, as opposed to Ringnorm which requires ~250
    def __init__(self):
        TwonormDataset.__init__(self)
        SparsityMetaDataset.__init__(self, 3, 5, 50, 10, np.arange(5, 51, 5))


""" Ringnorm with Exp kernel - np.arange(5, 306, 10):
results_uniform = [-3898.66433323, -3739.97035914, -3620.23629755, -3406.30837793,
                   -3006.92534855, -2767.35980403, -2352.11704331, -2236.35214327,
                   -2021.42079009, -2073.75616252, -1888.33317192, -1797.38724893,
                   -1742.952209,   -1667.66876573, -1686.72736108, -1596.8301633,
                   -1534.49869884, -1511.49915485, -1442.68489758, -1393.88218358,
                   -1313.49253569, -1251.62894903, -1192.22602323, -1140.83516079,
                   -1127.750226,   -1121.40126453, -1121.30799078, -1112.58198287,
                   -1110.23301563, -1111.59863358, -1112.56619262]
results_kmeans = [-3526.5080751,  -3512.32934056, -3488.19006687, -3461.50586789,
                  -3397.57441602, -3246.29982754, -2959.31789217, -2677.11597607,
                  -2424.85457623, -2103.63645971, -2161.66445391, -1825.09303177,
                  -1779.97284295, -1698.33982313, -1635.13580031, -1572.74152513,
                  -1531.10089482, -1484.75723886, -1455.88155197, -1442.23053917,
                  -1414.59601183, -1384.73593019, -1372.79699282, -1347.68948776,
                  -1340.51923467, -1318.86166008, -1309.27551358, -1296.48451548,
                  -1281.02945362, -1271.30818276, -1265.0366249]
results_gv = [-3973.53481162, -3944.63256465, -3900.51788397, -3520.95301494,
              -3382.01788093, -2705.88482369, -2775.27475406, -2566.73860189,
              -2275.23508154, -2283.89890084, -1995.83095559, -1828.61905992,
              -2009.59846106, -1793.4154635,  -1767.54063342, -1564.87398159,
              -1588.19832361, -1568.04983754, -1441.8179984,  -1357.68894854,
              -1296.78773805, -1237.88066877, -1096.15116867, -1000.21537697,
              -995.70653021,  -992.97122679,  -991.33675035,  -989.93258068,
              -988.04297943,  -986.86622979,  -986.33677597]
results_hgv = [-3957.46327061, -3889.61729723, -3839.42069026, -3293.78296912,
               -2693.66601991, -2686.00756728, -2605.8743586,  -2572.78351148,
               -2441.61584767, -2448.39278645, -2222.35870473, -2159.9753885,
               -2182.08242191, -2099.49345541, -1991.33223853, -1869.61142949,
               -1792.96284388, -1644.38974353, -1474.62795655, -1362.68887795,
               -1259.62906694, -1165.99748628, -1070.91194744, -988.26451898,
               -984.59579463,  -982.77330344,  -981.11920296,  -980.16404528,
               -978.95422732,  -978.28747759,  -977.49577389]
optimal = -968.12391203 (infeasible, so we take the max value achieved from any iteration)
"""

# TODO: ?
""" Ringnorm with Exp kernel - np.arange(5, 306, 10):
results_hgv = ...
results_hgv_then_optimise = ...
optimal = ...
"""


# TODO: Plot
# TODO: Just graph results from 105 to 305? -> more stable
class RingnormSparsityMetaDataset(RingnormDataset, SparsityMetaDataset):
    def __init__(self):
        RingnormDataset.__init__(self)
        SparsityMetaDataset.__init__(self, 3, 5, 50, 10, np.arange(5, 306, 10))
