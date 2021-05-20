import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

"""
A suite for plotting sparsity experiments from saved results.
"""


def collate_results():
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
    optimal = -968.12391203

    results = np.array(list(zip(results_uniform, results_kmeans, results_gv, results_hgv)))
    optimal = np.full(len(results), optimal)

    return results, optimal


def plot_results(name, M_array, results, optimal):
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.tick_params(labelright=True)

    # Axis labels
    plt.title('Comparison of Inducing Point Methods - {} Dataset'.format(name))
    plt.ylabel('ELBO')
    plt.xlabel('Number of Inducing Points')
    # Axis limits
    plt.xlim(M_array[0], M_array[-1])
    plt.xticks(M_array)

    # Setup each subplot
    plt.plot(M_array, optimal, color='k', linestyle='dashed', label='Optimal')
    plt.plot(M_array, results[:, 0], label='Uniform')
    plt.plot(M_array, results[:, 1], label='K-means')
    plt.plot(M_array, results[:, 2], label='Greedy Variance')
    plt.plot(M_array, results[:, 3], label='Heteroscedastic Greedy Variance')

    plt.legend(loc='lower right', prop={'size': 11})
    plt.show()


if __name__ == '__main__':
    results, optimal = collate_results()
    M_array = np.arange(5, 306, 10)
    slice_start = 10
    plot_results("Ringnorm", M_array[slice_start:], results[slice_start:], optimal[slice_start:])
