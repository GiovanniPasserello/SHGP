import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

"""
A suite for plotting sparsity experiments from saved results.
"""


def collate_results():
    results_uniform = [-444.77996138, -386.91316907, -386.02783327, -385.31543353, -384.54316826,
                       -383.67183261, -382.85791003, -382.46500407, -381.84566833, -380.99304483,
                       -380.63460644, -380.55328098, -380.27110464, -379.96738077, -379.8475552,
                       -379.90802159, -379.40674155, -379.26510242, -379.15252496, -379.1945228,
                       -379.04461289, -379.1288633, -378.97580987, -378.85781204, -378.83165375,
                       -378.77207542, -378.71204709, -378.62128625, -378.41665454, -378.53594639]
    results_kmeans = [-412.01661822, -385.87567976, -385.15533115, -384.31392658, -383.22495125,
                      -382.52105175, -381.99257373, -380.89674274, -380.75191388, -380.17271875,
                      -379.39160705, -379.47724897, -379.35385998, -379.1131317, -378.93623123,
                      -378.8262073, -378.64420163, -378.6724664, -378.52242826, -378.45192916,
                      -378.30231571, -378.35490827, -378.19587052, -378.2522335, -378.20945799,
                      -378.21006778, -378.0734994, -378.05026178, -378.01002372, -377.99380257]
    results_gv = [-423.64588235, -386.59501815, -386.17788556, -385.68507796, -384.89533349,
                  -383.84076949, -382.54249711, -381.6088387, -381.27863342, -380.61205978,
                  -380.16222904, -379.75945861, -379.48135565, -379.35240091, -378.95491056,
                  -378.80356969, -378.65048198, -378.5303823, -378.40855262, -378.3538156,
                  -378.2628084, -378.16800592, -378.07626095, -378.03704864, -377.94699722,
                  -377.9417581, -377.87207947, -377.85919065, -377.80839534, -377.82408316]
    results_hgv = [-420.21889021, -386.66910472, -386.27469259, -385.7520532, -385.11243869,
                   -383.90052552, -382.9113214, -382.01660468, -381.17262278, -380.69372039,
                   -380.22351617, -379.7882304, -379.40044747, -379.2088569, -379.02888323,
                   -378.88890559, -378.65256517, -378.48233193, -378.40623597, -378.34645726,
                   -378.22606101, -378.17310304, -378.11322305, -378.05940044, -377.96486717,
                   -377.93209324, -377.91239849, -377.86808434, -377.82296594, -377.80904598]
    optimal = -377.60474770202654

    results = np.array(list(zip(results_uniform, results_kmeans, results_gv, results_hgv)))
    optimal = np.full(len(results), optimal)

    return results, optimal


def plot_results(name, M_array, results, optimal):
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.tick_params(labelright=True)

    # Axis labels
    plt.title('{} Dataset'.format(name))
    plt.ylabel('ELBO')
    plt.xlabel('Number of Inducing Points')
    # Axis limits
    plt.xlim(M_array[0], M_array[-1])
    plt.xticks(M_array)

    # Setup each subplot
    plt.plot(M_array, optimal, color='k', linestyle='dashed', label='Optimal', zorder=101)
    plt.plot(M_array, results[:, 0], label='Uniform')
    plt.plot(M_array, results[:, 1], label='K-means')
    plt.plot(M_array, results[:, 2], label='GV')
    plt.plot(M_array, results[:, 3], label='HGV')

    plt.legend(loc='lower right', prop={'size': 12})
    plt.show()


if __name__ == '__main__':
    results, optimal = collate_results()
    M_array = np.arange(5, 151, 5)
    slice_start = 1
    plot_results("Pima Diabetes", M_array[slice_start:], results[slice_start:], optimal[slice_start:])
