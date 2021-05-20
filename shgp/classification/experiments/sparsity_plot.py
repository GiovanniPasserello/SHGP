import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

"""
A suite for plotting sparsity experiments from saved results.
"""


def collate_results():
    results_uniform = [-252.13643516, -208.20541972, -157.02072046, -140.62518881, -131.01991315, -126.45785361,
                       -125.58773707, -123.02937557, -122.18254487, -121.02846546]
    results_kmeans = [-261.4396643, -148.19500976, -134.64923173, -128.44973186, -123.8891034, -122.41569379,
                      -121.3741855, -120.59062521, -120.25771766, -120.03329343]
    results_gv = [-263.96390801, -193.47278328, -146.06442775, -131.69559018, -125.21248857, -122.07170094,
                  -120.68262996, -120.08218722, -119.87603769, -119.79144935]
    results_hgv = [-265.21812865, -199.13950339, -146.41359466, -132.61964685, -124.94452836,
                   -122.44237289, -120.67969943, -120.04857654, -119.89991836, -119.80383267]
    optimal = -119.74156303725914

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
    M_array = np.arange(5, 51, 5)
    slice_start = 0
    plot_results("Banana", M_array[slice_start:], results[slice_start:], optimal[slice_start:])
