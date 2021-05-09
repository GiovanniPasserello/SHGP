import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from shgp.inducing.initialisation_methods import reinitialise_PGPR, h_reinitialise_PGPR
from shgp.utilities.train_pgpr import train_pgpr
from shgp.classification.sparsity_experiments.metadata import \
    BreastCancerSparsityMetaDataset, FertilitySparsityMetaDataset, MagicSparsityMetaDataset


np.random.seed(42)
tf.random.set_seed(42)

"""
A comparison of PGPR with two different inducing point initialisation procedures. Here we investigate
the effect of the number of inducing points on the ELBO. This allows us to analyse the benefit of sparsity
which is afforded to us by the use of greedy variance / heteroscedastic greedy variance. It is important to 
note that we do not compare to SVGP Bernoulli, here - what we care about is the sparsity of inducing point 
selection (at what point does the ELBO converge). For comparisons against Bernoulli, see other experiments.
"""


def run_experiment(X, Y, M, inner_iters, opt_iters, ci_iters):
    elbo_pgpr_gv = train_pgpr(X, Y, inner_iters, opt_iters, ci_iters, init_method=reinitialise_PGPR, M=M)
    print("pgpr_gv trained: ELBO = {}".format(elbo_pgpr_gv))
    elbo_pgpr_hgv = train_pgpr(X, Y, inner_iters, opt_iters, ci_iters, init_method=h_reinitialise_PGPR, M=M)
    print("pgpr_hgv trained: ELBO = {}".format(elbo_pgpr_hgv))
    return elbo_pgpr_gv, elbo_pgpr_hgv


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
    plt.plot(M_array, results[:, 0], color="#ff7f0e", label='Greedy Variance')
    plt.plot(M_array, results[:, 1], color="#1f77b4", label='Heteroscedastic Greedy Variance')

    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    # Load data
    dataset = MagicSparsityMetaDataset()  # to test another dataset, just change this definition
    X, Y = dataset.load_data()

    ################################################
    # PGPR with Different Reinitialisation Methods #
    ################################################
    results_gv = np.zeros_like(dataset.M_array, dtype=float)
    results_hgv = np.zeros_like(dataset.M_array, dtype=float)
    for c in range(dataset.num_cycles):  # run NUM_CYCLES times and take an average
        print("Beginning cycle {}...".format(c + 1))
        for i, m in enumerate(dataset.M_array):
            print("Beginning training for", m, "inducing points...")
            res_gv, res_hgv = run_experiment(X, Y, m, dataset.inner_iters, dataset.opt_iters, dataset.ci_iters)
            results_gv[i] += res_gv
            results_hgv[i] += res_hgv
        print("Completed cycle {}.".format(c + 1))
    results_gv = results_gv / dataset.num_cycles
    results_hgv = results_hgv / dataset.num_cycles
    results = list(zip(results_gv, results_hgv))

    ##############################################
    # PGPR with Full Inducing Points ('Optimal') #
    ##############################################
    elbo_pgpr = train_pgpr(X, Y, dataset.inner_iters, dataset.opt_iters, dataset.ci_iters)
    print("pgpr trained: ELBO = {}".format(elbo_pgpr))
    optimal = np.full(len(results), elbo_pgpr)

    ############
    # Plotting #
    ############
    plot_results(dataset.name, dataset.M_array, np.array(results), optimal)

    print("Final results:")
    print("results_gv = {}".format(results_gv))
    print("results_hgv = {}".format(results_hgv))
    print("optimal = {}".format(elbo_pgpr))
