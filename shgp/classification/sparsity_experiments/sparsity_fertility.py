import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from shgp.data.dataset import FertilityDataset
from shgp.inducing.initialisation_methods import reinitialise_PGPR, h_reinitialise_PGPR
from shgp.utilities.train_pgpr import train_pgpr


np.random.seed(42)
tf.random.set_seed(42)

# TODO: Move duplicate sparsity experiments to a common file

"""
A comparison of PGPR with two different inducing point initialisation procedures. Here we investigate
the effect of the number of inducing points on the ELBO. This allows us to analyse the benefit of sparsity
which is afforded to us by the use of greedy variance / heteroscedastic greedy variance. It is important to 
note that we do not compare to SVGP Bernoulli, here - what we care about is the sparsity of inducing point 
selection (at what point does the ELBO converge). For comparisons against Bernoulli, see other experiments.

Note also that this is a very small-scale problem and so the benefits are less visible. For a more concrete 
analysis, compare and contrast the results from other dataset.

Average over 3 runs with Sigmoid kernel:
M = [1...30]
results_gv = [-39.35468916 -39.35466383 -39.3546598  -39.35465601 -39.35465262
 -39.35465383 -39.35465009 -39.35465064 -39.35464906 -39.35464906
 -39.3546483  -39.35464857 -39.35465128 -39.35464744 -39.35464725
 -39.35464665 -39.35464743 -39.35464755 -39.35464867 -39.35464871
 -39.3546472  -39.354646   -39.35464609 -39.35464577 -39.35464597
 -39.35464575 -39.35464578 -39.35464784 -39.35464561 -39.35464615]
results_hgv = [-39.35469343 -39.35466487 -39.35466136 -39.35465743 -39.35465314
 -39.35465078 -39.35465207 -39.35465045 -39.35464854 -39.35464787
 -39.35464769 -39.35464721 -39.35464693 -39.35464723 -39.35464695
 -39.3546474  -39.35464833 -39.35464691 -39.3546475  -39.35465055
 -39.35464638 -39.35464629 -39.35464583 -39.35464599 -39.35464592
 -39.3546458  -39.35464564 -39.35464764 -39.35464556 -39.35464551]
optimal = -39.35464404913119

Average over 10 runs with Exp kernel (beware of large outliers):
M = [1...30]
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


def run_experiment(M, inner_iters=10, opt_iters=250, ci_iters=10):
    elbo_pgpr_gv = train_pgpr(X, Y, inner_iters, opt_iters, ci_iters, init_method=reinitialise_PGPR, M=M)
    print("pgpr_gv trained: ELBO = {}".format(elbo_pgpr_gv))
    elbo_pgpr_hgv = train_pgpr(X, Y, inner_iters, opt_iters, ci_iters, init_method=h_reinitialise_PGPR, M=M)
    print("pgpr_hgv trained: ELBO = {}".format(elbo_pgpr_hgv))
    return elbo_pgpr_gv, elbo_pgpr_hgv


def plot_results(M, results, optimal):
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.tick_params(labelright=True)

    # TODO: Abstract, and add template for dataset name
    # Axis labels
    plt.title('Comparison of Inducing Point Methods - Fertility Dataset')
    plt.ylabel('ELBO')
    plt.xlabel('Number of Inducing Points')
    # Axis limits
    plt.xlim(M[0], M[-1])
    plt.xticks(M)

    # Setup each subplot
    plt.plot(M, optimal, color='k', linestyle='dashed', label='Optimal')
    plt.plot(M, results[:, 0], color="#ff7f0e", label='Greedy Variance')
    plt.plot(M, results[:, 1], color="#1f77b4", label='Heteroscedastic Greedy Variance')

    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    # Load data
    # TODO: Add SparsityMetaDataset class to abstract away training params and M array
    X, Y = FertilityDataset().load_data()

    # Test different numbers of inducing points
    M = np.arange(1, 31)

    NUM_CYCLES = 10

    ################################################
    # PGPR with Different Reinitialisation Methods #
    ################################################
    results_gv = np.zeros_like(M, dtype=float)
    results_hgv = np.zeros_like(M, dtype=float)
    for c in range(NUM_CYCLES):  # run NUM_CYCLES times and take an average
        print("Beginning cycle {}...".format(c + 1))
        for i, m in enumerate(M):
            print("Beginning training for", m, "inducing points...")
            res_gv, res_hgv = run_experiment(m)
            results_gv[i] += res_gv
            results_hgv[i] += res_hgv
        print("Completed cycle {}.".format(c + 1))
    results_gv = results_gv / NUM_CYCLES
    results_hgv = results_hgv / NUM_CYCLES
    results = list(zip(results_gv, results_hgv))

    ##############################################
    # PGPR with Full Inducing Points ('Optimal') #
    ##############################################
    elbo_pgpr = train_pgpr(X, Y, 10, 250, 10)
    print("pgpr trained: ELBO = {}".format(elbo_pgpr))
    optimal = np.full(len(results), elbo_pgpr)

    ############
    # Plotting #
    ############
    plot_results(M, np.array(results), optimal)

    print("Final results:")
    print("results_gv = {}".format(results_gv))
    print("results_hgv = {}".format(results_hgv))
    print("optimal = {}".format(elbo_pgpr))
