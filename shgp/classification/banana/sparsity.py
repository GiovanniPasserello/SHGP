import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from shgp.inducing.initialisation_methods import reinitialise_PGPR, h_reinitialise_PGPR
from shgp.kernels.ConstrainedSEKernel import ConstrainedSEKernel
from shgp.models.pgpr import PGPR

np.random.seed(42)
tf.random.set_seed(42)


"""
A comparison of PGPR with two different inducing point initialisation procedures. Here we investigate
the effect of the number of inducing points on the ELBO. This allows us to analyse the benefit of sparsity
which is afforded to us by the use of greedy variance / heteroscedastic greedy variance. It is important to 
note that we do not compare to SVGP Bernoulli, here - what we care about is the sparsity of inducing point 
selection (at what point does the ELBO converge). For comparisons against Bernoulli, see other experiments.

Note also that this is a very small-scale problem and so the benefits are less visible. For a more concrete 
analysis, compare and constrast the results from other dataset.

M = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
results_gv = [-264.66666667 -198.33333333 -145.66666667 -132.33333333 -125.66666667
 -122.         -121.         -120.         -120.         -120.
 -120.         -120.         -120.         -120.         -120.
 -120.         -120.         -120.         -120.         -120.        ]
results_hgv = [-265.66666667 -203.         -145.66666667 -133.33333333 -125.66666667
 -122.33333333 -121.         -120.         -120.         -120.
 -120.         -120.         -120.         -120.         -120.
 -120.         -120.         -120.         -120.         -120.        ]
optimal = -120.29952203029663

M = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
results_gv = [-265.         -215.         -146.         -131.33333333 -125.33333333
 -122.         -121.         -120.         -120.         -120.        ]
results_hgv = [-266.         -198.         -146.33333333 -132.66666667 -125.66666667
 -122.33333333 -121.         -120.         -120.         -120.        ]
optimal = -120.29952203029663
"""


def train_full_model():
    pgpr = PGPR(
        data=(X, Y),
        kernel=ConstrainedSEKernel(),
        inducing_variable=X.copy()
    )
    gpflow.set_trainable(pgpr.inducing_variable, False)
    opt = gpflow.optimizers.Scipy()
    for _ in range(NUM_LOCAL_ITERS):
        opt.minimize(pgpr.training_loss, variables=pgpr.trainable_variables, options=dict(maxiter=NUM_OPT_ITERS))
        pgpr.optimise_ci(num_iters=NUM_CI_ITERS)
    return pgpr.elbo()


def train_reinit_model(initialisation_method, m):
    pgpr = PGPR(
        data=(X, Y),
        kernel=ConstrainedSEKernel()
    )
    opt = gpflow.optimizers.Scipy()

    if m == len(Y):  # if we use full dataset, don't use inducing point selection
        gpflow.set_trainable(pgpr.inducing_variable, False)

        # Optimize model
        for _ in range(NUM_LOCAL_ITERS):
            opt.minimize(pgpr.training_loss, variables=pgpr.trainable_variables, options=dict(maxiter=NUM_OPT_ITERS))
            pgpr.optimise_ci(num_iters=NUM_CI_ITERS)

        return pgpr.elbo()
    else:
        prev_elbo = pgpr.elbo()
        iter_limit = 10
        elbos = []
        while True:
            # Reinitialise inducing points
            initialisation_method(pgpr, X, m)

            # Optimize model
            for _ in range(NUM_LOCAL_ITERS):
                opt.minimize(pgpr.training_loss, variables=pgpr.trainable_variables, options=dict(maxiter=NUM_OPT_ITERS))
                pgpr.optimise_ci(num_iters=NUM_CI_ITERS)

            # Check convergence
            next_elbo = pgpr.elbo()
            elbos.append(next_elbo)
            if np.abs(next_elbo - prev_elbo) <= 1e-3 or iter_limit == 0:
                break
            prev_elbo = next_elbo
            iter_limit -= 1

        return np.max(elbos)


def run_experiment(M):
    elbo_pgpr_gv = train_reinit_model(reinitialise_PGPR, M)
    print("pgpr_gv trained: ELBO = {}".format(elbo_pgpr_gv))
    elbo_pgpr_hgv = train_reinit_model(h_reinitialise_PGPR, M)
    print("pgpr_hgv trained: ELBO = {}".format(elbo_pgpr_hgv))
    return elbo_pgpr_gv, elbo_pgpr_hgv


def plot_results(M, results, optimal):
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.tick_params(labelright=True)

    # Axis labels
    plt.title('Comparison of Inducing Point Methods - Banana Dataset')
    plt.ylabel('ELBO')
    plt.xlabel('Number of Inducing Points')
    # Axis limits
    plt.ylim(-285, -110)
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
    X = np.loadtxt("../../data/toy/banana_X.csv", delimiter=",")
    Y = np.loadtxt("../../data/toy/banana_Y.csv").reshape(-1, 1)

    # Test different numbers of inducing points
    M = np.arange(5, 51, 5)  # np.arange(5, 101, 5)

    NUM_CYCLES = 3
    NUM_LOCAL_ITERS = 10
    NUM_OPT_ITERS = 250
    NUM_CI_ITERS = 10

    ################################################
    # PGPR with Different Reinitialisation Methods #
    ################################################
    results_gv = np.zeros_like(M)
    results_hgv = np.zeros_like(M)
    for c in range(NUM_CYCLES):  # run 3 times and take an average
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
    elbo_pgpr = train_full_model()
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
