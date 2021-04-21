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

M = np.arange(5, 306, 10)
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


# TODO: Move to utilities
def standardise_features(data):
    """
    Standardise all features to 0 mean and unit variance.

    :param: data - the input data.
    :return: the normalised data.
    """
    data_means = data.mean(axis=0)  # mean value per feature
    data_stds = data.std(axis=0)  # standard deviation per feature

    # standardise each feature
    return (data - data_means) / data_stds


# TODO: Move to train utilities
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


# TODO: Move to train utilities
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


def plot_results(M, results, optimal=None):
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.tick_params(labelright=True)

    # Axis labels
    plt.title('Comparison of Inducing Point Methods - Magic Dataset')
    plt.ylabel('ELBO')
    plt.xlabel('Number of Inducing Points')
    # Axis limits
    plt.xlim(M[0], M[-1])

    # Setup each subplot
    if optimal is not None:
        plt.plot(M, optimal, color='k', linestyle='dashed', label='Optimal')
    plt.plot(M, results[:, 0], color="#ff7f0e", label='Greedy Variance')
    plt.plot(M, results[:, 1], color="#1f77b4", label='Heteroscedastic Greedy Variance')

    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    # Load data
    dataset = "../data/classification/magic.txt"
    data = np.loadtxt(dataset, delimiter=",")
    X = standardise_features(data[:, :-1])
    Y = data[:, -1].reshape(-1, 1)
    N = len(Y)

    # Prune the dataset, otherwise computations are infeasible
    random_subset = np.random.choice(N, N // 4)  # This is 4755/19020 datapoints of the total dataset
    X = X[random_subset]
    Y = Y[random_subset]

    # Test different numbers of inducing points
    M = np.arange(5, 306, 10)

    NUM_CYCLES = 3
    NUM_LOCAL_ITERS = 5
    NUM_OPT_ITERS = 50
    NUM_CI_ITERS = 10

    ################################################
    # PGPR with Different Reinitialisation Methods #
    ################################################
    results_gv = np.zeros_like(M, dtype=np.float)
    results_hgv = np.zeros_like(M, dtype=np.float)
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
