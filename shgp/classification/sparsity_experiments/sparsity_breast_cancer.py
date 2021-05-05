import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from shgp.inducing.initialisation_methods import reinitialise_PGPR, h_reinitialise_PGPR
from shgp.robustness.contrained_kernels import ConstrainedSigmoidSEKernel
from shgp.models.pgpr import PGPR

np.random.seed(0)
tf.random.set_seed(0)


"""
A comparison of PGPR with two different inducing point initialisation procedures. Here we investigate
the effect of the number of inducing points on the ELBO. This allows us to analyse the benefit of sparsity
which is afforded to us by the use of greedy variance / heteroscedastic greedy variance. It is important to 
note that we do not compare to SVGP Bernoulli, here - what we care about is the sparsity of inducing point 
selection (at what point does the ELBO converge). For comparisons against Bernoulli, see other experiments.

M = np.arange(5, 101, 5)
results_gv = [-217.88657076  -96.73570866  -87.8114045   -82.66026768  -79.86418064
  -78.34576833  -77.62225389  -77.01309174  -76.54907791  -76.2119282
  -75.97489667  -75.87565941  -75.76426664  -75.68699773  -75.60315725
  -75.55829306  -75.52519887  -75.49966281  -75.46643782  -75.44281655]
results_hgv = [-213.54499313  -98.33300655  -86.97868378  -82.22728811  -79.00359283
  -77.55862879  -76.87227229  -76.43349924  -76.13059877  -75.87849282
  -75.7657115   -75.69325919  -75.60631159  -75.53840403  -75.51047344
  -75.4812239   -75.45382388  -75.43568348  -75.42589175  -75.40969307]
optimal = -75.52529596850775
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
        kernel=ConstrainedSigmoidSEKernel(max_lengthscale=1000.0, max_variance=1000.0),
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
        kernel=ConstrainedSigmoidSEKernel(max_lengthscale=1000.0, max_variance=1000.0)
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
    try:
        elbo_pgpr_gv = train_reinit_model(reinitialise_PGPR, M)
        print("pgpr_gv trained: ELBO = {}".format(elbo_pgpr_gv))
        elbo_pgpr_hgv = train_reinit_model(h_reinitialise_PGPR, M)
        print("pgpr_hgv trained: ELBO = {}".format(elbo_pgpr_hgv))
        return elbo_pgpr_gv, elbo_pgpr_hgv
    except Exception:
        # The exception is due to a rare/random inversion error.
        # We want to keep retrying until it succeeds.
        print("Exception caught, retrying!")
        return run_experiment(M)


def plot_results(M, results, optimal):
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.tick_params(labelright=True)

    # Axis labels
    plt.title('Comparison of Inducing Point Methods - Breast Cancer Dataset')
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
    dataset = "../../data/breast-cancer-diagnostic.txt"
    data = np.loadtxt(dataset, delimiter=",")
    X = standardise_features(data[:, 2:])
    Y = data[:, 1].reshape(-1, 1)

    # Test different numbers of inducing points
    M = np.arange(5, 101, 5)

    NUM_CYCLES = 3
    NUM_LOCAL_ITERS = 10
    NUM_OPT_ITERS = 100
    NUM_CI_ITERS = 10

    ################################################
    # PGPR with Different Reinitialisation Methods #
    ################################################
    results_gv = np.zeros_like(M, dtype=float)
    results_hgv = np.zeros_like(M, dtype=float)
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
