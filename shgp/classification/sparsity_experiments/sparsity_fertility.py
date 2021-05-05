import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from shgp.inducing.initialisation_methods import reinitialise_PGPR, h_reinitialise_PGPR
from shgp.robustness.contrained_kernels import ConstrainedExpSEKernel
from shgp.models.pgpr import PGPR

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
        kernel=ConstrainedExpSEKernel(),
        inducing_variable=X.copy()
    )
    gpflow.set_trainable(pgpr.inducing_variable, False)
    opt = gpflow.optimizers.Scipy()
    try:
        for _ in range(NUM_LOCAL_ITERS):
            opt.minimize(pgpr.training_loss, variables=pgpr.trainable_variables, options=dict(maxiter=NUM_OPT_ITERS))
            pgpr.optimise_ci(num_iters=NUM_CI_ITERS)
    except tf.errors.InvalidArgumentError as error:
        msg = error.message
        if "Cholesky" not in msg and "invertible" not in msg:
            raise error
        else:
            print("Cholesky error caught, retrying...")
            return train_full_model()  # we failed due to a spurious Cholesky error, restart

    return pgpr.elbo()


# TODO: Move to train utilities
def train_reinit_model(initialisation_method, m):
    pgpr = PGPR(
        data=(X, Y),
        kernel=ConstrainedExpSEKernel()
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
    except tf.errors.InvalidArgumentError as error:
        msg = error.message
        if "Cholesky" not in msg and "invertible" not in msg:
            raise error
        else:
            print("Cholesky error caught, retrying...")
            return run_experiment(M)  # we failed due to a spurious Cholesky error, restart
    return elbo_pgpr_gv, elbo_pgpr_hgv


def plot_results(M, results, optimal):
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.tick_params(labelright=True)

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
    dataset = "../../data/fertility.txt"
    data = np.loadtxt(dataset, delimiter=",")
    X = standardise_features(data[:, :-1])
    Y = data[:, -1].reshape(-1, 1)

    # Test different numbers of inducing points
    M = np.arange(1, 31)

    NUM_CYCLES = 10
    NUM_LOCAL_ITERS = 10
    NUM_OPT_ITERS = 250
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
