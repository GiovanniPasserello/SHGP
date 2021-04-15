import gpflow
import numpy as np
import tensorflow as tf

from datetime import datetime
from gpflow.models.util import inducingpoint_wrapper
from tensorflow import sigmoid

from shgp.inducing.greedy_variance import h_greedy_variance, greedy_variance
from shgp.models.pgpr import PGPR


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


def load_fertility():
    dataset = "../data/classification/fertility.txt"

    data = np.loadtxt(dataset, delimiter=",")
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1)

    NUM_INDUCING = 100
    BERN_ITERS = 100
    PGPR_ITERS = (5, 50, 5)
    GREEDY_THRESHOLD = 1e-5

    return X, Y, NUM_INDUCING, BERN_ITERS, PGPR_ITERS, GREEDY_THRESHOLD


def load_breast_cancer():
    dataset = "../data/classification/breast_cancer.txt"

    data = np.loadtxt(dataset, delimiter=",")
    X = data[:, 2:]  # the kernel goes to identity matrix unless we take logs
    Y = data[:, 1].reshape(-1, 1) - 1

    NUM_INDUCING = 194
    BERN_ITERS = 100
    PGPR_ITERS = (3, 25, 3)
    GREEDY_THRESHOLD = 1e-1

    return X, Y, NUM_INDUCING, BERN_ITERS, PGPR_ITERS, GREEDY_THRESHOLD


def load_magic():
    dataset = "../data/classification/magic.txt"

    data = np.loadtxt(dataset, delimiter=",")
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1)

    NUM_INDUCING = 200
    BERN_ITERS = 100
    PGPR_ITERS = (5, 25, 5)
    GREEDY_THRESHOLD = 100

    return X, Y, NUM_INDUCING, BERN_ITERS, PGPR_ITERS, GREEDY_THRESHOLD


def classification_demo():
    ########
    # SVGP #
    ########

    inducing_idx1 = np.random.choice(np.arange(X.shape[0]), size=NUM_INDUCING, replace=False)
    inducing_vars1 = gpflow.inducing_variables.InducingPoints(X[inducing_idx1])
    svgp = gpflow.models.SVGP(
        kernel=gpflow.kernels.SquaredExponential(),
        likelihood=gpflow.likelihoods.Bernoulli(invlink=sigmoid),
        inducing_variable=inducing_vars1
    )

    svgp_start = datetime.now()
    gpflow.optimizers.Scipy().minimize(
        svgp.training_loss_closure((X, Y)),
        variables=svgp.trainable_variables,
        options=dict(maxiter=SVGP_ITERS)
    )
    svgp_time = datetime.now() - svgp_start

    print("svgp trained in {:.2f} seconds".format(svgp_time.total_seconds()))
    print("ELBO = {:.6f}".format(svgp.elbo((X,Y))))

    ########
    # PGPR #
    ########

    # Greedy variance selection
    kernel = gpflow.kernels.SquaredExponential()
    pgpr = PGPR(data=(X,Y), kernel=kernel)

    pgpr_start = datetime.now()

    # Inducing point selection comparison on MAGIC dataset.
    # h_greedy vs greedy gives (num_ind,ELBO,time(s)):
    # The main results are found using thresholds (inducing point selection early stopping)
    # Times vary largely between runs depending on laptop usage
    # h_greedy performs better for all equal sizes of inducing points
    # h_greedy can beat greedy with fewer points
    # h_greedy - [(200/172,-6606.2547,206.43), (100/61,-6771.3247,92.56), (50,-7054.9747,49.11), (30,-7813.4151,47.04), (10,-8767.7210,10.08)]
    # greedy - [(200,-6616.7805,279.80), (100,-6724.6128,126.72), (50,-7100.9152,64.38), (30,-8276.1728,47.88), (10,-8778.8325,25.00)]
    # h_greedy forced number of points for comparison: [(200,-6538.0476,278.66), (100,-6691.8442,159.38)]

    theta_inv = tf.math.reciprocal(pgpr.likelihood.compute_theta())
    inducing_locs2, inducing_idx2 = h_greedy_variance(X, theta_inv, NUM_INDUCING, kernel, GREEDY_THRESHOLD)
    inducing_vars2 = gpflow.inducing_variables.InducingPoints(inducing_locs2)
    pgpr.inducing_variable = inducingpoint_wrapper(inducing_vars2)
    gpflow.set_trainable(pgpr.inducing_variable, False)
    prev_elbo = pgpr.elbo()

    iter_limit = 10  # to avoid infinite loops
    opt = gpflow.optimizers.Scipy()
    while True:
        # Optimize model
        for _ in range(PGPR_ITERS[0]):
            opt.minimize(pgpr.training_loss, variables=pgpr.trainable_variables, options=dict(maxiter=PGPR_ITERS[1]))
            pgpr.optimise_ci(PGPR_ITERS[2])

        next_elbo = pgpr.elbo()
        print("Previous ELBO: {}, Next ELBO: {}".format(prev_elbo, next_elbo))
        if np.abs(next_elbo - prev_elbo) <= 1e-3 or iter_limit == 0:
            break

        theta_inv = tf.math.reciprocal(pgpr.likelihood.compute_theta())
        inducing_locs2, inducing_idx2 = h_greedy_variance(X, theta_inv, NUM_INDUCING, kernel, GREEDY_THRESHOLD)
        inducing_vars2 = gpflow.inducing_variables.InducingPoints(inducing_locs2)
        pgpr.inducing_variable = inducingpoint_wrapper(inducing_vars2)
        gpflow.set_trainable(pgpr.inducing_variable, False)

        prev_elbo = next_elbo
        iter_limit -= 1

    pgpr_time = datetime.now() - pgpr_start
    pgpr.elbo()

    print("pgpr trained in {:.2f} seconds".format(pgpr_time.total_seconds()))
    print("Final number of inducing points: {}".format(pgpr.inducing_variable.num_inducing))
    print("ELBO = {:.6f}".format(pgpr.elbo()))


if __name__ == '__main__':
    X, Y, NUM_INDUCING, SVGP_ITERS, PGPR_ITERS, GREEDY_THRESHOLD = load_magic()
    X = standardise_features(X)

    classification_demo()
