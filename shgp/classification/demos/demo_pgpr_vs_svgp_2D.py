import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from shgp.data.dataset import BananaDataset
from shgp.likelihoods.pg_bernoulli import PolyaGammaBernoulli
from shgp.utilities.general import invlink
from shgp.utilities.train_pgpr import train_pgpr

np.random.seed(0)
tf.random.set_seed(0)


def model_comparison():
    ######################
    # Model Optimisation #
    ######################

    # PGPR
    pgpr, pgpr_elbo = train_pgpr(
        X, Y,
        20, 2000, 20,
        kernel_type=gpflow.kernels.SquaredExponential
    )
    print("pgpr trained")

    # TODO: This comparison of Bernoulli vs PG is worth showing in `Evaluation'.
    # SVGP (choose Bernoulli or PG likelihood for comparison)
    likelihood = gpflow.likelihoods.Bernoulli(invlink=tf.sigmoid)
    # likelihood = PolyaGammaBernoulli()
    svgp = gpflow.models.SVGP(
        kernel=gpflow.kernels.SquaredExponential(),
        likelihood=likelihood,
        inducing_variable=pgpr.inducing_variable.Z
    )
    gpflow.set_trainable(svgp.inducing_variable, False)
    gpflow.optimizers.Scipy().minimize(svgp.training_loss_closure((X, Y)), variables=svgp.trainable_variables)
    svgp_elbo = svgp.elbo((X, Y))
    print("svgp trained")

    ##############
    # Prediction #
    ##############

    # Take predictions from PGPR
    X_test_mean_pgpr, _ = pgpr.predict_f(X_test)
    P_test_pgpr = invlink(X_test_mean_pgpr)

    # Take predictions from SVGP
    X_test_mean_svgp, _ = svgp.predict_f(X_test)
    P_test_svgp = invlink(X_test_mean_svgp)

    ############
    # Plotting #
    ############

    # Plot data
    plt.plot(X[mask, 0], X[mask, 1], "oC0", mew=0, alpha=0.5)
    plt.plot(X[~mask, 0], X[~mask, 1], "oC1", mew=0, alpha=0.5)

    # Plot PGPR decision boundary
    c1 = plt.contour(
        *X_grid,
        P_test_pgpr.reshape(NUM_TEST_INDICES, NUM_TEST_INDICES),
        [0.5],  # p=0.5 decision boundary
        colors="b",
        linewidths=1.8,
        zorder=101
    )

    # Plot SVGP decision boundary
    c2 = plt.contour(
        *X_grid,
        P_test_svgp.reshape(NUM_TEST_INDICES, NUM_TEST_INDICES),
        [0.5],  # p=0.5 decision boundary
        colors="r",
        linewidths=1.8,
        zorder=100
    )

    c1.collections[0].set_label('PGPR ({:.2f})'.format(pgpr_elbo))
    c2.collections[0].set_label('SVGP ({:.2f})'.format(svgp_elbo))

    plt.xlabel('x1')
    plt.ylabel('x2', rotation=1)
    plt.title('PGPR vs SVGP - Banana Dataset')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    # Load data
    X, Y = BananaDataset().load_data()
    mask = Y[:, 0] == 1

    # Test data
    NUM_TEST_INDICES = 100
    X_range = np.linspace(-3, 3, NUM_TEST_INDICES)
    X_grid = np.meshgrid(X_range, X_range)
    X_test = np.asarray(X_grid).transpose([1, 2, 0]).reshape(-1, 2)

    # Plot params
    plt.rcParams["figure.figsize"] = (7, 7)

    model_comparison()
