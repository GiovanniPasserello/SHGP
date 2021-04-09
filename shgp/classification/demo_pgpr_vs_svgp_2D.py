import gpflow
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import sigmoid

from shgp.models.pgpr import PGPR
from shgp.likelihoods.pg_bernoulli import PolyaGammaBernoulli

INDUCING_INTERVAL = 20


# Polya-Gamma uses logit link / sigmoid
def invlink(f):
    return gpflow.likelihoods.Bernoulli(invlink=sigmoid).invlink(f).numpy()


def model_comparison():
    ######################
    # Model Optimisation #
    ######################

    # SVGP (choose Bernoulli or PG likelihood)
    #likelihood = gpflow.likelihoods.Bernoulli(invlink=sigmoid)
    likelihood = PolyaGammaBernoulli()
    svgp = gpflow.models.SVGP(
        kernel=gpflow.kernels.SquaredExponential(),
        likelihood=likelihood,
        inducing_variable=X[::INDUCING_INTERVAL].copy()
    )
    gpflow.set_trainable(svgp.inducing_variable, False)
    gpflow.optimizers.Scipy().minimize(svgp.training_loss_closure((X, Y)), variables=svgp.trainable_variables)
    print("svgp trained")

    # PGPR
    pgpr = PGPR(
        data=(X, Y),
        kernel=gpflow.kernels.SquaredExponential(),
        inducing_variable=X[::INDUCING_INTERVAL].copy()
    )
    gpflow.set_trainable(pgpr.inducing_variable, False)
    opt = gpflow.optimizers.Scipy()
    for _ in range(20):
        opt.minimize(pgpr.training_loss, variables=pgpr.trainable_variables, options=dict(maxiter=250))
        pgpr.optimise_ci()
    print("pgpr trained")

    ##############
    # Prediction #
    ##############

    # Take predictions from SVGP
    X_test_mean_svgp, _ = svgp.predict_f(X_test)
    P_test_svgp = invlink(X_test_mean_svgp)

    # Take predictions from PGPR
    X_test_mean_pgpr, _ = pgpr.predict_f(X_test)
    P_test_pgpr = invlink(X_test_mean_pgpr)

    ############
    # Plotting #
    ############

    # Plot data
    plt.plot(X[mask, 0], X[mask, 1], "oC0", mew=0, alpha=0.5)
    plt.plot(X[~mask, 0], X[~mask, 1], "oC1", mew=0, alpha=0.5)

    # Plot SVGP decision boundary
    c1 = plt.contour(
        *X_grid,
        P_test_svgp.reshape(NUM_TEST_INDICES, NUM_TEST_INDICES),
        [0.5],  # p=0.5 decision boundary
        colors="r",
        linewidths=1.8,
        zorder=100
    )

    # Plot PGPR decision boundary
    c2 = plt.contour(
        *X_grid,
        P_test_pgpr.reshape(NUM_TEST_INDICES, NUM_TEST_INDICES),
        [0.5],  # p=0.5 decision boundary
        colors="b",
        linewidths=1.8,
        zorder=100
    )

    svgp_elbo = svgp.elbo((X, Y))
    pgpr_elbo = pgpr.elbo()

    c1.collections[0].set_label('SVGP ({:.2f})'.format(svgp_elbo))
    c2.collections[0].set_label('PGPR ({:.2f})'.format(pgpr_elbo))

    plt.title('SVGP vs PGPR')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    # Load data
    X = np.loadtxt("data/banana_X.csv", delimiter=",")
    Y = np.loadtxt("data/banana_Y.csv").reshape(-1, 1)
    mask = Y[:, 0] == 1
    # Test data
    NUM_TEST_INDICES = 40
    X_range = np.linspace(-3, 3, NUM_TEST_INDICES)
    X_grid = np.meshgrid(X_range, X_range)
    X_test = np.asarray(X_grid).transpose([1, 2, 0]).reshape(-1, 2)
    # Plot params
    plt.rcParams["figure.figsize"] = (7, 7)

    model_comparison()
