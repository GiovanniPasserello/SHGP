import gpflow
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import sigmoid

from shgp.models.pgpr import PGPR
from shgp.likelihoods.pg_bernoulli import PolyaGammaBernoulli

INDUCING_INTERVAL = 1


# Polya-Gamma uses logit link / sigmoid
def invlink(f):
    return gpflow.likelihoods.Bernoulli(invlink=sigmoid).invlink(f).numpy()


def model_comparison():
    ######################
    # Model Optimisation #
    ######################

    # SVGP (choose Bernoulli or PG likelihood)
    # likelihood = gpflow.likelihoods.Bernoulli(invlink=sigmoid)
    likelihood = PolyaGammaBernoulli()
    svgp = gpflow.models.SVGP(
        kernel=gpflow.kernels.Matern52(),
        likelihood=likelihood,
        inducing_variable=X[::INDUCING_INTERVAL].copy()
    )
    loss = svgp.training_loss_closure((X, Y))
    gpflow.set_trainable(svgp.inducing_variable, False)
    gpflow.optimizers.Scipy().minimize(loss, variables=svgp.trainable_variables)
    print("svgp trained")

    # PGPR
    pgpr = PGPR(
        data=(X, Y),
        kernel=gpflow.kernels.Matern52(),
        inducing_variable=X[::INDUCING_INTERVAL].copy()
    )
    gpflow.set_trainable(pgpr.inducing_variable, False)
    opt = gpflow.optimizers.Scipy()
    for _ in range(10):
        opt.minimize(pgpr.training_loss, variables=pgpr.trainable_variables)
        pgpr.optimise_ci(num_iters=10)
    print("pgpr trained")

    ##############
    # Prediction #
    ##############

    # Take predictions from SVGP
    X_test_mean_svgp, _ = svgp.predict_f(X_test)
    P_test_svgp = invlink(X_test_mean_svgp)
    X_train_mean_svgp, _ = svgp.predict_f(X)

    # Take predictions from PGPR
    X_test_mean_pgpr, _ = pgpr.predict_f(X_test)
    P_test_pgpr = invlink(X_test_mean_pgpr)
    X_train_mean_pgpr, _ = pgpr.predict_f(X)

    ############
    # Plotting #
    ############

    svgp_elbo = svgp.elbo((X, Y))
    pgpr_elbo = pgpr.elbo()

    # Plot squashed predictions
    plt.plot(X_test, P_test_svgp, "r", lw=1, label='SVGP ({:.2f})'.format(svgp_elbo))
    plt.plot(X_test, P_test_pgpr, "b", lw=1, label='PGPR ({:.2f})'.format(pgpr_elbo))

    # Plot data
    plt.plot(X, Y, "x", color='k', ms=7, mew=1)

    plt.ylim((-0.5, 1.5))
    plt.title('SVGP vs PGPR')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    # Load data
    X = np.genfromtxt("data/classif_1D_X.csv").reshape(-1, 1)
    Y = np.genfromtxt("data/classif_1D_Y.csv").reshape(-1, 1)
    X_test = np.linspace(0, 6, 200).reshape(-1, 1)
    # Plot params
    plt.rcParams["figure.figsize"] = (8, 4)

    model_comparison()
