import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from shgp.inducing.initialisation_methods import uniform_subsample
from shgp.likelihoods.pg_bernoulli import PolyaGammaBernoulli
from shgp.utilities.general import invlink
from shgp.utilities.train_pgpr import train_pgpr

np.random.seed(0)
tf.random.set_seed(0)


def model_comparison():
    ######################
    # Model Optimisation #
    ######################

    num_inducing = 10

    # PGPR
    pgpr, pgpr_elbo = train_pgpr(
        X, Y,
        10, 1000, 10,
        kernel=gpflow.kernels.Matern52(),
        M=num_inducing,
        init_method=uniform_subsample
    )
    print("pgpr trained")

    # TODO: This comparison of Bernoulli vs PG is worth showing in `Evaluation'.
    # SVGP (choose Bernoulli or PG likelihood for comparison)
    # likelihood = gpflow.likelihoods.Bernoulli(invlink=tf.sigmoid)
    likelihood = PolyaGammaBernoulli()
    svgp = gpflow.models.SVGP(
        kernel=gpflow.kernels.Matern52(),
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
    X_train_mean_pgpr, _ = pgpr.predict_f(X)

    # Take predictions from SVGP
    X_test_mean_svgp, _ = svgp.predict_f(X_test)
    P_test_svgp = invlink(X_test_mean_svgp)
    X_train_mean_svgp, _ = svgp.predict_f(X)

    ############
    # Plotting #
    ############

    # Plot squashed predictions
    plt.plot(X_test, P_test_pgpr, "b", lw=1, label='PGPR ({:.2f})'.format(pgpr_elbo), zorder=101)
    plt.plot(X_test, P_test_svgp, "r", lw=1, label='SVGP ({:.2f})'.format(svgp_elbo))

    # Plot data
    plt.plot(X, Y, "x", color='k', ms=7, mew=1)

    plt.ylim((-0.5, 1.5))
    plt.title('SVGP vs PGPR')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    # Load data
    X = np.genfromtxt("../../data/toy/classif_1D_X.csv").reshape(-1, 1)
    Y = np.genfromtxt("../../data/toy/classif_1D_Y.csv").reshape(-1, 1)
    X_test = np.linspace(0, 6, 200).reshape(-1, 1)
    # Plot params
    plt.rcParams["figure.figsize"] = (8, 4)

    model_comparison()
