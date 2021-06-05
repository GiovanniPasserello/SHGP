import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from shgp.data.dataset import PlatformDataset
from shgp.utilities.general import invlink
from shgp.utilities.train_pgpr import train_pgpr

np.random.seed(0)
tf.random.set_seed(0)

"""
Comparison of non-sparse PGPR and SVGP on the 'platform' dataset.
We plot the datapoints and the predictive decision boundaries.
We denote the ELBOs achieved by each model in the legend.
"""


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

    # SVGP
    likelihood = gpflow.likelihoods.Bernoulli(invlink=tf.sigmoid)
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

    # Meta
    plt.title('PGPR vs SVGP - Platform Dataset')
    plt.xlim((-2, 2))
    plt.xlabel('x')
    plt.ylim((-0.5, 1.5))
    plt.yticks([0, 0.5, 1])
    plt.ylabel('p( y=1 | x )')

    # Display
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    # Load data
    X, Y = PlatformDataset().load_data()
    X_test = np.linspace(-2, 2, 200).reshape(-1, 1)

    # Plot params
    plt.rcParams["figure.figsize"] = (8, 4)

    model_comparison()
