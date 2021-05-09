import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from shgp.inducing.initialisation_methods import uniform_subsample
from shgp.likelihoods.pg_bernoulli import PolyaGammaBernoulli
from shgp.models.pgpr import PGPR
from shgp.utilities.utils import invlink

np.random.seed(42)
tf.random.set_seed(42)

"""
A comparison of SVGP with a Polya Gamma likelihood and PGPR. The inducing points are fixed, but gradient-based
optimisation of both models will converge to the same results.

ELBO results for M = [4, 8, 16, 32, 64, 400]:

These results slightly differ because of the differing amounts of jitter.
As I added robust_cholesky to pgpr, it only uses jitter if needed and often attains a better ELBO.
svgp_pg = [-234.8614, -229.7986, -174.8937, -125.2562, -120.9208, -120.2989]
pgpr    = [-234.8613, -229.7983, -173.5446, -125.2301, -120.8549, -120.2990]
"""


def run_experiment(M):
    initial_inducing_inputs, _ = uniform_subsample(X, M)

    ################
    # Optimisation #
    ################

    #############################
    # SVGP PG likelihood #
    #############################
    svgp_pg = gpflow.models.SVGP(
        kernel=gpflow.kernels.SquaredExponential(),
        likelihood=PolyaGammaBernoulli(),
        inducing_variable=initial_inducing_inputs.copy()
    )
    gpflow.set_trainable(svgp_pg.inducing_variable, False)
    gpflow.optimizers.Scipy().minimize(
        svgp_pg.training_loss_closure((X, Y)),
        variables=svgp_pg.trainable_variables
    )
    print("svgp_pg trained: ELBO = {}".format(svgp_pg.elbo((X, Y))))

    ########
    # PGPR #
    ########
    pgpr = PGPR(
        data=(X, Y),
        kernel=gpflow.kernels.SquaredExponential(),
        inducing_variable=initial_inducing_inputs.copy()
    )
    gpflow.set_trainable(pgpr.inducing_variable, False)
    opt = gpflow.optimizers.Scipy()
    for _ in range(20):
        opt.minimize(pgpr.training_loss, variables=pgpr.trainable_variables)
        pgpr.optimise_ci(num_iters=20)
    print("pgpr trained: ELBO = {}".format(pgpr.elbo()))

    ##############
    # Prediction #
    ##############

    # Take predictions from SVGP PG
    X_test_mean_svgp_pg, _ = svgp_pg.predict_f(X_test)
    P_test_svgp_pg = invlink(X_test_mean_svgp_pg)

    # Take predictions from PGPR HGV
    X_test_mean_pgpr, _ = pgpr.predict_f(X_test)
    P_test_pgpr = invlink(X_test_mean_pgpr)

    # Collate results
    results_svgp_pg = (P_test_svgp_pg, svgp_pg.inducing_variable.Z.variables[0])
    results_pgpr = (P_test_pgpr, pgpr.inducing_variable.Z.variables[0])

    return results_svgp_pg, results_pgpr


def plot_results(M, results):
    plt.rcParams["figure.figsize"] = (8, 2.666)
    fig, axes = plt.subplots(2, len(M), squeeze=False)

    # Setup labels
    models = ['SVGP PG', 'PGPR']
    for axis, m in zip(axes[0], M):
        axis.set_title('M = {}'.format(m))
    for subplot, model in zip(axes[:, 0], models):
        subplot.set_ylabel(model)

    # Setup each subplot
    for axis in axes:
        for subplot in axis:
            # Styling
            subplot.set_xticklabels([]), subplot.set_xticks([])
            subplot.set_yticklabels([]), subplot.set_yticks([])
            subplot.set_aspect('equal')
            # Plot the data
            subplot.plot(X[mask, 0], X[mask, 1], "oC0", mew=0, alpha=0.35)
            subplot.plot(X[~mask, 0], X[~mask, 1], "oC1", mew=0, alpha=0.35)

    # Plot the results
    for i, m in enumerate(M):
        # Collate results
        result, axis = results[i], axes[:, i]

        P_test_svgp_pg, ind_points_svgp_pg = result[0]
        P_test_pgpr, ind_points_pgpr = result[1]

        # Plot inducing points locations
        if m != len(Y):  # don't plot if we use all points
            axis[0].scatter(ind_points_svgp_pg[:, 0], ind_points_svgp_pg[:, 1], c="k", s=5, zorder=1000)
            axis[1].scatter(ind_points_pgpr[:, 0], ind_points_pgpr[:, 1], c="k", s=5, zorder=1000)

        # Plot SVGP PG decision boundary
        _ = axis[0].contour(
            *X_grid,
            P_test_svgp_pg.reshape(40, 40),
            [0.5],  # p=0.5 decision boundary
            colors="k",
            linewidths=1.5,
            zorder=100
        )

        # Plot PGPR decision boundary
        _ = axis[1].contour(
            *X_grid,
            P_test_pgpr.reshape(40, 40),
            [0.5],  # p=0.5 decision boundary
            colors="k",
            linewidths=1.5,
            zorder=100
        )

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Load data
    X = np.loadtxt("../../data/toy/banana_X.csv", delimiter=",")
    Y = np.loadtxt("../../data/toy/banana_Y.csv").reshape(-1, 1)
    mask = Y[:, 0] == 1
    # Test data
    X_range = np.linspace(-3, 3, 40)
    X_grid = np.meshgrid(X_range, X_range)
    X_test = np.asarray(X_grid).transpose([1, 2, 0]).reshape(-1, 2)

    # Test different numbers of inducing points
    M = [4, 8, 16, 32, 64, len(Y)]

    results = []
    for m in M:
        print("Beginning training for", m, "inducing points...")
        results.append(run_experiment(m))
        print("Completed training for", m, "inducing points.")

    plot_results(M, results)
