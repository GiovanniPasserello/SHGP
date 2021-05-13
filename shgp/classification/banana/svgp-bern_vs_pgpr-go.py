import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import sigmoid

from shgp.data.dataset import BananaDataset
from shgp.inducing.initialisation_methods import uniform_subsample
from shgp.models.pgpr import PGPR
from shgp.utilities.general import invlink

np.random.seed(0)
tf.random.set_seed(0)

"""
A comparison of SVGP with a Bernoulli likelihood and PGPR. The inducing points of both models 
are uniformly subsampled and then optimised using gradient-based optimisation.
The results for M=4 are highly dependent on the random seed used - this result isn't representative.

ELBO results for M = [4, 8, 16, 32, 64, 400]:

pgpr_go   = [-215.7537, -153.3771, -127.5161, -119.9482, -119.7410, -119.7408]
svgp_bern = [-210.9973, -138.5364, -112.1889, -106.1611, -105.9458, -105.9441]
"""


def run_experiment(M):
    initial_inducing_inputs, _ = uniform_subsample(X, M)

    #############################
    # SVGP Bernoulli likelihood #
    #############################

    svgp_bern = gpflow.models.SVGP(
        kernel=gpflow.kernels.SquaredExponential(),
        likelihood=gpflow.likelihoods.Bernoulli(invlink=sigmoid),
        inducing_variable=initial_inducing_inputs.copy()
    )
    if m == len(Y):
        gpflow.set_trainable(svgp_bern.inducing_variable, False)
    gpflow.optimizers.Scipy().minimize(
        svgp_bern.training_loss_closure((X, Y)),
        variables=svgp_bern.trainable_variables
    )
    print("svgp_bern trained: ELBO = {}".format(svgp_bern.elbo((X, Y))))

    ################################################
    # PGPR with Gradient-Optimised Inducing Points #
    ################################################

    pgpr_go = PGPR(
        data=(X, Y),
        kernel=gpflow.kernels.SquaredExponential(),
        inducing_variable=initial_inducing_inputs.copy()
    )
    if m == len(Y):
        gpflow.set_trainable(pgpr_go.inducing_variable, False)
    opt = gpflow.optimizers.Scipy()
    for _ in range(20):
        opt.minimize(pgpr_go.training_loss, variables=pgpr_go.trainable_variables)
        pgpr_go.optimise_ci(num_iters=20)
    print("pgpr_go trained: ELBO = {}".format(pgpr_go.elbo()))

    ##############
    # Prediction #
    ##############

    # Take predictions from SVGP Bernoulli
    X_test_mean_svgp_bern, _ = svgp_bern.predict_f(X_test)
    P_test_svgp_bern = invlink(X_test_mean_svgp_bern)

    # Take predictions from PGPR GO
    X_test_mean_pgpr_go, _ = pgpr_go.predict_f(X_test)
    P_test_pgpr_go = invlink(X_test_mean_pgpr_go)

    # Collate results
    results_svgp_bern = (P_test_svgp_bern, svgp_bern.inducing_variable.Z.variables[0])
    results_pgpr_go = (P_test_pgpr_go, pgpr_go.inducing_variable.Z.variables[0])

    return results_svgp_bern, results_pgpr_go


def plot_results(M, results):
    plt.rcParams["figure.figsize"] = (8, 2.666)
    fig, axes = plt.subplots(2, len(M), squeeze=False)

    # Setup labels
    models = ['SVGP BERN', 'PGPR GO']
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

        P_test_svgp_bern, ind_points_svgp_bern = result[0]
        P_test_pgpr_go, ind_points_pgpr_go = result[1]

        # Plot inducing points locations
        if m != len(Y):  # don't plot if we use all points
            axis[0].scatter(ind_points_svgp_bern[:, 0], ind_points_svgp_bern[:, 1], c="k", s=5, zorder=1000)
            axis[1].scatter(ind_points_pgpr_go[:, 0], ind_points_pgpr_go[:, 1], c="k", s=5, zorder=1000)

        # Plot SVGP Bernoulli decision boundary
        _ = axis[0].contour(
            *X_grid,
            P_test_svgp_bern.reshape(NUM_TEST_INDICES, NUM_TEST_INDICES),
            [0.5],  # p=0.5 decision boundary
            colors="k",
            linewidths=1.5,
            zorder=100
        )

        # Plot PGPR GO decision boundary
        _ = axis[1].contour(
            *X_grid,
            P_test_pgpr_go.reshape(NUM_TEST_INDICES, NUM_TEST_INDICES),
            [0.5],  # p=0.5 decision boundary
            colors="k",
            linewidths=1.5,
            zorder=100
        )

    fig.tight_layout()
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

    # Test different numbers of inducing points
    M = [4, 8, 16, 32, 64, len(Y)]

    results = []
    for m in M:
        print("Beginning training for", m, "inducing points...")
        results.append(run_experiment(m))
        print("Completed training for", m, "inducing points.")

    plot_results(M, results)
