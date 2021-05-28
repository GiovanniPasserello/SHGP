import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from shgp.data.dataset import BananaDataset
from shgp.data.metadata_reinit import ReinitMetaDataset
from shgp.inducing.initialisation_methods import uniform_subsample, reinitialise_PGPR, h_reinitialise_PGPR
from shgp.utilities.general import invlink
from shgp.utilities.train_pgpr import train_pgpr
from shgp.utilities.train_svgp import train_svgp

np.random.seed(0)
tf.random.set_seed(0)

"""
A comparison of SVGP vs PGPR with three different inducing point initialisation procedures. The inducing points 
of SVGP and the first PGPR model are uniformly subsampled and then optimised using gradient-based optimisation. 
The inducing points of the second PGPR model are selected via greedy variance reinitialisation, and the inducing
points of the final PGPR are selected via heteroscedastic greedy variance reinitialisation.

ELBO results for M = [4, 8, 16, 32, 64, 400]:

svgp_go  = [-210.9972, -156.3718, -112.6626, -106.2265, -105.9962, -105.9441]
pgpr_go  = [-215.7537, -153.3771, -127.5161, -119.9483, -119.7410, -119.7408]
pgpr_gv  = [-265.5385, -225.0653, -140.7713, -121.4397, -119.7441, -119.7408]
pgpr_hgv = [-268.0160, -259.0477, -144.6468, -121.3274, -119.7440, -119.7408]
"""


def run_experiment(M):
    inner_iters, opt_iters, ci_iters = 20, 1000, 20

    ################################################
    # PGPR with Gradient-Optimised Inducing Points #
    ################################################

    pgpr_go, pgpr_go_elbo = train_pgpr(
        X, Y,
        inner_iters, opt_iters, ci_iters,
        kernel_type=gpflow.kernels.SquaredExponential,
        M=M,
        init_method=uniform_subsample,
        optimise_Z=True
    )
    print("pgpr_go trained: ELBO = {}".format(pgpr_go_elbo))

    ################################################
    # SVGP with Gradient-Optimised Inducing Points #
    ################################################

    svgp_go, svgp_go_elbo = train_svgp(
        X, Y,
        M=M,
        train_iters=opt_iters,
        init_method=uniform_subsample,
        optimise_Z=True
    )
    print("svgp_go trained: ELBO = {}".format(svgp_go_elbo))

    #############################################
    # PGPR with Greedy Variance #
    #############################################

    pgpr_gv, pgpr_gv_elbo = train_pgpr(
        X, Y,
        inner_iters, opt_iters, ci_iters,
        kernel_type=gpflow.kernels.SquaredExponential,
        M=M,
        init_method=reinitialise_PGPR,
        reinit_metadata=ReinitMetaDataset()
    )
    print("pgpr_gv trained: ELBO = {}".format(pgpr_gv_elbo))

    #############################################
    # PGPR with Heteroscedastic Greedy Variance #
    #############################################

    pgpr_hgv, pgpr_hgv_elbo = train_pgpr(
        X, Y,
        inner_iters, opt_iters, ci_iters,
        kernel_type=gpflow.kernels.SquaredExponential,
        M=M,
        init_method=h_reinitialise_PGPR,
        reinit_metadata=ReinitMetaDataset()
    )
    print("pgpr_hgv trained: ELBO = {}".format(pgpr_hgv_elbo))

    ##############
    # Prediction #
    ##############

    # Take predictions from SVGP GO
    X_test_mean_svgp_go, _ = svgp_go.predict_f(X_test)
    P_test_svgp_go = invlink(X_test_mean_svgp_go)

    # Take predictions from PGPR GO
    X_test_mean_pgpr_go, _ = pgpr_go.predict_f(X_test)
    P_test_pgpr_go = invlink(X_test_mean_pgpr_go)

    # Take predictions from PGPR GV
    X_test_mean_pgpr_gv, _ = pgpr_gv.predict_f(X_test)
    P_test_pgpr_gv = invlink(X_test_mean_pgpr_gv)

    # Take predictions from PGPR HGV
    X_test_mean_pgpr_hgv, _ = pgpr_hgv.predict_f(X_test)
    P_test_pgpr_hgv = invlink(X_test_mean_pgpr_hgv)

    # Collate results
    results_svgp_go = (P_test_svgp_go, svgp_go.inducing_variable.Z.variables[0])
    results_pgpr_go = (P_test_pgpr_go, pgpr_go.inducing_variable.Z.variables[0])
    results_pgpr_gv = (P_test_pgpr_gv, pgpr_gv.inducing_variable.Z.variables[0])
    results_pgpr_hgv = (P_test_pgpr_hgv, pgpr_hgv.inducing_variable.Z.variables[0])

    return results_svgp_go, results_pgpr_go, results_pgpr_gv, results_pgpr_hgv


def plot_results(M, results):
    plt.rcParams["figure.figsize"] = (8, 5.333)
    fig, axes = plt.subplots(4, len(M), squeeze=False)

    # Setup labels
    models = ['SVGP GO', 'PGPR GO', 'PGPR GV', 'PGPR HGV']
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

        P_test_svgp_go, ind_points_svgp_go = result[0]
        P_test_pgpr_go, ind_points_pgpr_go = result[1]
        P_test_pgpr_gv, ind_points_pgpr_gv = result[2]
        P_test_pgpr_hgv, ind_points_pgpr_hgv = result[3]

        # Plot inducing points locations
        if m != len(Y):  # don't plot if we use all points
            axis[0].scatter(ind_points_svgp_go[:, 0], ind_points_svgp_go[:, 1], c="k", s=5, zorder=1000)
            axis[1].scatter(ind_points_pgpr_go[:, 0], ind_points_pgpr_go[:, 1], c="k", s=5, zorder=1000)
            axis[2].scatter(ind_points_pgpr_gv[:, 0], ind_points_pgpr_gv[:, 1], c="k", s=5, zorder=1000)
            axis[3].scatter(ind_points_pgpr_hgv[:, 0], ind_points_pgpr_hgv[:, 1], c="k", s=5, zorder=1000)

        # Plot SVGP GO decision boundary
        _ = axis[0].contour(
            *X_grid,
            P_test_svgp_go.reshape(NUM_TEST_INDICES, NUM_TEST_INDICES),
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

        # Plot PGPR GV decision boundary
        _ = axis[2].contour(
            *X_grid,
            P_test_pgpr_gv.reshape(NUM_TEST_INDICES, NUM_TEST_INDICES),
            [0.5],  # p=0.5 decision boundary
            colors="k",
            linewidths=1.5,
            zorder=100
        )

        # Plot PGPR HGV decision boundary
        _ = axis[3].contour(
            *X_grid,
            P_test_pgpr_hgv.reshape(NUM_TEST_INDICES, NUM_TEST_INDICES),
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
