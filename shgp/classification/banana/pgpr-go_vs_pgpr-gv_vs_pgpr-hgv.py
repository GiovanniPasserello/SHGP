import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import sigmoid

from shgp.inducing.initialisation_methods import uniform_subsample, reinitialise_PGPR, h_reinitialise_PGPR
from shgp.models.pgpr import PGPR

np.random.seed(42)
tf.random.set_seed(42)


"""
A comparison of PGPR with three different inducing point initialisation procedures. The inducing points 
of the first model are uniformly subsampled and then optimised using gradient-based optimisation. 
The inducing points of the second PGPR model use greedy variance reinitialisation, and the inducing
points of the final PGPR use heteroscedastic greedy variance reinitialisation.

Key takeaways: For very small and fixed M, GV/HGV adds extra complexity and tunable parameters than does
not provide much benefit - in fact it can take longer than GO without efficient convergence checks in place. 
For larger scale problems however, the initialisations can be quite superior and significantly outperform in
terms of the ELBO - this is with both a lower M and with less compute. PG in general adds complexity / fragility
of additional hyperparameters but it also brings many benefits - with a robust empirical setup I believe that it 
could be a large step forward in the GP classification framework.

Gradient based optimisation is superior for small M, but HGV is better as M grows larger. For small M, GV appears
to outperform HGV. This is likely because the heteroscedastic 'theta' is based on the model's outputs - when the 
model is inaccurate (with small M), then an initialisation procedure based on this will not be desirable - the 
plain Nystrom difference is a more reliable metric. However as M grows and the model becomes more accurate, so 
does 'theta', and so the HGV initialisation is superior as we can accurately model the heteroscedasticity.

Another downside is that GO is susceptible to local minimums - its performance may vary between runs due to randomness.
These differences are less evident in this small-scale problem, but we will see the benefits of HGV in other 
comparisons on larger-scale problems. Note also that I have not standardised the features in this 2D example which would
be more important as we scale up the number of dimensions - this example has similar variances in X1 & X2.

ELBO results for M = [4, 8, 16, 32, 64, 400]:

pgpr_go  = [-226.0542, -154.0238, -128.1118, -120.5023, -120.2990, -120.2990]
pgpr_gv  = [-270.2287, -260.6265, -146.7758, -122.9001, -120.3034, -120.2990]
pgpr_hgv = [-271.8793, -261.2196, -142.4491, -122.4783, -120.3020, -120.2990]
"""


# TODO: Move to utils
# Polya-Gamma uses logit link / sigmoid
def invlink(f):
    return gpflow.likelihoods.Bernoulli(invlink=sigmoid).invlink(f).numpy()


def run_experiment(M):
    initial_inducing_inputs, _ = uniform_subsample(X, M)

    ################
    # Optimisation #
    ################

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

    #############################################
    # PGPR with Greedy Variance #
    #############################################
    pgpr_gv = PGPR(
        data=(X, Y),
        kernel=gpflow.kernels.SquaredExponential(),
        inducing_variable=initial_inducing_inputs.copy()
    )
    opt = gpflow.optimizers.Scipy()
    # If we use full dataset, don't use inducing point selection
    if m == len(Y):
        gpflow.set_trainable(pgpr_gv.inducing_variable, False)
        # Optimize model
        for _ in range(20):
            opt.minimize(pgpr_gv.training_loss, variables=pgpr_gv.trainable_variables)
            pgpr_gv.optimise_ci(num_iters=20)
    else:
        prev_elbo = pgpr_gv.elbo()
        iter_limit = 10
        while True:
            # Reinitialise inducing points
            reinitialise_PGPR(pgpr_gv, X, m)

            # Optimize model
            for _ in range(20):
                opt.minimize(pgpr_gv.training_loss, variables=pgpr_gv.trainable_variables)
                pgpr_gv.optimise_ci(num_iters=20)

            # Check convergence
            next_elbo = pgpr_gv.elbo()
            if np.abs(next_elbo - prev_elbo) <= 1e-3 or iter_limit == 0:
                if iter_limit == 0:
                    print("pgpr_gv did not converge: prev={}, next={}".format(prev_elbo, next_elbo))
                break
            prev_elbo = next_elbo
            iter_limit -= 1
    print("pgpr_gv trained: ELBO = {}".format(pgpr_gv.elbo()))

    #############################################
    # PGPR with Heteroscedastic Greedy Variance #
    #############################################
    pgpr_hgv = PGPR(
        data=(X, Y),
        kernel=gpflow.kernels.SquaredExponential(),
        inducing_variable=initial_inducing_inputs.copy()
    )
    opt = gpflow.optimizers.Scipy()
    # If we use full dataset, don't use inducing point selection
    if m == len(Y):
        gpflow.set_trainable(pgpr_hgv.inducing_variable, False)
        # Optimize model
        for _ in range(20):
            opt.minimize(pgpr_hgv.training_loss, variables=pgpr_hgv.trainable_variables)
            pgpr_hgv.optimise_ci(num_iters=20)
    else:
        prev_elbo = pgpr_hgv.elbo()
        iter_limit = 10
        while True:
            # Reinitialise inducing points
            h_reinitialise_PGPR(pgpr_hgv, X, m)

            # Optimize model
            for _ in range(20):
                opt.minimize(pgpr_hgv.training_loss, variables=pgpr_hgv.trainable_variables)
                pgpr_hgv.optimise_ci(num_iters=20)

            # Check convergence
            next_elbo = pgpr_hgv.elbo()
            if np.abs(next_elbo - prev_elbo) <= 1e-3 or iter_limit == 0:
                if iter_limit == 0:
                    print("pgpr_hgv did not converge: prev={}, next={}".format(prev_elbo, next_elbo))
                break
            prev_elbo = next_elbo
            iter_limit -= 1
    print("pgpr_hgv trained: ELBO = {}".format(pgpr_hgv.elbo()))

    ##############
    # Prediction #
    ##############

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
    results_pgpr_go = (P_test_pgpr_go, pgpr_go.inducing_variable.Z.variables[0])
    results_pgpr_gv = (P_test_pgpr_gv, pgpr_gv.inducing_variable.Z.variables[0])
    results_pgpr_hgv = (P_test_pgpr_hgv, pgpr_hgv.inducing_variable.Z.variables[0])

    return results_pgpr_go, results_pgpr_gv, results_pgpr_hgv


def plot_results(M, results):
    plt.rcParams["figure.figsize"] = (8, 4)
    fig, axes = plt.subplots(3, len(M), squeeze=False)

    # Setup labels
    models = ['PGPR GO', 'PGPR GV', 'PGPR HGV']
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

        P_test_pgpr_go, ind_points_pgpr_go = result[0]
        P_test_pgpr_gv, ind_points_pgpr_gv = result[1]
        P_test_pgpr_hgv, ind_points_pgpr_hgv = result[2]

        # Plot inducing points locations
        if m != len(Y):  # don't plot if we use all points
            axis[0].scatter(ind_points_pgpr_go[:, 0], ind_points_pgpr_go[:, 1], c="k", s=5, zorder=1000)
            axis[1].scatter(ind_points_pgpr_gv[:, 0], ind_points_pgpr_gv[:, 1], c="k", s=5, zorder=1000)
            axis[2].scatter(ind_points_pgpr_hgv[:, 0], ind_points_pgpr_hgv[:, 1], c="k", s=5, zorder=1000)

        # Plot PGPR GO decision boundary
        _ = axis[0].contour(
            *X_grid,
            P_test_pgpr_go.reshape(40, 40),
            [0.5],  # p=0.5 decision boundary
            colors="k",
            linewidths=1.5,
            zorder=100
        )

        # Plot PGPR GV decision boundary
        _ = axis[1].contour(
            *X_grid,
            P_test_pgpr_gv.reshape(40, 40),
            [0.5],  # p=0.5 decision boundary
            colors="k",
            linewidths=1.5,
            zorder=100
        )

        # Plot PGPR HGV decision boundary
        _ = axis[2].contour(
            *X_grid,
            P_test_pgpr_hgv.reshape(40, 40),
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
