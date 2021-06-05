import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from shgp.data.dataset import BananaDataset
from shgp.data.metadata_reinit import ReinitMetaDataset
from shgp.inducing.initialisation_methods import h_reinitialise_PGPR, k_means
from shgp.utilities.general import invlink
from shgp.utilities.train_pgpr import train_pgpr

np.random.seed(0)
tf.random.set_seed(0)

"""
A visual demonstration comparing optimised inducing points vs. HGV for PGPR on the 'banana' dataset.
We plot the chosen inducing points as dark blue dots, and see that HGV prefers to select points 
closer to the predictive boundaries.
"""


def inducing_demo():
    num_inducing, inner_iters, opt_iters, ci_iters = 30, 10, 500, 10

    # Uniform subsampling with gradient-based optimisation
    model1, elbo1 = train_pgpr(
        X, Y,
        inner_iters, opt_iters, ci_iters,
        M=num_inducing,
        init_method=k_means,
        optimise_Z=True
    )

    # Heteroscedastic greedy variance selection
    model2, _ = train_pgpr(
        X, Y,
        inner_iters, opt_iters, ci_iters,
        M=num_inducing,
        init_method=h_reinitialise_PGPR,
        reinit_metadata=ReinitMetaDataset()
    )
    elbo2 = model2.elbo()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ###########
    # Model 1 #
    ###########

    X_test_mean, _ = model1.predict_f(X_test)
    P_test = invlink(X_test_mean)

    # Plot data
    ax1.plot(X[mask, 0], X[mask, 1], "oC0", mew=0, alpha=0.5)
    ax1.plot(X[~mask, 0], X[~mask, 1], "oC1", mew=0, alpha=0.5)

    # Plot decision boundary
    _ = ax1.contour(
        *X_grid,
        P_test.reshape(NUM_TEST_INDICES, NUM_TEST_INDICES),
        [0.5],  # p=0.5 decision boundary
        colors="k",
        linewidths=1.8,
        zorder=100,
    )

    ###########
    # Model 2 #
    ###########

    X_test_mean, _ = model2.predict_f(X_test)
    P_test = invlink(X_test_mean)

    # Plot data
    ax2.plot(X[mask, 0], X[mask, 1], "oC0", mew=0, alpha=0.5)
    ax2.plot(X[~mask, 0], X[~mask, 1], "oC1", mew=0, alpha=0.5)

    # Plot decision boundary
    _ = ax2.contour(
        *X_grid,
        P_test.reshape(NUM_TEST_INDICES, NUM_TEST_INDICES),
        [0.5],  # p=0.5 decision boundary
        colors="k",
        linewidths=1.8,
        zorder=100,
    )

    ############
    # Inducing #
    ############

    inducing_inputs1 = model1.inducing_variable.Z.variables[0]
    inducing_inputs2 = model2.inducing_variable.Z.variables[0]
    ax1.scatter(inducing_inputs1[:, 0], inducing_inputs1[:, 1], c="b", label='ind point', zorder=1000)
    ax2.scatter(inducing_inputs2[:, 0], inducing_inputs2[:, 1], c="b", label='ind point', zorder=1000)

    fig.tight_layout(pad=4)
    ax1.set_title('Optimized Naive Selection')
    ax2.set_title('Polya-Gamma Greedy Variance')
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')

    print(elbo1, elbo2)

    plt.show()


if __name__ == "__main__":
    # Load data
    X, Y = BananaDataset().load_data()
    mask = Y[:, 0] == 1

    # Test data
    NUM_TEST_INDICES = 100
    X_range = np.linspace(-3, 3, NUM_TEST_INDICES)
    X_grid = np.meshgrid(X_range, X_range)
    X_test = np.asarray(X_grid).transpose([1, 2, 0]).reshape(-1, 2)

    inducing_demo()
