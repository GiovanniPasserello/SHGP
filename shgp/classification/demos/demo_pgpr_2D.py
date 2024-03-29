import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from shgp.data.dataset import BananaDataset
from shgp.utilities.general import invlink
from shgp.utilities.train_pgpr import train_pgpr

np.random.seed(0)
tf.random.set_seed(0)

"""
Demonstration of non-sparse PGPR on the 'banana' dataset.
We plot the datapoints and the predictive decision boundaries on the left.
We plot the Polya-Gamma variance as contours on the right and see that the
variance is the lowest at the predictive boundaries.
"""


def classification_demo():
    # Train model
    m, elbo = train_pgpr(
        X, Y,
        10, 250, 10,
        kernel_type=gpflow.kernels.SquaredExponential
    )
    print(elbo)

    # Take predictions
    X_test_mean, X_test_var = m.predict_f(X_test)
    P_test = invlink(X_test_mean)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

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
    _ = ax2.contour(
        *X_grid,
        P_test.reshape(NUM_TEST_INDICES, NUM_TEST_INDICES),
        [0.5],  # p=0.5 decision boundary
        colors="k",
        linewidths=1.8,
        zorder=100,
    )

    # Plot contours of the PG variance (yellow is low variance).
    # This allows us to inspect how PG variance behaves around boundaries and
    # how this may influence inducing point selection methods.
    test_c_i = m.likelihood.compute_c_i(X_test_mean, X_test_var)
    test_theta = m.likelihood.compute_theta(test_c_i).numpy()
    polya_gamma_vars = np.reciprocal(test_theta)
    _ = ax2.contourf(
        *X_grid,
        polya_gamma_vars.reshape(NUM_TEST_INDICES, NUM_TEST_INDICES),
        zorder=-100,
        cmap=plt.cm.viridis_r
    )

    ax1.set_title('PGPR Classification Boundaries')
    ax2.set_title('PGPR Variance Contours')

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

    classification_demo()
