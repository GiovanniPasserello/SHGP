import gpflow
import matplotlib.pyplot as plt
import numpy as np

from shgp.models.pgpr import PGPR
from shgp.utilities.general import invlink


def classification_demo():
    # Define model
    m = PGPR(
        data=(X, Y),
        kernel=gpflow.kernels.SquaredExponential(),
        inducing_variable=X.copy()
    )
    gpflow.set_trainable(m.inducing_variable, False)

    # Optimize model
    opt = gpflow.optimizers.Scipy()
    for _ in range(10):
        opt.minimize(m.training_loss, variables=m.trainable_variables, options=dict(maxiter=250))
        m.optimise_ci()

    # Take predictions
    X_test_mean, X_test_var = m.predict_f(X_test)
    P_test = invlink(X_test_mean)

    print(m.elbo())

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

    # Plot contours of the PG variance.
    # This allows us to inspect how PG variance behaves around boundaries and
    # how this may influence inducing point selection methods.
    test_c_i = m.likelihood.compute_c_i(X_test_mean, X_test_var)
    test_theta = m.likelihood.compute_theta(test_c_i).numpy()
    polya_gamma_vars = np.reciprocal(test_theta)
    cf = ax2.contourf(
        *X_grid,
        polya_gamma_vars.reshape(NUM_TEST_INDICES, NUM_TEST_INDICES),
        zorder=-100,
        cmap=plt.cm.viridis_r
    )

    ax1.set_title('PGPR Classification Boundaries')
    ax2.set_title('Polya-Gamma Variance Contours')

    # Optional sidebar (yellow is low variance)
    # plt.colorbar(cf)
    plt.show()


if __name__ == '__main__':
    # Load data
    X = np.loadtxt("../../data/toy/banana_X.csv", delimiter=",")
    Y = np.loadtxt("../../data/toy/banana_Y.csv").reshape(-1, 1)
    mask = Y[:, 0] == 1
    # Test data
    NUM_TEST_INDICES = 40
    X_range = np.linspace(-3, 3, NUM_TEST_INDICES)
    X_grid = np.meshgrid(X_range, X_range)
    X_test = np.asarray(X_grid).transpose([1, 2, 0]).reshape(-1, 2)

    classification_demo()
