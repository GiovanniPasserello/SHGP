import gpflow
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import sigmoid

from shgp.models.wenzel import Wenzel


# Polya-Gamma uses logit link / sigmoid
def invlink(f):
    return gpflow.likelihoods.Bernoulli(invlink=sigmoid).invlink(f).numpy()


def classification_demo():
    # Define model
    m = Wenzel(
        data=(X, Y),
        kernel=gpflow.kernels.SquaredExponential(),
        inducing_variable=X[::50].copy()
    )
    gpflow.set_trainable(m.inducing_variable, False)

    # Optimize model
    opt = gpflow.optimizers.Scipy()
    for _ in range(10):
        opt.minimize(m.training_loss, variables=m.trainable_variables, options=dict(maxiter=250))
        m.optimise_ci()
        print(m.elbo())

    # Take predictions
    X_test_mean, _ = m.predict_f(X_test)
    P_test = invlink(X_test_mean)

    print(m.elbo())

    # Plot data
    plt.plot(X[mask, 0], X[mask, 1], "oC0", mew=0, alpha=0.5)
    plt.plot(X[~mask, 0], X[~mask, 1], "oC1", mew=0, alpha=0.5)

    # Plot decision boundary
    _ = plt.contour(
        *X_grid,
        P_test.reshape(NUM_TEST_INDICES, NUM_TEST_INDICES),
        [0.5],  # p=0.5 decision boundary
        colors="k",
        linewidths=1.8,
        zorder=100,
    )

    plt.show()


if __name__ == '__main__':
    # Load data
    X = np.loadtxt("data/banana_X.csv", delimiter=",")
    Y = np.loadtxt("data/banana_Y.csv").reshape(-1, 1)
    mask = Y[:, 0] == 1
    # Test data
    NUM_TEST_INDICES = 40
    X_range = np.linspace(-3, 3, NUM_TEST_INDICES)
    X_grid = np.meshgrid(X_range, X_range)
    X_test = np.asarray(X_grid).transpose([1, 2, 0]).reshape(-1, 2)
    # Plot params
    plt.rcParams["figure.figsize"] = (7, 7)

    classification_demo()
