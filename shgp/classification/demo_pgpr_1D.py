import gpflow
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import sigmoid

from shgp.models.pgpr import PGPR


# Polya-Gamma uses logit link / sigmoid
def invlink(f):
    return gpflow.likelihoods.Bernoulli(invlink=sigmoid).invlink(f).numpy()


def classification_demo():
    # Define model
    m = PGPR(
        data=(X, Y),
        kernel=gpflow.kernels.Matern52(),
        inducing_variable=X.copy()
    )
    gpflow.set_trainable(m.inducing_variable, False)

    # Optimize model
    opt = gpflow.optimizers.Scipy()
    for _ in range(10):
        opt.minimize(m.training_loss, variables=m.trainable_variables)
        m.optimise_ci()

    # Take predictions
    X_test_mean, X_test_var = m.predict_f(X_test)

    # Plot mean prediction
    plt.plot(X_test, X_test_mean, "C0", lw=1)

    # Plot uncertainty bars
    plt.fill_between(
        X_test.flatten(),
        np.ravel(X_test_mean + 2 * np.sqrt(X_test_var)),
        np.ravel(X_test_mean - 2 * np.sqrt(X_test_var)),
        alpha=0.3,
        color="C0",
    )

    # Plot linked / 'squashed' predictions
    P_test = invlink(X_test_mean)
    plt.plot(X_test, P_test, "C1", lw=1)

    # Plot data classification
    X_train_mean, _ = m.predict_f(X)
    P_train = invlink(X_train_mean)
    correct = P_train.round() == Y
    plt.scatter(X[correct], Y[correct], c="g", s=40, marker='x', label='correct')
    plt.scatter(X[~correct], Y[~correct], c="r", s=40, marker='x', label='incorrect')

    print(m.elbo())

    # Plot
    plt.ylim((-0.5, 1.5))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Load data
    X = np.genfromtxt("../data/toy/classif_1D_X.csv").reshape(-1, 1)
    Y = np.genfromtxt("../data/toy/classif_1D_Y.csv").reshape(-1, 1)
    X_test = np.linspace(0, 6, 200).reshape(-1, 1)
    # Plot params
    plt.rcParams["figure.figsize"] = (8, 4)

    classification_demo()
