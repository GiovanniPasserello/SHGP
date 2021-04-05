import gpflow
import matplotlib.pyplot as plt
import numpy as np


def logit(x):
    return np.exp(x) / (1 + np.exp(x))


# Polya-Gamma uses logit link, but we can remove this to default to probit
def invlink(f):
    return gpflow.likelihoods.Bernoulli(invlink=logit).invlink(f)


def classification_demo():
    # Fit model
    m = gpflow.models.VGP(
        (X, Y), likelihood=gpflow.likelihoods.Bernoulli(), kernel=gpflow.kernels.Matern52()
    )
    opt = gpflow.optimizers.Scipy()
    opt.minimize(m.training_loss, variables=m.trainable_variables)

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
    p = invlink(X_test_mean)
    plt.plot(X_test, p, "C1", lw=1)

    # Plot data classification
    X_mean = invlink(m.predict_f(X)[0])
    correct = X_mean.round() == Y
    plt.scatter(X[correct], Y[correct], c="g", s=40, marker='x', label='correct')
    plt.scatter(X[~correct], Y[~correct], c="r", s=40, marker='x', label='incorrect')

    # plot
    plt.ylim((-0.5, 1.5))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Load data
    X = np.genfromtxt("data/classif_1D_X.csv").reshape(-1, 1)
    Y = np.genfromtxt("data/classif_1D_Y.csv").reshape(-1, 1)
    X_test = np.linspace(0, 6, 200).reshape(-1, 1)
    # Plot params
    plt.rcParams["figure.figsize"] = (8, 4)

    classification_demo()
