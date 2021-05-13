import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from shgp.utilities.general import invlink
from shgp.utilities.train_pgpr import train_pgpr

np.random.seed(42)
tf.random.set_seed(42)


def classification_demo():
    # Train model
    m, elbo = train_pgpr(
        X, Y,
        10, 1000, 10,
        kernel_type=gpflow.kernels.Matern52
    )
    print(elbo)

    # Take predictions
    X_test_mean, X_test_var = m.predict_f(X_test)

    # Plot mean prediction
    plt.plot(X_test, X_test_mean, "C0", lw=1)

    # To inspect Polya-Gamma variance at boundaries
    # from matplotlib import cm
    # test_c_i = m.likelihood.compute_c_i(X_test_mean, X_test_var)
    # test_theta = m.likelihood.compute_theta(test_c_i).numpy()
    # polya_gamma_vars = np.reciprocal(test_theta).flatten()
    # color_map = cm.hot(polya_gamma_vars / polya_gamma_vars.max())
    # # Plot linked / 'squashed' predictions
    # P_test = invlink(X_test_mean)
    # plt.scatter(X_test, P_test, c=color_map)

    # Plot linked / 'squashed' predictions
    P_test = invlink(X_test_mean)
    plt.plot(X_test, P_test, "C1", lw=1)

    # Plot data classification
    X_train_mean, _ = m.predict_f(X)
    P_train = invlink(X_train_mean)
    correct = P_train.round() == Y
    plt.scatter(X[correct], Y[correct], c="g", s=40, marker='x', label='correct')
    plt.scatter(X[~correct], Y[~correct], c="r", s=40, marker='x', label='incorrect')

    plt.title("PGPR 1D Toy Dataset")
    plt.ylim((-0.5, 1.5))

    # Plot
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Load data
    X = np.genfromtxt("../../data/datasets/toy/classif_1D_X.csv").reshape(-1, 1)
    Y = np.genfromtxt("../../data/datasets/toy/classif_1D_Y.csv").reshape(-1, 1)
    X_test = np.linspace(0, 6, 200).reshape(-1, 1)
    # Plot params
    plt.rcParams["figure.figsize"] = (8, 4)

    classification_demo()
