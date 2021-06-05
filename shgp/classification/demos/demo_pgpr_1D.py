import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from shgp.data.dataset import PlatformDataset
from shgp.utilities.general import invlink
from shgp.utilities.train_pgpr import train_pgpr

np.random.seed(42)
tf.random.set_seed(42)

"""
Demonstration of non-sparse PGPR on the 'platform' dataset.
We plot the datapoints and the predictive decision boundaries.
"""


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

    # Plot linked / 'squashed' predictions
    P_test = invlink(X_test_mean)
    plt.plot(X_test, P_test, "C1", lw=1)

    # Plot data classification
    X_train_mean, _ = m.predict_f(X)
    P_train = invlink(X_train_mean)
    correct = P_train.round() == Y
    plt.scatter(X[correct], Y[correct], c="g", s=40, marker='x', label='correct')
    plt.scatter(X[~correct], Y[~correct], c="r", s=40, marker='x', label='incorrect')

    # Meta
    plt.title("PGPR - Platform Dataset")
    plt.xlim((-2, 2))
    plt.ylim((-0.5, 1.5))

    # Display
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Load data
    X, Y = PlatformDataset().load_data()
    X_test = np.linspace(-2, 2, 200).reshape(-1, 1)

    # Plot params
    plt.rcParams["figure.figsize"] = (8, 4)

    classification_demo()
