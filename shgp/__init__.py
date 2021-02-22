import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import gpflow
from gpflow.ci_utils import ci_niter

from shgp.models.hgpr import HGPR


np.random.seed(42)  # for reproducibility


def generate_data(N=120):
    X = np.random.rand(N)[:, None] * 10 - 5  # Inputs, shape N x 1
    X = np.sort(X.flatten()).reshape(N, 1)
    F = 2.5 * np.sin(6 * X) + np.cos(3 * X)  # Mean function values
    NoiseVar = np.abs(0.25 * X) + 0.15  # Noise variances
    Y = F + np.random.randn(N, 1) * np.sqrt(NoiseVar)  # Noisy data
    return X, Y, NoiseVar


def train_model(model, n_iter=100):
    adam = tf.optimizers.Adam()

    for i in range(ci_niter(n_iter)):
        adam.minimize(model.training_loss, model.trainable_variables)

    return model


if __name__ == "__main__":
    X, Y, NoiseVar = generate_data()
    xx = np.linspace(-5, 5, 200)[:, None]

    # Shared metadata
    inducing_locs = X

    kernel1 = gpflow.kernels.SquaredExponential(lengthscales=0.2)
    inducing_vars1 = gpflow.inducing_variables.InducingPoints(inducing_locs)
    model1 = gpflow.models.SGPR((X, Y), kernel=kernel1, inducing_variable=inducing_vars1)
    model1 = train_model(model1, 10)
    print("model1 trained")

    kernel2 = gpflow.kernels.SquaredExponential(lengthscales=0.2)
    inducing_vars2 = gpflow.inducing_variables.InducingPoints(inducing_locs)
    model2 = HGPR((X, Y), kernel=kernel2, inducing_variable=inducing_vars2)
    model2 = train_model(model2, 10)
    print("model2 trained")

    mu1, var1 = model1.predict_f(xx)
    var1_y = var1 + model1.likelihood.variance
    mu2, var2 = model2.predict_f(xx)
    _, var2_y = model2.predict_y(xx)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6))
    ax1.plot(xx, mu1, "C0", label='mean')
    ax1.plot(xx, mu1 + 2 * np.sqrt(var1), "C0", lw=0.5, label='f_var')
    ax1.plot(xx, mu1 - 2 * np.sqrt(var1), "C0", lw=0.5)
    ax1.plot(xx, mu1 + 2 * np.sqrt(var1_y), "r", lw=0.5, label='y_var')
    ax1.plot(xx, mu1 - 2 * np.sqrt(var1_y), "r", lw=0.5)
    ax2.plot(xx, mu2, "C0", label='mean')
    ax2.plot(xx, mu2 + 2 * np.sqrt(var2), "C0", lw=0.5, label='f_var')
    ax2.plot(xx, mu2 - 2 * np.sqrt(var2), "C0", lw=0.5)
    ax2.plot(xx, mu2 + 2 * np.sqrt(var2_y), "r", lw=0.5, label='y_var')
    ax2.plot(xx, mu2 - 2 * np.sqrt(var2_y), "r", lw=0.5)

    ax1.errorbar(
        X.squeeze(),
        Y.squeeze(),
        yerr=2 * (np.sqrt(NoiseVar)).squeeze(),
        marker="x",
        lw=0,
        elinewidth=1.0,
        color="C1",
    )
    ax2.errorbar(
        X.squeeze(),
        Y.squeeze(),
        yerr=2 * (np.sqrt(NoiseVar)).squeeze(),
        marker="x",
        lw=0,
        elinewidth=1.0,
        color="C1",
    )

    fig.tight_layout(pad=4)
    ax1.set_title('SGPR')
    ax2.set_title('HGPR')
    ax1.set_xlim(-5, 5)
    ax2.set_xlim(-5, 5)
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')

    print(model2.elbo())

    plt.show()
