import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


np.random.seed(42)
tf.random.set_seed(69)


"""
Used to generate a toy example image for my thesis.
"""


def generate_data():
    X = [-3.10, -3.05, -2.75, -2.60, -0.05, -0.03, 0.05, 0.06, 0.13, 2.12, 2.13, 2.14, 2.35, 2.85]
    F = [0.1, 0.05, 0.13, 0.15, -0.20, -0.21, -0.20, -0.20, -0.19, 0.07, 0.08, 0.09, 0.15, 0.19]
    return np.array([X]).T / 3.5, np.array([F]).T * 3


if __name__ == "__main__":
    X, Y = generate_data()
    X_test = np.linspace(-1, 1, 100)[:, None]

    ticks = [-1, 0, 1]

    plt.rcParams["figure.figsize"] = (16, 4)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)
    ax2.set_xticks(ticks)
    ax2.set_yticks(ticks)

    # Model definitions
    kernel1 = gpflow.kernels.SquaredExponential(lengthscales=0.5)
    model1 = gpflow.models.GPR((X, np.zeros_like(X)), kernel=kernel1)

    # Make predictions
    mu1, var1 = np.zeros_like(X_test), np.full_like(X_test, 0.3)
    samples1 = model1.predict_f_samples(X_test, 5)
    ax1.plot(X_test, mu1, "C0", linewidth=3)
    ax1.plot(X_test, samples1[:, :, 0].numpy().T, "g", alpha=0.8, linewidth=2)
    ax1.plot(X.squeeze(), Y.squeeze(), "kx", zorder=1000, markersize=10, linewidth=3)
    ax1.fill_between(
        X_test.flatten(),
        np.ravel(mu1 - np.sqrt(var1)),
        np.ravel(mu1 + np.sqrt(var1)),
        color="C0",
        alpha=0.35,
    )
    ax1.fill_between(
        X_test.flatten(),
        np.ravel(mu1 - 2 * np.sqrt(var1)),
        np.ravel(mu1 + 2 * np.sqrt(var1)),
        color="C0",
        alpha=0.25,
    )
    ax1.set_xlim(-1, 1)

    # Optimise
    kernel2 = gpflow.kernels.SquaredExponential()
    model2 = gpflow.models.GPR((X, Y), kernel=kernel2)
    gpflow.optimizers.Scipy().minimize(model2.training_loss, variables=model2.trainable_variables)
    print("sgpr trained")

    # Make predictions
    mu2, var2 = model2.predict_f(X_test)
    samples2 = model2.predict_f_samples(X_test, 5)
    ax2.plot(X_test, mu2, "C0", linewidth=3)
    ax2.plot(X_test, samples2[:, :, 0].numpy().T, "g", alpha=0.8, linewidth=2)
    ax2.plot(X.squeeze(), Y.squeeze(), "kx", zorder=1000, markersize=10, linewidth=3)
    ax2.fill_between(
        X_test.flatten(),
        np.ravel(mu2 - np.sqrt(var2)),
        np.ravel(mu2 + np.sqrt(var2)),
        color="C0",
        alpha=0.35,
    )
    ax2.fill_between(
        X_test.flatten(),
        np.ravel(mu2 - 2 * np.sqrt(var2)),
        np.ravel(mu2 + 2 * np.sqrt(var2)),
        color="C0",
        alpha=0.25,
    )
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(ax1.get_ylim())

    fig.tight_layout()
    plt.show()
