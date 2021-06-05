import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(12)


"""
Generates a toy example image for my thesis, demonstrating what inducing points are.
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

    # Full model definitions
    gpr_kernel = gpflow.kernels.SquaredExponential()
    gpr_model = gpflow.models.GPR((X, Y), kernel=gpr_kernel)
    gpflow.optimizers.Scipy().minimize(gpr_model.training_loss, variables=gpr_model.trainable_variables)
    print("gpr trained")

    # Make predictions
    gpr_mu, gpr_var = gpr_model.predict_f(X_test)
    ax1.plot(X_test, gpr_mu, "C0", linewidth=3, zorder=101)
    ax1.plot(X.squeeze(), Y.squeeze(), "kx", zorder=1000, markersize=10, linewidth=3)
    ax1.fill_between(
        X_test.flatten(),
        np.ravel(gpr_mu - np.sqrt(gpr_var)),
        np.ravel(gpr_mu + np.sqrt(gpr_var)),
        color="C0",
        alpha=0.35,
    )
    ax1.fill_between(
        X_test.flatten(),
        np.ravel(gpr_mu - 2 * np.sqrt(gpr_var)),
        np.ravel(gpr_mu + 2 * np.sqrt(gpr_var)),
        color="C0",
        alpha=0.25,
    )
    ax1.set_xlim(-1, 1)

    # Sparse model definitions
    sgpr_kernel = gpflow.kernels.SquaredExponential()
    inducing_locs = [1, 3, 6, 11, -1]
    inducing_vars = gpflow.inducing_variables.InducingPoints(X[inducing_locs].copy())
    sgpr_model = gpflow.models.SGPR((X, Y), kernel=sgpr_kernel, inducing_variable=inducing_vars)
    gpflow.set_trainable(sgpr_model.inducing_variable, False)
    gpflow.optimizers.Scipy().minimize(sgpr_model.training_loss, variables=sgpr_model.trainable_variables)
    print("sgpr trained")

    # Make predictions
    sgpr_mu, sgpr_var = sgpr_model.predict_f(X_test)
    ax2.plot(X_test, sgpr_mu, "C0", linewidth=3, zorder=101)
    ax2.plot(X.squeeze(), Y.squeeze(), "kx", zorder=1000, markersize=10, linewidth=3)
    ax2.fill_between(
        X_test.flatten(),
        np.ravel(sgpr_mu - np.sqrt(sgpr_var)),
        np.ravel(sgpr_mu + np.sqrt(sgpr_var)),
        color="C0",
        alpha=0.35,
    )
    ax2.fill_between(
        X_test.flatten(),
        np.ravel(sgpr_mu - 2 * np.sqrt(sgpr_var)),
        np.ravel(sgpr_mu + 2 * np.sqrt(sgpr_var)),
        color="C0",
        alpha=0.25,
    )
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(ax1.get_ylim())

    # This works as X and Y are sorted, if they aren't make sure to sort them
    ax2.scatter(X[inducing_locs].squeeze(), Y[inducing_locs].squeeze(), c="r", s=150, zorder=1001)

    fig.tight_layout()
    plt.show()
