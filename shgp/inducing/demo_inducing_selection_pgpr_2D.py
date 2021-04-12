import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from gpflow.models.util import inducingpoint_wrapper
from tensorflow import sigmoid

from shgp.inducing.greedy_variance import h_greedy_variance
from shgp.models.pgpr import PGPR


# Polya-Gamma uses logit link / sigmoid
def invlink(f):
    return gpflow.likelihoods.Bernoulli(invlink=sigmoid).invlink(f).numpy()


def inducing_demo():
    num_inducing = 30
    num_iters = 10

    # Naive random selection and optimisations
    kernel1 = gpflow.kernels.SquaredExponential()
    inducing_idx1 = np.random.choice(np.arange(X.shape[0]), size=num_inducing, replace=False)
    inducing_vars1 = gpflow.inducing_variables.InducingPoints(X[inducing_idx1])
    model1 = PGPR((X, Y), kernel=kernel1, inducing_variable=inducing_vars1)
    # Optimize model
    opt = gpflow.optimizers.Scipy()
    for _ in range(num_iters):
        opt.minimize(model1.training_loss, variables=model1.trainable_variables)
        model1.optimise_ci()
    elbo1 = model1.elbo()

    # Greedy variance selection
    threshold = 1e-6
    kernel2 = gpflow.kernels.SquaredExponential()
    model2 = PGPR((X, Y), kernel=kernel2)
    theta_inv = tf.math.reciprocal(model2.likelihood.compute_theta())
    inducing_locs2, inducing_idx2 = h_greedy_variance(X, theta_inv, num_inducing, kernel2, threshold)
    inducing_vars2 = gpflow.inducing_variables.InducingPoints(inducing_locs2)
    model2.inducing_variable = inducingpoint_wrapper(inducing_vars2)
    gpflow.set_trainable(model2.inducing_variable, False)
    prev_elbo = model2.elbo()
    # TODO: Better guarantees for convergence
    iter_limit = 10  # to avoid infinite loops
    while True:
        # Optimize model
        opt = gpflow.optimizers.Scipy()
        for _ in range(num_iters):
            opt.minimize(model2.training_loss, variables=model2.trainable_variables)
            model2.optimise_ci()

        next_elbo = model2.elbo()
        print("Previous ELBO: {}, Next ELBO: {}".format(prev_elbo, next_elbo))
        if np.abs(next_elbo - prev_elbo) <= 1e-6 or iter_limit == 0:
            break

        theta_inv = tf.math.reciprocal(model2.likelihood.compute_theta())
        inducing_locs2, inducing_idx2 = h_greedy_variance(X, theta_inv, num_inducing, kernel2, threshold)
        inducing_vars2 = gpflow.inducing_variables.InducingPoints(inducing_locs2)
        model2.inducing_variable = inducingpoint_wrapper(inducing_vars2)
        gpflow.set_trainable(model2.inducing_variable, False)

        prev_elbo = next_elbo
        iter_limit -= 1

    print("Final number of inducing points:", model2.inducing_variable.num_inducing)

    elbo2 = model2.elbo()

    fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 12))

    ###########
    # Model 1 #
    ###########

    X_test_mean, _ = model1.predict_f(X_test)
    P_test = invlink(X_test_mean)

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

    ###########
    # Model 2 #
    ###########

    X_test_mean, _ = model2.predict_f(X_test)
    P_test = invlink(X_test_mean)

    # Plot data
    ax2.plot(X[mask, 0], X[mask, 1], "oC0", mew=0, alpha=0.5)
    ax2.plot(X[~mask, 0], X[~mask, 1], "oC1", mew=0, alpha=0.5)

    # Plot decision boundary
    _ = ax2.contour(
        *X_grid,
        P_test.reshape(NUM_TEST_INDICES, NUM_TEST_INDICES),
        [0.5],  # p=0.5 decision boundary
        colors="k",
        linewidths=1.8,
        zorder=100,
    )

    ############
    # Inducing #
    ############

    inducing_points = model1.inducing_variable.Z.variables[0]
    ax1.scatter(inducing_points[:, 0], inducing_points[:, 1], c="b", label='ind point', zorder=1000)
    ax2.scatter(X[inducing_idx2, 0].squeeze(), X[inducing_idx2, 1].squeeze(), c="b", label='ind point', zorder=1000)

    # Inspect average noise of inducing and non-inducing points
    print(model2.likelihood.compute_theta().numpy().flatten()[inducing_idx2].mean())
    print(model2.likelihood.compute_theta().numpy().flatten()[np.where([a not in inducing_idx2 for a in np.arange(50)])].mean())

    fig.tight_layout(pad=4)
    ax1.set_title('Optimized Naive Selection')
    ax2.set_title('Polya-Gamma Greedy Variance')
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')

    print(elbo1, elbo2)

    plt.show()


if __name__ == "__main__":
    # Load data
    X = np.loadtxt("../classification/data/banana_X.csv", delimiter=",")
    Y = np.loadtxt("../classification/data/banana_Y.csv").reshape(-1, 1)
    mask = Y[:, 0] == 1
    # Test data
    NUM_TEST_INDICES = 40
    X_range = np.linspace(-3, 3, NUM_TEST_INDICES)
    X_grid = np.meshgrid(X_range, X_range)
    X_test = np.asarray(X_grid).transpose([1, 2, 0]).reshape(-1, 2)

    inducing_demo()
