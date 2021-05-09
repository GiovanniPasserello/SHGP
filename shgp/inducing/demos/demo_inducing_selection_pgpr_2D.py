import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from shgp.inducing.initialisation_methods import h_reinitialise_PGPR
from shgp.robustness.contrained_kernels import ConstrainedSigmoidSEKernel
from shgp.models.pgpr import PGPR
from shgp.utilities.general import invlink

np.random.seed(0)
tf.random.set_seed(0)


def inducing_demo():
    num_inducing = 30
    num_iters = 10

    # A comparison of greedy_variance vs h_greedy_variance
    # greedy is jumpy (plot ELBO over epoch to show stability), h_greedy is stable
    # greedy spreads mass, h_greedy places at boundaries (our hypothesis)
    # greedy performs better for very low number of inducing points
    # h_greedy performs better for larger number of inducing points
    # h_greedy can beat greedy with fewer points when using larger number of inducing points
    # The below results are found using threshold=1e-6 (except 200/65 and 200/55 which used 1e-1)
    # h_greedy=[(200/118,-120.2989),(200/55,-120.3338),(30,-123.8708),(20,-134.7702),(15,-156.7694),(10,-243.0055)]
    # greedy = [(200/129,-120.2989),(200/65,-120.3027),(30, -123.0022), (20,-136.1941), (15,-149.1684), (10, -220.4174)]
    # h_greedy vs gradient_optim: (118,-120.2989), (200,-120.2996) but gradient_optim is good for very few points

    # Naive random selection and optimisations
    kernel1 = ConstrainedSigmoidSEKernel()
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
    kernel2 = ConstrainedSigmoidSEKernel()
    model2 = PGPR((X, Y), kernel=kernel2)
    prev_elbo = model2.elbo()
    iter_limit = 10  # to avoid infinite loops
    while True:
        _, inducing_idx2 = h_reinitialise_PGPR(model2, X, num_inducing, threshold)

        # Optimize model
        opt = gpflow.optimizers.Scipy()
        for _ in range(num_iters):
            opt.minimize(model2.training_loss, variables=model2.trainable_variables)
            model2.optimise_ci()

        next_elbo = model2.elbo()
        print("Previous ELBO: {}, Next ELBO: {}".format(prev_elbo, next_elbo))
        if np.abs(next_elbo - prev_elbo) <= 1e-6 or iter_limit == 0:
            break

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

    fig.tight_layout(pad=4)
    ax1.set_title('Optimized Naive Selection')
    ax2.set_title('Polya-Gamma Greedy Variance')
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')

    print(elbo1, elbo2)

    plt.show()


if __name__ == "__main__":
    # Load data
    X = np.loadtxt("../../data/toy/banana_X.csv", delimiter=",")
    Y = np.loadtxt("../../data/toy/banana_Y.csv").reshape(-1, 1)
    mask = Y[:, 0] == 1
    # Test data
    NUM_TEST_INDICES = 40
    X_range = np.linspace(-3, 3, NUM_TEST_INDICES)
    X_grid = np.meshgrid(X_range, X_range)
    X_test = np.asarray(X_grid).transpose([1, 2, 0]).reshape(-1, 2)

    inducing_demo()
