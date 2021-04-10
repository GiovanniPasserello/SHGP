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


# TODO: Verify whether this is the correct way of selecting inducing points for PG
def inducing_demo():
    num_inducing = 50
    num_iters = 5

    # Naive random selection and optimisations
    kernel1 = gpflow.kernels.Matern52()
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
    threshold = 1e-1
    kernel2 = gpflow.kernels.Matern52()
    model2 = PGPR((X, Y), kernel=kernel2)
    theta_inv = tf.math.reciprocal(model2.likelihood.compute_theta())
    inducing_locs2, inducing_idx2 = h_greedy_variance(X, theta_inv, num_inducing, kernel2, threshold)
    inducing_vars2 = gpflow.inducing_variables.InducingPoints(inducing_locs2)
    model2.inducing_variable = inducingpoint_wrapper(inducing_vars2)
    gpflow.set_trainable(model2.inducing_variable, False)
    prev_elbo = model2.elbo()
    # iter_limit = 10  # to avoid infinite loops
    while True:
        # Optimize model
        opt = gpflow.optimizers.Scipy()
        for _ in range(num_iters):
            opt.minimize(model2.training_loss, variables=model2.trainable_variables)
            model2.optimise_ci()

        next_elbo = model2.elbo()
        print("Previous ELBO: {}, Next ELBO: {}".format(prev_elbo, next_elbo))
        if np.abs(next_elbo - prev_elbo) <= 1e-6:  # or iter_limit == 0:
            break

        theta_inv = tf.math.reciprocal(model2.likelihood.compute_theta())
        inducing_locs2, inducing_idx2 = h_greedy_variance(X, theta_inv, num_inducing, kernel2, threshold)
        inducing_vars2 = gpflow.inducing_variables.InducingPoints(inducing_locs2)
        model2.inducing_variable = inducingpoint_wrapper(inducing_vars2)
        gpflow.set_trainable(model2.inducing_variable, False)

        prev_elbo = next_elbo
        # iter_limit -= 1

    print("Final number of inducing points:", model2.inducing_variable.num_inducing)

    # Optionally optimize at the end
    #gpflow.set_trainable(model2.inducing_variable, True)
    #gpflow.optimizers.Scipy().minimize(model2.training_loss, variables=model2.trainable_variables)

    elbo2 = model2.elbo()

    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6))

    ###########
    # Model 1 #
    ###########

    # Take predictions
    X_test_mean, X_test_var = model1.predict_f(X_test)
    # Plot mean prediction
    ax1.plot(X_test, X_test_mean, "C0", lw=1)
    # Plot linked / 'squashed' predictions
    P_test = invlink(X_test_mean)
    ax1.plot(X_test, P_test, "C1", lw=1)
    # Plot data classification
    X_train_mean, _ = model1.predict_f(X)
    P_train = invlink(X_train_mean)
    correct = P_train.round() == Y
    ax1.scatter(X[correct], Y[correct], c="g", s=40, marker='x', label='correct')
    ax1.scatter(X[~correct], Y[~correct], c="r", s=40, marker='x', label='incorrect')

    ###########
    # Model 2 #
    ###########

    # Take predictions
    X_test_mean, X_test_var = model2.predict_f(X_test)
    # Plot mean prediction
    ax2.plot(X_test, X_test_mean, "C0", lw=1)
    # Plot linked / 'squashed' predictions
    P_test = invlink(X_test_mean)
    ax2.plot(X_test, P_test, "C1", lw=1)
    # Plot data classification
    X_train_mean, _ = model2.predict_f(X)
    P_train = invlink(X_train_mean)
    correct = P_train.round() == Y
    ax2.scatter(X[correct], Y[correct], c="g", s=40, marker='x', label='correct')
    ax2.scatter(X[~correct], Y[~correct], c="r", s=40, marker='x', label='incorrect')

    ############
    # Inducing #
    ############

    # This works as X and Y are sorted, if they aren't make sure to sort them
    ax1.scatter(X[inducing_idx1].squeeze(), Y[inducing_idx1].squeeze(), c="b", label='ind point', zorder=1000)
    ax2.scatter(X[inducing_idx2].squeeze(), Y[inducing_idx2].squeeze(), c="b", label='ind point', zorder=1000)

    fig.tight_layout(pad=4)
    ax1.set_title('Optimized Naive Selection')
    ax2.set_title('Polya-Gamma Greedy Variance')
    ax1.set_ylim((-0.5, 1.5))
    ax2.set_ylim((-0.5, 1.5))
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')

    print(elbo1, elbo2)

    plt.show()


if __name__ == "__main__":
    # Load data
    X = np.genfromtxt("../classification/data/classif_1D_X.csv").reshape(-1, 1)
    Y = np.genfromtxt("../classification/data/classif_1D_Y.csv").reshape(-1, 1)
    X_test = np.linspace(0, 6, 200).reshape(-1, 1)

    inducing_demo()
