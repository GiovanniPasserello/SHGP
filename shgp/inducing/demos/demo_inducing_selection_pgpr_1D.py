import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from shgp.inducing.initialisation_methods import h_reinitialise_PGPR
from shgp.models.pgpr import PGPR
from shgp.utilities.general import invlink

np.random.seed(0)
tf.random.set_seed(0)


def inducing_demo():
    num_inducing = 15
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
    prev_elbo = model2.elbo()
    while True:
        _, inducing_idx2 = h_reinitialise_PGPR(model2, X, num_inducing, threshold)

        # Optimize model
        opt = gpflow.optimizers.Scipy()
        for _ in range(num_iters):
            opt.minimize(model2.training_loss, variables=model2.trainable_variables)
            model2.optimise_ci()

        next_elbo = model2.elbo()
        print("Previous ELBO: {}, Next ELBO: {}".format(prev_elbo, next_elbo))
        if np.abs(next_elbo - prev_elbo) <= 1e-6:
            break
        prev_elbo = next_elbo

    print("Final number of inducing points:", model2.inducing_variable.num_inducing)

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

    # Plot the inducing points on the mean line - only the x-coordinate really matters
    inducing_inputs1 = model1.inducing_variable.Z.variables[0]
    inducing_outputs1, _ = model1.predict_f(inducing_inputs1)
    p_inducing_outputs1 = invlink(inducing_outputs1)
    inducing_outputs2, _ = model1.predict_f(X[inducing_idx2])
    p_inducing_outputs2 = invlink(inducing_outputs2)
    ax1.scatter(inducing_inputs1, p_inducing_outputs1, c="b", label='ind point', zorder=1000)
    ax2.scatter(X[inducing_idx2].squeeze(), p_inducing_outputs2, c="b", label='ind point', zorder=1000)

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
    X = np.genfromtxt("../../data/toy/classif_1D_X.csv").reshape(-1, 1)
    Y = np.genfromtxt("../../data/toy/classif_1D_Y.csv").reshape(-1, 1)
    X_test = np.linspace(0, 6, 200).reshape(-1, 1)

    inducing_demo()
