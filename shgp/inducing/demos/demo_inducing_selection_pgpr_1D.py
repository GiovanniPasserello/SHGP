import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from shgp.data.dataset import PlatformDataset
from shgp.data.metadata_reinit import ReinitMetaDataset
from shgp.inducing.initialisation_methods import h_reinitialise_PGPR, k_means
from shgp.utilities.general import invlink
from shgp.utilities.train_pgpr import train_pgpr

np.random.seed(0)
tf.random.set_seed(0)


def inducing_demo():
    num_inducing, inner_iters, opt_iters, ci_iters = 15, 5, 100, 10

    # Uniform subsampling with gradient-based optimisation
    model1, elbo1 = train_pgpr(
        X, Y,
        inner_iters, opt_iters, ci_iters,
        kernel_type=gpflow.kernels.SquaredExponential,
        M=num_inducing,
        init_method=k_means,
        optimise_Z=True
    )

    # Heteroscedastic greedy variance selection
    model2, _ = train_pgpr(
        X, Y,
        inner_iters, opt_iters, ci_iters,
        kernel_type=gpflow.kernels.SquaredExponential,
        M=num_inducing,
        init_method=h_reinitialise_PGPR,
        reinit_metadata=ReinitMetaDataset()
    )
    elbo2 = model2.elbo()

    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8))

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
    inducing_inputs2 = model2.inducing_variable.Z.variables[0]
    inducing_outputs2, _ = model1.predict_f(inducing_inputs2)
    p_inducing_outputs2 = invlink(inducing_outputs2)
    ax1.scatter(inducing_inputs1, p_inducing_outputs1, c="b", label='ind point', zorder=1000)
    ax2.scatter(inducing_inputs2, p_inducing_outputs2, c="b", label='ind point', zorder=1000)

    fig.tight_layout(pad=4)
    ax1.set_title('Optimized Naive Selection')
    ax2.set_title('Polya-Gamma Greedy Variance')
    ax1.set_xlim((-2, 2))
    ax1.set_ylim((-0.5, 1.5))
    ax2.set_xlim((-2, 2))
    ax2.set_ylim((-0.5, 1.5))
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')

    print(elbo1, elbo2)

    plt.show()


if __name__ == "__main__":
    # Load data
    X, Y = PlatformDataset().load_data()
    X_test = np.linspace(-2, 2, 200).reshape(-1, 1)

    inducing_demo()
