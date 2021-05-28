import gpflow
import matplotlib.pyplot as plt
import numpy as np

from gpflow.models.util import inducingpoint_wrapper

from shgp.data.utils import generate_polynomial_noise_data
from shgp.inducing.initialisation_methods import h_greedy_variance, uniform_subsample
from shgp.likelihoods.heteroscedastic import HeteroscedasticPolynomial
from shgp.models.hgpr import HGPR


np.random.seed(42)


"""
This class effectively implements SHGP/shgp/utilities/train_pgpr, but for HGPR.
"""


if __name__ == "__main__":
    NUM_DATA = 200
    X, Y, NoiseVar = generate_polynomial_noise_data(NUM_DATA)
    xx = np.linspace(-5, 5, 200)[:, None]
    num_inducing = 20

    # Uniform subsampling with gradient-based optimisation
    likelihood1 = HeteroscedasticPolynomial(degree=2)
    kernel1 = gpflow.kernels.SquaredExponential()
    inducing_vars1 = uniform_subsample(X, num_inducing)
    model1 = HGPR((X, Y), kernel=kernel1, inducing_variable=inducing_vars1, likelihood=likelihood1)
    gpflow.optimizers.Scipy().minimize(model1.training_loss, variables=model1.trainable_variables)
    mu1, var1 = model1.predict_f(xx)
    _, var1_y = model1.predict_y(xx)
    elbo1 = model1.elbo()

    # Heteroscedastic greedy variance selection
    likelihood2 = HeteroscedasticPolynomial(degree=2)
    kernel2 = gpflow.kernels.SquaredExponential()
    model2 = HGPR((X, Y), kernel=kernel2, inducing_variable=X.copy(), likelihood=likelihood2)
    prev_elbo = model2.elbo()
    while True:
        # Reinitialise inducing points
        inducing_locs2, inducing_idx2 = h_greedy_variance(X, model2.likelihood.noise_variance(X), num_inducing, kernel2)
        inducing_vars2 = gpflow.inducing_variables.InducingPoints(inducing_locs2)
        model2.inducing_variable = inducingpoint_wrapper(inducing_vars2)
        gpflow.set_trainable(model2.inducing_variable, False)

        # Optimise model
        gpflow.optimizers.Scipy().minimize(model2.training_loss, variables=model2.trainable_variables)

        # Check convergence
        next_elbo = model2.elbo()
        print("Previous ELBO: {}, Next ELBO: {}".format(prev_elbo, next_elbo))
        if np.abs(next_elbo - prev_elbo) <= 1e-6:
            break
        prev_elbo = next_elbo

    print("Final number of inducing points:", model2.inducing_variable.num_inducing)

    mu2, var2 = model2.predict_f(xx)
    _, var2_y = model2.predict_y(xx)
    elbo2 = model2.elbo()

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

    # Plot the inducing points on the mean line - only the x-coordinate really matters
    inducing_inputs1 = model1.inducing_variable.Z.variables[0]
    inducing_outputs1, _ = model1.predict_f(inducing_inputs1)
    inducing_outputs2, _ = model2.predict_f(X[inducing_idx2])
    ax1.scatter(inducing_inputs1, inducing_outputs1, c="b", label='ind point', zorder=1000)
    ax2.scatter(X[inducing_idx2].squeeze(), inducing_outputs2, c="b", label='ind point', zorder=1000)

    fig.tight_layout(pad=4)
    ax1.set_title('Optimized Naive Selection')
    ax2.set_title('Heteroscedastic Greedy Variance')
    ax1.set_xlim(-5, 5)
    ax2.set_xlim(-5, 5)
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')

    print(elbo1, elbo2)

    plt.show()
