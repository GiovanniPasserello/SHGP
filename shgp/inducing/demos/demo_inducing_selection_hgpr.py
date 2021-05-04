import gpflow
import matplotlib.pyplot as plt
import numpy as np

from gpflow.models.util import inducingpoint_wrapper

from shgp.inducing.initialisation_methods import h_greedy_variance
from shgp.likelihoods.heteroscedastic import HeteroscedasticPolynomial
from shgp.models.hgpr import HGPR


np.random.seed(42)
NUM_DATA = 200


def generate_polynomial_noise_data(N=NUM_DATA):
    X = np.random.rand(N)[:, None] * 10 - 5  # Inputs, shape N x 1
    X = np.sort(X.flatten()).reshape(N, 1)
    F = 2.5 * np.sin(6 * X) + np.cos(3 * X)  # Mean function values
    NoiseVar = np.abs(0.25 * X**2 + 0.1 * X)  # Quadratic noise variances
    Y = F + np.random.randn(N, 1) * np.sqrt(NoiseVar)  # Noisy data
    return X, Y, NoiseVar


if __name__ == "__main__":
    X, Y, NoiseVar = generate_polynomial_noise_data()
    xx = np.linspace(-5, 5, 200)[:, None]
    num_inducing = 20

    # Naive random selection and optimisations
    likelihood1 = HeteroscedasticPolynomial(degree=2)
    kernel1 = gpflow.kernels.SquaredExponential(lengthscales=0.2)
    inducing_idx1 = np.random.choice(np.arange(NUM_DATA), size=num_inducing, replace=False)
    inducing_vars1 = gpflow.inducing_variables.InducingPoints(X[inducing_idx1])
    model1 = HGPR((X, Y), kernel=kernel1, inducing_variable=inducing_vars1, likelihood=likelihood1)
    gpflow.optimizers.Scipy().minimize(model1.training_loss, variables=model1.trainable_variables)
    mu1, var1 = model1.predict_f(xx)
    _, var1_y = model1.predict_y(xx)
    elbo1 = model1.elbo()

    # Greedy variance selection
    threshold = 1e-6
    likelihood2 = HeteroscedasticPolynomial(degree=2)
    kernel2 = gpflow.kernels.SquaredExponential(lengthscales=0.2)
    inducing_locs2, inducing_idx2 = h_greedy_variance(X, likelihood2.noise_variance(X), num_inducing, kernel2, threshold)
    inducing_vars2 = gpflow.inducing_variables.InducingPoints(inducing_locs2)
    model2 = HGPR((X, Y), kernel=kernel2, inducing_variable=inducing_vars2, likelihood=likelihood2)
    gpflow.set_trainable(model2.inducing_variable, False)
    prev_elbo = model2.elbo()
    # iter_limit = 10  # to avoid infinite loops
    while True:
        gpflow.optimizers.Scipy().minimize(model2.training_loss, variables=model2.trainable_variables)

        next_elbo = model2.elbo()
        print("Previous ELBO: {}, Next ELBO: {}".format(prev_elbo, next_elbo))
        if np.abs(next_elbo - prev_elbo) <= 1e-6:  # or iter_limit == 0:
            break

        inducing_locs2, inducing_idx2 = h_greedy_variance(X, model2.likelihood.noise_variance(X), num_inducing, kernel2, threshold)
        inducing_vars2 = gpflow.inducing_variables.InducingPoints(inducing_locs2)
        model2.inducing_variable = inducingpoint_wrapper(inducing_vars2)
        gpflow.set_trainable(model2.inducing_variable, False)

        prev_elbo = next_elbo
        # iter_limit -= 1

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

    inducing_inputs = model1.inducing_variable.Z.variables[0]
    inducing_outputs, _ = model1.predict_f(inducing_inputs)
    ax1.scatter(inducing_inputs, inducing_outputs, c="b", label='ind point', zorder=1000)
    ax2.scatter(X[inducing_idx2].squeeze(), Y[inducing_idx2].squeeze(), c="b", label='ind point', zorder=1000)

    # Inspect average noise of inducing and non-inducing points
    print(model2.likelihood.noise_variance(X).numpy().flatten()[inducing_idx2].mean())
    print(model2.likelihood.noise_variance(X).numpy().flatten()[np.where([a not in inducing_idx2 for a in np.arange(50)])].mean())

    fig.tight_layout(pad=4)
    ax1.set_title('Optimized Naive Selection')
    ax2.set_title('Heteroscedastic Greedy Variance')
    ax1.set_xlim(-5, 5)
    ax2.set_xlim(-5, 5)
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')

    print(elbo1, elbo2)

    plt.show()
