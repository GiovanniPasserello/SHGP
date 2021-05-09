import gpflow
import matplotlib.pyplot as plt
import numpy as np

from shgp.data.utils import generate_polynomial_noise_data
from shgp.likelihoods.heteroscedastic import HeteroscedasticPolynomial, HeteroscedasticGP
from shgp.models.hgpr import HGPR


np.random.seed(42)


if __name__ == "__main__":
    NUM_DATA = 120
    X, Y, NoiseVar = generate_polynomial_noise_data(NUM_DATA)
    X_test = np.linspace(-5, 5, 200)[:, None]

    # Inducing points
    inducing_locs = X
    inducing_vars1 = gpflow.inducing_variables.InducingPoints(inducing_locs.copy())
    inducing_vars2 = gpflow.inducing_variables.InducingPoints(inducing_locs.copy())

    # Simple heteroscedastic likelihood or more complex GP likelihood (choose either)
    # likelihood = HeteroscedasticPolynomial(degree=2)
    # Requires knowledge of NoiseVar.
    likelihood_kernel = gpflow.kernels.Matern52()
    likelihood = HeteroscedasticGP((X, NoiseVar), likelihood_kernel, inducing_vars2)

    # Model definitions
    kernel1 = gpflow.kernels.SquaredExponential(lengthscales=0.2)
    model1 = gpflow.models.SGPR((X, Y), kernel=kernel1, inducing_variable=inducing_vars1)
    gpflow.set_trainable(model1.inducing_variable, False)
    gpflow.optimizers.Scipy().minimize(model1.training_loss, variables=model1.trainable_variables)
    print("sgpr trained")

    kernel2 = gpflow.kernels.SquaredExponential(lengthscales=0.2)
    model2 = HGPR((X, Y), kernel=kernel2, inducing_variable=inducing_vars2, likelihood=likelihood)
    gpflow.set_trainable(model2.inducing_variable, False)
    gpflow.optimizers.Scipy().minimize(model2.training_loss, variables=model2.trainable_variables)
    print("hgpr trained")

    # Make predictions
    mu1, var1 = model1.predict_f(X_test)
    _, var1_y = model1.predict_y(X_test)
    mu2, var2 = model2.predict_f(X_test)
    _, var2_y = model2.predict_y(X_test)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6))
    ax1.plot(X_test, mu1, "C0", label='mean')
    ax1.plot(X_test, mu1 + 2 * np.sqrt(var1), "C0", lw=0.5, label='f_var')
    ax1.plot(X_test, mu1 - 2 * np.sqrt(var1), "C0", lw=0.5)
    ax1.plot(X_test, mu1 + 2 * np.sqrt(var1_y), "r", lw=0.5, label='y_var')
    ax1.plot(X_test, mu1 - 2 * np.sqrt(var1_y), "r", lw=0.5)
    ax2.plot(X_test, mu2, "C0", label='mean')
    ax2.plot(X_test, mu2 + 2 * np.sqrt(var2), "C0", lw=0.5, label='f_var')
    ax2.plot(X_test, mu2 - 2 * np.sqrt(var2), "C0", lw=0.5)
    ax2.plot(X_test, mu2 + 2 * np.sqrt(var2_y), "r", lw=0.5, label='y_var')
    ax2.plot(X_test, mu2 - 2 * np.sqrt(var2_y), "r", lw=0.5)

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

    print(model1.elbo(), model2.elbo())

    plt.show()
