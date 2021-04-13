import gpflow
import matplotlib.pyplot as plt
import numpy as np

from shgp.likelihoods.heteroscedastic import HeteroscedasticPolynomial
from shgp.models.hgpr import HGPR
from tensorflow_probability import distributions


np.random.seed(42)


if __name__ == "__main__":
    X = np.genfromtxt("../data/snelson/train_inputs").reshape(-1, 1)
    Y = np.genfromtxt("../data/snelson/train_outputs").reshape(-1, 1)
    xx = np.genfromtxt("../data/snelson/test_inputs").reshape(-1, 1)

    # TODO: Another likelihood -> GP likelihood would be much better if we know how to fit it.
    # Currently we don't have a way to fit without knowing the noise
    likelihood = HeteroscedasticPolynomial(degree=1)

    # Shared metadata
    inducing_locs = X

    kernel1 = gpflow.kernels.SquaredExponential(lengthscales=0.2)
    inducing_vars1 = gpflow.inducing_variables.InducingPoints(inducing_locs)
    model1 = gpflow.models.SGPR((X, Y), kernel=kernel1, inducing_variable=inducing_vars1)
    gpflow.optimizers.Scipy().minimize(model1.training_loss, variables=model1.trainable_variables)
    print("sgpr trained")

    kernel2 = gpflow.kernels.SquaredExponential(lengthscales=0.2)
    inducing_vars2 = gpflow.inducing_variables.InducingPoints(inducing_locs)
    model2 = HGPR((X, Y), kernel=kernel2, inducing_variable=inducing_vars2, likelihood=likelihood)
    gpflow.optimizers.Scipy().minimize(model2.training_loss, variables=model2.trainable_variables)
    print("hgpr trained")

    mu1, var1 = model1.predict_f(xx)
    var1_y = var1 + model1.likelihood.variance
    mu2, var2 = model2.predict_f(xx)
    _, var2_y = model2.predict_y(xx)

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

    fig.tight_layout(pad=4)
    ax1.set_title('SGPR')
    ax2.set_title('HGPR')
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')

    print(model1.elbo(), model2.elbo())
    print(model2.likelihood.trainable_parameters)

    plt.show()
