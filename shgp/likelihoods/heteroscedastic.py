import abc
import numpy as np
import tensorflow as tf

from gpflow.base import Parameter
from gpflow.likelihoods.base import Likelihood
from tensorflow_probability import distributions

DEFAULT_VARIANCE_LOWER_BOUND = 1e-6


class Heteroscedastic(Likelihood, metaclass=abc.ABCMeta):
    """
    The Heteroscedastic likelihood is a base class for any heteroscedastic likelihood.
    We define a new class that inherits this for any heteroscedastic noise model we want to use.
    Very small uncertainties can lead to numerical instability during the optimization process.
    A lower bound of 1e-6 is therefore imposed on the likelihood variance by default.
    """

    def __init__(self, variance: float, variance_lower_bound: float, latent_dim: int, observation_dim: int):
        """
        :param variance: Input-dependent noise variance; must be greater than ``variance_lower_bound``.
        :param variance_lower_bound: The lower (exclusive) bound of ``variance``.
        :param latent_dim: the dimension of the vector F of latent functions for a single data point
        :param observation_dim: the dimension of the observation vector Y for a single data point
        """

        super().__init__(latent_dim=latent_dim, observation_dim=observation_dim)

        if variance < variance_lower_bound:
            raise ValueError(
                f"Intial variance of the Heteroscedastic likelihood must be strictly greater than {variance_lower_bound}"
            )

        self.variance_lower_bound = variance_lower_bound

    def noise_variance(self, X):
        noise = self._noise_variance(X)
        return tf.clip_by_value(noise, self.variance_lower_bound, np.float("inf"))

    @abc.abstractmethod
    def _noise_variance(self, X):
        raise NotImplementedError

    def _log_prob(self, F, Y):
        raise NotImplementedError

    def _predict_mean_and_var(self, Fmu, Fvar):
        raise NotImplementedError

    def _predict_log_density(self, Fmu, Fvar, Y):
        raise NotImplementedError

    def _variational_expectations(self, Fmu, Fvar, Y):
        raise NotImplementedError


# Example noise model
class HeteroscedasticPolynomial(Heteroscedastic):
    """
    The HeteroscedasticPolynomial likelihood is a simple heteroscedastic likelihood that assumes
    a polynomial noise model up to some degree.
    Very small uncertainties can lead to numerical instability during the optimization process.
    A lower bound of 1e-6 is therefore imposed on the likelihood variance by default.
    """

    def __init__(self, degree: int, variance: float = 1.0, variance_lower_bound: float = DEFAULT_VARIANCE_LOWER_BOUND):
        """
        :param degree: Degree of the polynomial noise model.
        :param variance: Input-dependent noise variance; must be greater than ``variance_lower_bound``.
        :param variance_lower_bound: The lower (exclusive) bound of ``variance``.
        """

        super().__init__(variance, variance_lower_bound, latent_dim=1, observation_dim=1)

        self.degree = degree
        self.degrees = [Parameter(0.0) for _ in range(degree)]
        self.bias = Parameter(variance)

    def _noise_variance(self, X):
        var = 0.0
        for degree, scalar in enumerate(self.degrees):
            var += scalar * X ** (degree + 1)
        var += self.bias
        return tf.abs(var)


# Example noise model
class HeteroscedasticGaussian(Heteroscedastic):
    """
    The HeteroscedasticPolynomial likelihood is a simple heteroscedastic likelihood that assumes
    a Gaussian noise model.
    Very small uncertainties can lead to numerical instability during the optimization process.
    A lower bound of 1e-6 is therefore imposed on the likelihood variance by default.
    """

    def __init__(self, center: float = 0.0, variance: float = 1.0, variance_lower_bound: float = DEFAULT_VARIANCE_LOWER_BOUND):
        """
        :param center: The center of the Gaussian noise distribution.
        :param variance: The noise variance of the Gaussian noise distribution; must be greater than ``variance_lower_bound``.
        :param variance_lower_bound: The lower (exclusive) bound of ``variance``.
        """

        super().__init__(variance, variance_lower_bound, latent_dim=1, observation_dim=1)

        self.mu = Parameter(center)
        self.var = Parameter(variance)

    def _noise_variance(self, X):
        normal = distributions.Normal(self.mu, self.var)
        centered = (X - self.mu) / tf.sqrt(self.var)
        return normal.prob(centered)
