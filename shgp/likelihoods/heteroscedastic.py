import numpy as np
import tensorflow as tf

from gpflow.base import Parameter
from gpflow.likelihoods.base import Likelihood


class ParametricHeteroscedastic(Likelihood):
    """
    The ParametricHeteroscedastic likelihood is a simple heteroscedastic likelihood that assumes
    a polynomial noise model (this will be changed in future).
    Very small uncertainties can lead to numerical instability during the optimization process.
    A lower bound of 1e-6 is therefore imposed on the likelihood variance by default.
    """

    DEFAULT_VARIANCE_LOWER_BOUND = 1e-1

    def __init__(self, variance: float, variance_lower_bound=DEFAULT_VARIANCE_LOWER_BOUND):
        """
        :param variance: Input-dependent noise variance; all must be greater than
            ``variance_lower_bound``.
        :param variance_lower_bound: The lower (exclusive) bound of each element of ``variance``.
        :param kwargs: Keyword arguments forwarded to :class:`ScalarLikelihood`.
        """

        super().__init__(latent_dim=1, observation_dim=1)

        if variance < variance_lower_bound:
            raise ValueError(
                f"All variances of the ParametricHeteroscedastic likelihood must be strictly greater than {variance_lower_bound}"
            )

        self.variance_lower_bound = variance_lower_bound
        # TODO: Does this want to start at 0.0?
        # It works much better if set a value near true
        self.a = Parameter(0.0)
        self.b = Parameter(0.0)
        self.c = Parameter(variance)

    def noise_variance(self, F):
        noise = tf.abs(self.a * F**2 + self.b * F + self.c)
        return tf.clip_by_value(noise, self.variance_lower_bound, np.float("inf"))

    def _log_prob(self, F, Y):
        raise NotImplementedError

    def _predict_mean_and_var(self, Fmu, Fvar):
        raise NotImplementedError

    def _predict_log_density(self, Fmu, Fvar, Y):
        raise NotImplementedError

    def _variational_expectations(self, Fmu, Fvar, Y):
        raise NotImplementedError
