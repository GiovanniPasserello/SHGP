import tensorflow as tf

from gpflow.base import Parameter
from gpflow.likelihoods.base import Likelihood


class ParametricHeteroscedastic(Likelihood):
    """
    The HeteroscedasticGaussian likelihood is appropriate where uncertainties associated with
    the data are believed to follow a normal distribution, with input-dependent noise.
    Very small uncertainties can lead to numerical instability during the optimization process.
    A lower bound of 1e-6 is therefore imposed on the likelihood variance by default.
    """

    DEFAULT_VARIANCE_LOWER_BOUND = 1e-6

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
        # TODO: Use fixed values for now, during development
        self.a = Parameter(0.25)
        self.b = Parameter(0.0)

    def noise_variance(self, F):
        return tf.abs(self.a * F + self.b) + 0.15

    def _log_prob(self, F, Y):
        raise NotImplementedError

    def _predict_mean_and_var(self, Fmu, Fvar):
        raise NotImplementedError

    def _predict_log_density(self, Fmu, Fvar, Y):
        raise NotImplementedError

    def _variational_expectations(self, Fmu, Fvar, Y):
        raise NotImplementedError
