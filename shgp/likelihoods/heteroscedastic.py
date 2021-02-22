import numpy as np
import tensorflow as tf

from gpflow.base import Parameter
from gpflow.likelihoods.base import ScalarLikelihood
from gpflow.logdensities import multivariate_normal
from gpflow.utilities import positive


# TODO: Fix
class HeteroscedasticGaussian(ScalarLikelihood):
    """
    The HeteroscedasticGaussian likelihood is appropriate where uncertainties associated with
    the data are believed to follow a normal distribution, with input-dependent noise.
    Very small uncertainties can lead to numerical instability during the optimization process.
    A lower bound of 1e-6 is therefore imposed on the likelihood variance by default.
    """

    DEFAULT_VARIANCE_LOWER_BOUND = 1e-6

    def __init__(self, variance: tf.constant, variance_lower_bound=DEFAULT_VARIANCE_LOWER_BOUND, **kwargs):
        """
        :param variance: Input-dependent noise variance; all must be greater than
            ``variance_lower_bound``.
        :param variance_lower_bound: The lower (exclusive) bound of each element of ``variance``.
        :param kwargs: Keyword arguments forwarded to :class:`ScalarLikelihood`.
        """

        super().__init__(**kwargs)

        if tf.reduce_any(tf.less(variance, variance_lower_bound)):
            raise ValueError(
                f"All variances of the HeteroscedasticGaussian likelihood must be strictly greater than {variance_lower_bound}"
            )

        self.variance = Parameter(variance, transform=positive(lower=variance_lower_bound))

    def _scalar_log_prob(self, F, Y):
        cov = tf.linalg.diag(self.variance)
        return multivariate_normal(Y, F, cov)

    def _conditional_mean(self, F):
        return tf.identity(F)

    def _conditional_variance(self, F):
        return tf.linalg.diag(self.variance)  # TODO?

    def _predict_mean_and_var(self, Fmu, Fvar):
        return tf.identity(Fmu), Fvar + self.variance

    def _predict_log_density(self, Fmu, Fvar, Y):  # TODO?
        covF = tf.linalg.diag(Fvar)
        cov = tf.linalg.diag(self.variance)
        return tf.reduce_sum(multivariate_normal(Y, Fmu, covF + cov), axis=-1)

    # LOG LIKELIHOOD
    def _variational_expectations(self, Fmu, Fvar, Y):  # TODO: WRONG
        return tf.reduce_sum(
            -0.5 * np.log(2 * np.pi)
            - 0.5 * tf.math.log(self.variance)
            - 0.5 * ((Y - Fmu) ** 2 + Fvar) / self.variance,
            axis=-1,
        )
