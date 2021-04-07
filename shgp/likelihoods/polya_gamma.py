import abc
import numpy as np
import tensorflow as tf

from gpflow.likelihoods.base import Likelihood


class PolyaGammaLikelihood(Likelihood, metaclass=abc.ABCMeta):
    """
    PolyaGammaLikelihood is a heteroscedastic likelihood based on Polya-Gamma data augmentation.
    This class allows us to directly compute the optimal variance given a set of data.
    Very small uncertainties can lead to numerical instability during the optimization process.
    A lower bound of 1e-6 is therefore imposed on the likelihood variance by default.
    """

    def __init__(self, num_data: int, variance_lower_bound: float = 1e-6):
        """
        :param variance_lower_bound: The lower (exclusive) bound of ``variance``.
        """

        super().__init__(latent_dim=1, observation_dim=1)

        self.variance_lower_bound = variance_lower_bound
        self.c_i = tf.zeros(num_data) + 0.1  # TODO: How to initialise?

    def noise_variance(self, Fmu, Fvar):
        """
            Computes the noise (theta^-1) of datapoints corresponding to predicted mean and variance.
            :param Fmu: a 1D NumPy array containing the mean values of q(f).
            :param Fvar: a 1D NumPy array containing the marginal variances of q(f).
            :return: array of noise values.
        """
        c_i = PolyaGammaLikelihood.compute_c_i(Fmu, Fvar)
        theta = self.compute_theta(c_i)
        noise = tf.math.reciprocal(theta)
        return tf.clip_by_value(noise, self.variance_lower_bound, np.float("inf"))

    def update_c_i(self, Fmu, Fvar):
        self.c_i = PolyaGammaLikelihood.compute_c_i(Fmu, Fvar)

    @staticmethod
    def compute_c_i(Fmu, Fvar):
        """
            Computes the optimal c_i in closed form.
            :param Fmu: a 1D NumPy array containing the mean values of q(f).
            :param Fvar: a 1D NumPy array containing the marginal variances of q(f).
            :return: array of c_i values.
        """
        c_i = tf.sqrt(tf.square(Fmu) + Fvar)
        return tf.cast(c_i, dtype=tf.float64)

    def compute_theta(self, c_i=None):
        """
            Calculates Θ = diag(1/(2c_i) * tanh(c_i/2)).
            :param c_i: an optional 1D NumPy array containing the Polya-Gamma random variables.
            :return: a 1D NumPy array of theta values.
        """
        # if c_i not provided, use local variables
        if c_i is None:
            c_i = self.c_i
        theta = 0.5 * tf.math.reciprocal(c_i) * tf.tanh(0.5 * c_i)
        return tf.cast(theta, dtype=tf.float64)

    def kl_term(self):
        """
            Calculates KL[q(ω) || p(ω)] = KL[PG(1, c) || PG(1, 0)].
            :return: the KL divergence.
        """
        half_c_i = 0.5 * self.c_i
        return tf.reduce_sum(tf.math.log(tf.cosh(half_c_i)) - 0.5 * half_c_i * tf.tanh(half_c_i))

    def _log_prob(self, F, Y):
        raise NotImplementedError

    def _predict_mean_and_var(self, Fmu, Fvar):
        raise NotImplementedError

    def _predict_log_density(self, Fmu, Fvar, Y):
        raise NotImplementedError

    def _variational_expectations(self, Fmu, Fvar, Y):
        raise NotImplementedError
