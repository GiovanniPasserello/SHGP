import numpy as np
import tensorflow as tf

from gpflow.likelihoods import Bernoulli

"""
For use in an SVGP model to make it equivalent to PGPR at convergence.
"""


class PolyaGammaBernoulli(Bernoulli):
    def __init__(self):
        Bernoulli.__init__(self, invlink=tf.sigmoid)

    def variational_expectations(self, Fmu, Fvar, Y):
        """
        Calculates the variational expectations used by an SVGP model.

        :param Fmu: a 1D NumPy array containing the mean values of q(f).
        :param Fvar: a 1D NumPy array containing the marginal variances of q(f).
        :param Y: array of ground truth labels.
        :return: the variational expectations.
        """
        # Y must be in (-1, +1), not (0, 1)
        assert_01 = tf.Assert(tf.reduce_all((Y == 0.0) | (Y == 1.0)), [Y])
        with tf.control_dependencies([assert_01]):
            Y = Y * 2.0 - 1.0

        c2 = Fmu ** 2 + Fvar
        c = tf.sqrt(c2)
        theta = tf.tanh(c / 2) / (2 * c)
        varexp = 0.5 * (Y * Fmu - theta * c2)

        return varexp - PolyaGammaBernoulli.kl_term(c) - np.log(2.0)

    @staticmethod
    def kl_term(c_i):
        """
        Calculates KL[q(ω) || p(ω)] = KL[PG(1, c) || PG(1, 0)]

        :param c_i: array of values c
        :return: array of KL divergences
        """
        half_c_i = c_i / 2
        kl_i = tf.math.log(tf.cosh(half_c_i)) - half_c_i / 2 * tf.tanh(half_c_i)
        return kl_i
