from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from gpflow.config import default_float, default_jitter
from gpflow.covariances.dispatch import Kuf, Kuu
from gpflow.inducing_variables import InducingPoints
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.models.model import GPModel, MeanAndVariance
from gpflow.models.training_mixins import InputData, InternalDataTrainingLossMixin, RegressionData
from gpflow.models.util import data_input_to_tensor, inducingpoint_wrapper
from gpflow.utilities import to_default_float

from shgp.likelihoods.polya_gamma import PolyaGammaLikelihood

tf.config.run_functions_eagerly(True)


# TODO: Future work might consider superior methods for iterative optimisation of c_i.
class PGPR(GPModel, InternalDataTrainingLossMixin):
    """
        Collapsed implementation of Polya-Gamma GPR, based on the heteroscedastic
        implementation of SGPR with Polya-Gamma data augmentation. The key reference is

        @article{Wenzel_Galy-Fajou_Donner_Kloft_Opper_2019,
            title={Efficient Gaussian Process Classification Using Pólya-Gamma Data Augmentation},
            author={Wenzel, Florian and Galy-Fajou, Théo and Donner, Christan and Kloft, Marius and Opper, Manfred},
            journal={Proceedings of the AAAI Conference on Artificial Intelligence},
            pages={5417-5424},
            year={2019},
        }

        which is the SVGP building block for this work. This model is one of the key contributions
        of this thesis.
    """

    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        *,
        inducing_variable: Optional[InducingPoints] = None,
        mean_function: Optional[MeanFunction] = None,
        num_latent_gps: Optional[int] = None
    ):
        """
        `data`: a tuple of (X, Y), where the inputs X has shape [N, D]
            and the outputs Y has shape [N, R].
        `kernel`, `mean_function` are appropriate GPflow objects.
        `inducing_variable`: an InducingPoints instance or a matrix of
            the pseudo inputs Z, of shape [M, D].
        """

        X_data, Y_data = data_input_to_tensor(data)

        # Y_data must be in (-1, +1), not (0, 1)
        assert_y = tf.Assert(tf.reduce_all((Y_data == 0.0) | (Y_data == 1.0)), [Y_data])
        with tf.control_dependencies([assert_y]):
            Y_data = Y_data * 2.0 - 1.0

        self.data = X_data, Y_data
        self.num_data = X_data.shape[0]

        self.likelihood = PolyaGammaLikelihood(num_data=self.num_data, variance=0.1)
        num_latent_gps = Y_data.shape[-1] if num_latent_gps is None else num_latent_gps

        super().__init__(kernel, self.likelihood, mean_function, num_latent_gps=num_latent_gps)

        if inducing_variable is None:
            self.inducing_variable = inducingpoint_wrapper(data[0].copy())
        else:
            self.inducing_variable = inducingpoint_wrapper(inducing_variable)

    def maximum_log_likelihood_objective(self, *args, **kwargs) -> tf.Tensor:
        return self.elbo()

    def optimise_ci(self, num_iters=10):
        """
        Iteratively update the local parameters, this forms a cycle between
        updating c_i and q_u which we iterate a number of times. Typically we
        can choose num_iters < 10.
        """
        for _ in range(num_iters):
            Fmu, Fvar = self.predict_f(self.data[0])
            self.likelihood.update_c_i(Fmu, Fvar)

    def elbo(self) -> tf.Tensor:
        """
        Computes a lower bound on the marginal likelihood of the PGPR model.
        """

        # metadata
        X_data, Y_data = self.data
        num_inducing = to_default_float(self.inducing_variable.num_inducing)
        output_dim = to_default_float(tf.shape(Y_data)[1])
        num_data = to_default_float(self.num_data)

        # compute initial matrices
        err = Y_data - self.mean_function(X_data)
        Kdiag = self.kernel(X_data, full_cov=False)
        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        L = tf.linalg.cholesky(kuu)
        theta = tf.transpose(self.likelihood.compute_theta())
        theta_sqrt = tf.sqrt(theta)
        theta_sqrt_inv = tf.math.reciprocal(theta_sqrt)

        # compute intermediate matrices
        A = tf.linalg.triangular_solve(L, kuf, lower=True) * theta_sqrt
        AAT = tf.matmul(A, A, transpose_b=True)
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        A_theta_sqrt_inv_err = tf.matmul(A * theta_sqrt_inv, err)
        c = 0.5 * tf.linalg.triangular_solve(LB, A_theta_sqrt_inv_err, lower=True)

        # compute log marginal bound
        bound = -output_dim * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound -= 0.5 * output_dim * tf.reduce_sum(Kdiag * theta)
        bound += 0.5 * output_dim * tf.reduce_sum(tf.linalg.diag_part(AAT))
        bound -= num_data * np.log(2)
        bound -= self.likelihood.kl_term()

        # TODO: This was expected to be constant (with fixed c_i)?
        #print(self.hgpr_elbo() - bound, self.hgpr_elbo_difference())

        # # TODO: remove - Temporary code investigating rank-1 ELBO difference bounds
        # ###########
        # Y = np.round(np.random.randn(len(Y_data)).reshape(-1, 1))
        # z = np.random.randn(1).reshape(-1, 1)
        # k = self.kernel(z, z)
        # kuu_inv = tf.linalg.inv(kuu)
        # ku = tf.reshape(self.kernel(self.inducing_variable.Z, z), (-1, 1))
        # c = k - tf.matmul(tf.matmul(ku, kuu_inv, transpose_a=True), ku)
        # c = tf.math.reciprocal(tf.math.sqrt(c))
        # kf = tf.reshape(self.kernel(X_data, z), (-1, 1))
        # b = tf.matmul(tf.matmul(kuf, kuu_inv, transpose_a=True), ku) - kf
        # b *= c
        # bbT = tf.matmul(b, b, transpose_b=True)
        # Qff = tf.matmul(tf.matmul(kuf, kuu_inv, transpose_a=True), kuf)
        # ###########
        # theta_mat = tf.squeeze(tf.linalg.diag(theta))
        # theta_inv_mat = tf.squeeze(tf.linalg.diag(tf.math.reciprocal(theta)))
        # A = tf.linalg.inv(theta_inv_mat + Qff + bbT)
        # E = tf.linalg.inv(theta_inv_mat + Qff)
        # D = tf.matmul(tf.matmul(E, bbT), E)
        # D /= (1 + tf.matmul(tf.matmul(b, E, transpose_a=True), b))
        # C = E - D
        # ###########
        # # first = tf.matmul(
        # #     tf.matmul(
        # #         Y,
        # #         bbT + tf.matmul(tf.matmul(Qff, D), Qff) + tf.matmul(tf.matmul(bbT, D), bbT) + tf.matmul(tf.matmul(bbT, D), Qff) + tf.matmul(tf.matmul(Qff, D), bbT),
        # #         transpose_a=True
        # #     ),
        # #     Y)
        # # second = tf.matmul(
        # #     tf.matmul(
        # #         Y,
        # #         tf.matmul(tf.matmul(bbT, E), Qff) + tf.matmul(tf.matmul(Qff, E), bbT) + tf.matmul(tf.matmul(bbT, E), bbT),
        # #         transpose_a=True
        # #     ),
        # #     Y)
        # # first = tf.matmul(
        # #     tf.matmul(
        # #         Y,
        # #         bbT + tf.matmul(tf.matmul(Qff, D), Qff),
        # #         transpose_a=True
        # #     ),
        # #     Y)
        # # second = tf.matmul(
        # #     tf.matmul(
        # #         Y,
        # #         tf.matmul(tf.matmul(bbT, C), Qff) + tf.matmul(tf.matmul(Qff, C), bbT) + tf.matmul(tf.matmul(bbT, C), bbT),
        # #         transpose_a=True
        # #     ),
        # #     Y)
        # Qff_plus = Qff + bbT
        # Sigma = kuu + tf.matmul(tf.matmul(kuf, theta_mat), kuf, transpose_b=True)
        # Sigma_sqrt = tf.linalg.cholesky(Sigma)
        # s_inv_kuf = tf.linalg.triangular_solve(Sigma_sqrt, kuf)
        # mid = tf.matmul(tf.matmul(tf.matmul(theta_mat, s_inv_kuf, transpose_b=True), s_inv_kuf), theta_mat)
        #
        # # first = tf.matmul(
        # #     tf.matmul(
        # #         Y,
        # #         bbT + tf.matmul(tf.matmul(Qff, E), Qff) + tf.matmul(tf.matmul(Qff_plus, D), Qff_plus) + tf.matmul(tf.matmul(Qff_plus, mid), Qff_plus),
        # #         transpose_a=True
        # #     ),
        # #     Y)
        # # second = tf.matmul(
        # #     tf.matmul(
        # #         Y,
        # #         tf.matmul(tf.matmul(Qff_plus, theta_mat), Qff_plus),
        # #         transpose_a=True
        # #     ),
        # #     Y)
        # # first = tf.matmul(
        # #     tf.matmul(
        # #         Y,
        # #         bbT + tf.matmul(tf.matmul(Qff, E), Qff) + tf.matmul(tf.matmul(Qff_plus, D), Qff_plus),
        # #         transpose_a=True
        # #     ),
        # #     Y)
        # # second = tf.matmul(
        # #     tf.matmul(
        # #         Y,
        # #         tf.matmul(tf.matmul(Qff_plus, E), Qff_plus),
        # #         transpose_a=True
        # #     ),
        # #     Y)
        # first = tf.matmul(
        #     tf.matmul(
        #         Y,
        #         bbT + tf.matmul(tf.matmul(Qff_plus, D), Qff_plus),
        #         transpose_a=True
        #     ),
        #     Y)
        # second = tf.matmul(
        #     tf.matmul(
        #         Y,
        #         tf.matmul(tf.matmul(Qff, E), bbT) + tf.matmul(tf.matmul(bbT, E), Qff) + tf.matmul(tf.matmul(bbT, E), bbT),
        #         transpose_a=True
        #     ),
        #     Y)
        #
        # if first - second < 0:
        #     print(first - second)
        # ##########

        return bound

    # TODO: Remove
    # For comparison against HGPR
    # def hgpr_elbo(self) -> tf.Tensor:
    #     """
    #     Construct a tensorflow function to compute the bound on the marginal
    #     likelihood. For a derivation of the terms in here, see *** TODO.
    #     """
    #
    #     # metadata
    #     X_data, Y_data = self.data
    #     num_inducing = self.inducing_variable.num_inducing
    #     num_data = to_default_float(tf.shape(Y_data)[0])
    #     output_dim = to_default_float(tf.shape(Y_data)[1])
    #
    #     # compute initial matrices
    #     err = Y_data - self.mean_function(X_data)
    #     Kdiag = self.kernel(X_data, full_cov=False)
    #     kuf = Kuf(self.inducing_variable, self.kernel, X_data)
    #     kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
    #     L = tf.linalg.cholesky(kuu)
    #     theta = tf.transpose(self.likelihood.compute_theta())
    #     theta_sqrt = tf.sqrt(theta)
    #
    #     # compute intermediate matrices
    #     A = tf.linalg.triangular_solve(L, kuf, lower=True) * theta_sqrt
    #     AAT = tf.matmul(A, A, transpose_b=True)
    #     B = AAT + tf.eye(num_inducing, dtype=default_float())
    #     LB = tf.linalg.cholesky(B)
    #     A_rsig_err = tf.matmul(A * theta_sqrt, err)
    #     c = tf.linalg.triangular_solve(LB, A_rsig_err, lower=True)
    #
    #     # compute log marginal bound
    #     bound = -0.5 * num_data * output_dim * np.log(2 * np.pi)
    #     bound -= output_dim * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))
    #     bound -= 0.5 * output_dim * tf.reduce_sum(tf.math.log(tf.math.reciprocal(theta)))
    #     bound -= 0.5 * tf.reduce_sum(tf.square(err) * tf.transpose(theta))
    #     bound += 0.5 * tf.reduce_sum(tf.square(c))
    #     bound -= 0.5 * output_dim * tf.reduce_sum(Kdiag * theta)
    #     bound += 0.5 * output_dim * tf.reduce_sum(tf.linalg.diag_part(AAT))
    #
    #     return bound
    #
    # def hgpr_elbo_difference(self):
    #     """
    #     The difference between HGPR and PGPR ELBOs.
    #     """
    #
    #     # metadata
    #     X_data, Y_data = self.data
    #     output_dim = to_default_float(tf.shape(Y_data)[1])
    #     num_data = to_default_float(tf.shape(Y_data)[0])
    #
    #     # compute initial matrices
    #     err = Y_data - self.mean_function(X_data)
    #     kuf = Kuf(self.inducing_variable, self.kernel, X_data)
    #     kfu = tf.transpose(kuf)
    #     kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
    #     theta = tf.transpose(self.likelihood.compute_theta())
    #
    #     theta_mat = tf.squeeze(tf.linalg.diag(theta))
    #     sig = kuu + tf.matmul(tf.matmul(kuf, theta_mat), kfu)
    #     sig_sqrt = tf.linalg.cholesky(sig)
    #     sig_sqrt_inv_kuf = tf.linalg.triangular_solve(sig_sqrt, kuf)
    #     mid = tf.matmul(sig_sqrt_inv_kuf, sig_sqrt_inv_kuf, transpose_a=True)
    #
    #     term1 = theta_mat
    #     term2 = tf.matmul(tf.matmul(theta_mat, mid), theta_mat)
    #     term3 = mid
    #     total = term1 - term2 + 0.25 * term3
    #
    #     # compute log marginal bound
    #     diff = 0.5 * num_data * np.log(2 * np.pi)
    #     diff += 0.5 * tf.matmul(tf.matmul(err, total, transpose_a=True), err)
    #     diff -= 0.5 * output_dim * tf.reduce_sum(tf.math.log(theta))
    #     diff -= num_data * np.log(2)
    #     diff -= self.likelihood.kl_term()
    #
    #     return diff

    def predict_f(self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False) -> MeanAndVariance:
        """
        Compute the mean and variance of the latent function at some new points Xnew.
        """

        # metadata
        X_data, Y_data = self.data
        num_inducing = self.inducing_variable.num_inducing

        # compute initial matrices
        err = Y_data - self.mean_function(X_data)
        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        kus = Kuf(self.inducing_variable, self.kernel, Xnew)
        L = tf.linalg.cholesky(kuu)
        theta = tf.transpose(self.likelihood.compute_theta())
        theta_sqrt = tf.sqrt(theta)
        theta_sqrt_inv = tf.math.reciprocal(theta_sqrt)

        # compute intermediate matrices
        A = tf.linalg.triangular_solve(L, kuf, lower=True) * theta_sqrt
        AAT = tf.matmul(A, A, transpose_b=True)
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        A_theta_sqrt_inv_err = tf.matmul(A * theta_sqrt_inv, err)
        c = 0.5 * tf.linalg.triangular_solve(LB, A_theta_sqrt_inv_err)

        # compute predictive
        tmp1 = tf.linalg.triangular_solve(L, kus, lower=True)
        tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
        mean = tf.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = (
                self.kernel(Xnew)
                + tf.matmul(tmp2, tmp2, transpose_a=True)
                - tf.matmul(tmp1, tmp1, transpose_a=True)
            )
            var = tf.tile(var[None, ...], [self.num_latent_gps, 1, 1])  # [P, N, N]
        else:
            var = (
                self.kernel(Xnew, full_cov=False)
                + tf.reduce_sum(tf.square(tmp2), 0)
                - tf.reduce_sum(tf.square(tmp1), 0)
            )
            var = tf.tile(var[:, None], [1, self.num_latent_gps])

        return mean + self.mean_function(Xnew), var

    def predict_y(self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False) -> MeanAndVariance:
        """
        Predict the mean and variance for unobserved values at some new points Xnew.
        """
        mean, var = self.predict_f(Xnew)
        return mean, var + self.likelihood.noise_variance(mean, var)

    def compute_qu(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Computes the mean and variance of q(u) = N(m,S), the variational distribution on inducing outputs.
        """

        # metadata
        X_data, Y_data = self.data

        # compute initial matrices
        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        theta = tf.transpose(self.likelihood.compute_theta())

        # compute intermediate matrices
        err = Y_data - self.mean_function(X_data)
        kuf_theta = kuf * theta
        sig = kuu + tf.matmul(kuf_theta, kuf, transpose_b=True)
        sig_sqrt = tf.linalg.cholesky(sig)
        sig_sqrt_inv_kuu = tf.linalg.triangular_solve(sig_sqrt, kuu)
        kuf_err = tf.matmul(kuf, err)

        # compute distribution
        mu = 0.5 * (
            tf.matmul(
                sig_sqrt_inv_kuu,
                tf.linalg.triangular_solve(sig_sqrt, kuf_err),
                transpose_a=True
            )
        )
        cov = tf.matmul(sig_sqrt_inv_kuu, sig_sqrt_inv_kuu, transpose_a=True)

        return mu, cov
