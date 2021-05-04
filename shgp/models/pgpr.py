from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from gpflow.config import default_float
from gpflow.covariances.dispatch import Kuf, Kuu
from gpflow.inducing_variables import InducingPoints
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.models.model import GPModel, MeanAndVariance
from gpflow.models.training_mixins import InputData, InternalDataTrainingLossMixin, RegressionData
from gpflow.models.util import data_input_to_tensor, inducingpoint_wrapper
from gpflow.utilities import to_default_float

from shgp.likelihoods.polya_gamma import PolyaGammaLikelihood
from shgp.robustness.linalg import robust_cholesky

tf.config.run_functions_eagerly(True)


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
        kuu = Kuu(self.inducing_variable, self.kernel)
        L = robust_cholesky(kuu)
        theta = tf.transpose(self.likelihood.compute_theta())
        theta_sqrt = tf.sqrt(theta)
        theta_sqrt_inv = tf.math.reciprocal(theta_sqrt)

        # compute intermediate matrices
        A = tf.linalg.triangular_solve(L, kuf, lower=True) * theta_sqrt
        AAT = tf.matmul(A, A, transpose_b=True)
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        LB = robust_cholesky(B)
        A_theta_sqrt_inv_err = tf.matmul(A * theta_sqrt_inv, err)
        c = 0.5 * tf.linalg.triangular_solve(LB, A_theta_sqrt_inv_err, lower=True)

        # compute log marginal bound
        bound = -output_dim * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound -= 0.5 * output_dim * tf.reduce_sum(Kdiag * theta)
        bound += 0.5 * output_dim * tf.reduce_sum(tf.linalg.diag_part(AAT))
        bound -= num_data * np.log(2)
        bound -= self.likelihood.kl_term()

        return bound

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
        kuu = Kuu(self.inducing_variable, self.kernel)
        kus = Kuf(self.inducing_variable, self.kernel, Xnew)
        L = robust_cholesky(kuu)
        theta = tf.transpose(self.likelihood.compute_theta())
        theta_sqrt = tf.sqrt(theta)
        theta_sqrt_inv = tf.math.reciprocal(theta_sqrt)

        # compute intermediate matrices
        A = tf.linalg.triangular_solve(L, kuf, lower=True) * theta_sqrt
        AAT = tf.matmul(A, A, transpose_b=True)
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        LB = robust_cholesky(B)
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
        kuu = Kuu(self.inducing_variable, self.kernel)
        theta = tf.transpose(self.likelihood.compute_theta())

        # compute intermediate matrices
        err = Y_data - self.mean_function(X_data)
        kuf_theta = kuf * theta
        sig = kuu + tf.matmul(kuf_theta, kuf, transpose_b=True)
        sig_sqrt = robust_cholesky(sig)
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
