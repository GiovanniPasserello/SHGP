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

from shgp.likelihoods.heteroscedastic import HeteroscedasticLikelihood

tf.config.run_functions_eagerly(True)


class HGPR(GPModel, InternalDataTrainingLossMixin):
    """
        Heteroscedastic implementation of SGPR. The key reference is

        @inproceedings{titsias2009variational,
            title={Variational learning of inducing variables in sparse Gaussian processes},
            author={Titsias, Michalis K},
            booktitle={International Conference on Artificial Intelligence and Statistics},
            pages={567--574},
            year={2009}
        }
    """

    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        inducing_variable: InducingPoints,
        likelihood: HeteroscedasticLikelihood,
        *,
        mean_function: Optional[MeanFunction] = None,
        num_latent_gps: Optional[int] = None
    ):
        """
        `data`: a tuple of (X, Y), where the inputs X has shape [N, D]
            and the outputs Y has shape [N, R].
        `inducing_variable`: an InducingPoints instance or a matrix of
            the pseudo inputs Z, of shape [M, D].
        `kernel`, `mean_function` are appropriate GPflow objects
        This method only works with a HeteroscedasticLikelihood.
        """

        X_data, Y_data = data_input_to_tensor(data)

        self.likelihood = likelihood
        num_latent_gps = Y_data.shape[-1] if num_latent_gps is None else num_latent_gps

        super().__init__(kernel, likelihood, mean_function, num_latent_gps=num_latent_gps)

        self.data = X_data, Y_data
        self.num_data = X_data.shape[0]

        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

    def maximum_log_likelihood_objective(self, *args, **kwargs) -> tf.Tensor:
        return self.elbo()

    def elbo(self) -> tf.Tensor:
        """
        Computes a lower bound on the marginal likelihood of the heteroscedastic GP.
        """

        # metadata
        X_data, Y_data = self.data
        num_inducing = self.inducing_variable.num_inducing
        num_data = to_default_float(tf.shape(Y_data)[0])
        output_dim = to_default_float(tf.shape(Y_data)[1])

        # compute initial matrices
        err = Y_data - self.mean_function(X_data)
        Kdiag = self.kernel(X_data, full_cov=False)
        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        L = tf.linalg.cholesky(kuu)
        lmbda = tf.transpose(self.likelihood.noise_variance(X_data))
        rlmbda = tf.math.reciprocal(lmbda)  # lambda^-1
        rsigma = tf.sqrt(rlmbda)  # lambda^-1/2

        # compute intermediate matrices
        A = tf.linalg.triangular_solve(L, kuf, lower=True) * rsigma
        AAT = tf.matmul(A, A, transpose_b=True)
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        A_rsig_err = tf.matmul(A * rsigma, err)
        c = tf.linalg.triangular_solve(LB, A_rsig_err, lower=True)

        # compute log marginal bound
        bound = -0.5 * num_data * output_dim * np.log(2 * np.pi)
        bound -= output_dim * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))
        bound -= 0.5 * output_dim * tf.reduce_sum(tf.math.log(lmbda))
        bound -= 0.5 * tf.reduce_sum(tf.square(err) * tf.transpose(rlmbda))
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound -= 0.5 * output_dim * tf.reduce_sum(Kdiag * rlmbda)
        bound += 0.5 * output_dim * tf.reduce_sum(tf.linalg.diag_part(AAT))

        return bound

    def predict_f(self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False) -> MeanAndVariance:
        """
        Computes the mean and variance of the latent function at some new points Xnew.
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
        lmbda = tf.transpose(self.likelihood.noise_variance(X_data))
        rlmbda = tf.math.reciprocal(lmbda)  # lambda^-1
        rsigma = tf.sqrt(rlmbda)  # lambda^-1/2

        # compute intermediate matrices
        A = tf.linalg.triangular_solve(L, kuf, lower=True) * rsigma
        AAT = tf.matmul(A, A, transpose_b=True)
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        A_rsig_err = tf.matmul(A * rsigma, err)
        c = tf.linalg.triangular_solve(LB, A_rsig_err)

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
        Predicts the mean and variance for unobserved values at some new points Xnew.
        """
        mean, var = self.predict_f(Xnew)
        return mean, var + self.likelihood.noise_variance(Xnew)

    def compute_qu(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Computes the mean and variance of q(u) = N(m,S), the variational distribution on inducing outputs.
        """

        # metadata
        X_data, Y_data = self.data

        # compute initial matrices
        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        lmbda = tf.transpose(self.likelihood.noise_variance(X_data))
        rlmbda = tf.math.reciprocal(lmbda)  # lambda^-1

        # compute intermediate matrices
        err = Y_data - self.mean_function(X_data)
        kuf_rlmbda = kuf * rlmbda
        kuf_rlmbda_err = tf.matmul(kuf_rlmbda, err)
        sig = kuu + tf.matmul(kuf_rlmbda, kuf, transpose_b=True)
        sig_sqrt = tf.linalg.cholesky(sig)
        sig_sqrt_inv_kuu = tf.linalg.triangular_solve(sig_sqrt, kuu)

        # compute distribution
        mu = (
            tf.matmul(
                sig_sqrt_inv_kuu,
                tf.linalg.triangular_solve(sig_sqrt, kuf_rlmbda_err),
                transpose_a=True
            )
        )
        cov = tf.matmul(sig_sqrt_inv_kuu, sig_sqrt_inv_kuu, transpose_a=True)

        return mu, cov
