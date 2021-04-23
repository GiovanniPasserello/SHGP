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
from gpflow.base import Parameter
from gpflow.utilities import triangular

tf.config.run_functions_eagerly(True)


class Wenzel(GPModel, InternalDataTrainingLossMixin):
    """
        Implementation of SVGP with Polya-Gamma data augmentation. The key reference is

        @article{Wenzel_Galy-Fajou_Donner_Kloft_Opper_2019,
            title={Efficient Gaussian Process Classification Using Pólya-Gamma Data Augmentation},
            author={Wenzel, Florian and Galy-Fajou, Théo and Donner, Christan and Kloft, Marius and Opper, Manfred},
            journal={Proceedings of the AAAI Conference on Artificial Intelligence},
            pages={5417-5424},
            year={2019},
        }

        This is for comparison against the novel PGPR model. It is expected
        that Wenzel will converge in the limit to PGPR.
    """

    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        inducing_variable: InducingPoints,
        q_mu=None,
        q_sqrt=None,
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
        """

        X_data, Y_data = data_input_to_tensor(data)

        # Y must be in (-1, +1), not (0, 1)
        assert_y = tf.Assert(tf.reduce_all((Y_data == 0.0) | (Y_data == 1.0)), [Y_data])
        with tf.control_dependencies([assert_y]):
            Y_data = Y_data * 2.0 - 1.0

        self.data = X_data, Y_data
        self.num_data = X_data.shape[0]

        self.likelihood = PolyaGammaLikelihood(num_data=self.num_data, variance=0.1)
        num_latent_gps = Y_data.shape[-1] if num_latent_gps is None else num_latent_gps

        super().__init__(kernel, self.likelihood, mean_function, num_latent_gps=num_latent_gps)

        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        self._init_variational_parameters(q_mu, q_sqrt)

    def _init_variational_parameters(self, q_mu=None, q_sqrt=None):
        num_inducing = self.inducing_variable.num_inducing

        q_mu = np.zeros((num_inducing, self.num_latent_gps)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=default_float())  # [M, P]

        if q_sqrt is None:
            q_sqrt = [np.eye(num_inducing, dtype=default_float()) for _ in range(self.num_latent_gps)]
            q_sqrt = np.array(q_sqrt)
            self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [P, M, M]
        else:
            assert q_sqrt.ndim == 3
            self.num_latent_gps = q_sqrt.shape[0]
            self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [L|P, M, M]

    def maximum_log_likelihood_objective(self, *args, **kwargs) -> tf.Tensor:
        return self.elbo()

    def optimise_ci(self, num_iters=10):
        for _ in range(num_iters):
            Fmu, Fvar = self.predict_f(self.data[0])
            self.likelihood.update_c_i(Fmu, Fvar)

    def elbo(self) -> tf.Tensor:
        """
        Computes a lower bound on the marginal likelihood of the Wenzel GP.
        This is not an efficient implementation, it is simply for direct comparison.
        If this were to be used in practice, we would want to compute a Cholesky solution for stability.
        """

        m, S = self.compute_qu()

        # metadata
        X_data, Y_data = self.data
        num_data = to_default_float(self.num_data)
        num_inducing = to_default_float(self.inducing_variable.num_inducing)

        # compute initial matrices
        err = Y_data - self.mean_function(X_data)
        K = self.kernel(X_data, full_cov=True)
        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kfu = tf.transpose(kuf)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        kuu_inv = tf.linalg.inv(kuu)
        theta = tf.transpose(self.likelihood.compute_theta())
        theta_mat = tf.squeeze(tf.linalg.diag(theta))
        qnn = tf.matmul(tf.matmul(kfu, kuu_inv), kuf)
        k_tilde = K - qnn

        # compute log marginal bound
        bound = 0.5 * tf.math.log(tf.linalg.det(S))
        bound -= 0.5 * tf.math.log(tf.linalg.det(kuu))
        bound -= 0.5 * tf.linalg.trace(tf.matmul(kuu_inv, S))
        bound -= 0.5 * tf.matmul(tf.matmul(m, kuu_inv, transpose_a=True), m)
        bound += 0.5 * tf.matmul(tf.matmul(tf.matmul(err, kfu, transpose_a=True), kuu_inv), m)
        bound -= 0.5 * tf.linalg.trace(tf.matmul(theta_mat, k_tilde))
        term = tf.matmul(tf.matmul(tf.matmul(tf.matmul(kuu_inv, kuf), theta_mat), kfu), kuu_inv)
        bound -= 0.5 * tf.linalg.trace(tf.matmul(term, S))
        bound -= 0.5 * tf.matmul(tf.matmul(m, term, transpose_a=True), m)
        bound -= num_data * np.log(2)  # missed in =c
        bound += 0.5 * num_inducing  # missed in =c
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
        Computes the mean and variance of q(u) = N(mu, cov), the variational distribution on inducing outputs.
        """

        q_sqrt = tf.squeeze(self.q_sqrt)
        return self.q_mu, tf.matmul(q_sqrt, q_sqrt, transpose_b=True)
