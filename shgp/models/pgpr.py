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
        This method only works with a Heteroscedastic Gaussian likelihood, its variance is
        initialized to (`noise_variance`*I + diag[Kff - Kfu*Kuu^-1*Kuf]).
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
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. For a derivation of the terms in here, see *** TODO.
        """

        bound = None

        return bound

    def predict_f(self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False) -> MeanAndVariance:
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. For a derivation of the terms in here, see *** TODO.
        """

        mean, var = None, None

        return mean + self.mean_function(Xnew), var

    def predict_y(self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False) -> MeanAndVariance:
        """
        Predict the mean and variance for unobserved values at some new points
        Xnew. For a derivation of the terms in here, see *** TODO.
        """
        mean, var = self.predict_f(Xnew)
        return mean, var + self.likelihood.noise_variance(Xnew)

    def compute_qu(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Computes the mean and variance of q(u) = N(mu, cov), the variational distribution on
        inducing outputs.
        :return: mu, cov
        """

        mu, cov = None, None

        return mu, cov
