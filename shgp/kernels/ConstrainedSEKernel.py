import tensorflow_probability as tfp

from gpflow import Parameter
from gpflow.config import default_positive_minimum
from gpflow.kernels import SquaredExponential
from gpflow.utilities import to_default_float


class ConstrainedSEKernel(SquaredExponential):
    """"
    An implementation of the SquaredExponential kernel, wherein the kernel
    parameters are constrained. This means that they cannot grow too large
    or too small and helps to avoid Cholesky errors. This sometimes does come
    at the sacrifice of a small change in ELBO, but is better than errors!
    It also generally helps models to converge slightly faster.
    """

    # TODO: Constraints should be set depending on number of dimensions - could this be more robust than a naive limit?
    def __init__(self, max_lengthscale=2000.0, max_variance=2000.0):
        super().__init__()

        constrained_transform = tfp.bijectors.Sigmoid(
            to_default_float(default_positive_minimum()),
            to_default_float(max_lengthscale),
        )
        var_constrained_transform = tfp.bijectors.Sigmoid(
            to_default_float(default_positive_minimum()),
            to_default_float(max_variance),
        )
        self.lengthscales = Parameter(
            self.lengthscales.numpy(),
            transform=constrained_transform
        )
        self.variance = Parameter(
            self.variance.numpy(),
            transform=var_constrained_transform
        )
