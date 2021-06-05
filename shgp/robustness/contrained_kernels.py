import tensorflow_probability as tfp

from gpflow import Parameter
from gpflow.kernels import SquaredExponential
from gpflow.utilities import to_default_float


class ConstrainedExpSEKernel(SquaredExponential):
    """"
    An implementation of the SquaredExponential kernel, wherein the kernel
    parameters are constrained by an exponential bijector. This means that
    they cannot grow too large or too small which helps to avoid Cholesky errors.
    This sometimes does come at the sacrifice of a small change in ELBO.
    """

    def __init__(self):
        super().__init__()

        constrained_transform = tfp.bijectors.Exp()
        var_constrained_transform = tfp.bijectors.Exp()
        self.lengthscales = Parameter(
            self.lengthscales.numpy(),
            transform=constrained_transform
        )
        self.variance = Parameter(
            self.variance.numpy(),
            transform=var_constrained_transform
        )


class ConstrainedSigmoidSEKernel(SquaredExponential):
    """"
    An implementation of the SquaredExponential kernel, wherein the kernel
    parameters are constrained. This means that they cannot grow too large
    or too small which helps to avoid Cholesky errors. This sometimes does come
    at the sacrifice of a small change in ELBO, but generally helps models
    to converge slightly faster.
    """

    def __init__(
        self,
        min_lengthscale=1e-20, min_variance=1e-20,
        max_lengthscale=1000.0, max_variance=1000.0
    ):
        super().__init__()

        constrained_transform = tfp.bijectors.Sigmoid(
            to_default_float(min_lengthscale),
            to_default_float(max_lengthscale),
        )
        var_constrained_transform = tfp.bijectors.Sigmoid(
            to_default_float(min_variance),
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
