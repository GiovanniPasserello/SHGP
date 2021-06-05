from typing import Callable, Optional, Type

import gpflow
import numpy as np
import tensorflow as tf

from gpflow.kernels.base import Kernel

from shgp.inducing.initialisation_methods import k_means
from shgp.utilities.train_pgpr import result

"""
Utilities for training SVGP models in a standard and extensible manner.
"""


def train_svgp(
    X: np.ndarray,
    Y: np.ndarray,
    M: int,
    *,
    train_iters: int = 100,
    kernel_type: Type[Kernel] = gpflow.kernels.SquaredExponential,
    init_method: Callable = k_means,
    optimise_Z: bool = False,
    X_test: Optional[np.ndarray] = None,
    Y_test: Optional[np.ndarray] = None
):
    """
    Train an SVGP model with the following parameters.
    This is the only function in this file that should be externally called.

    :param X: [N,D], the training feature data.
    :param Y: [N,1], the training label data.
    :param M: The number of inducing points, if using a sparse model.
    :param train_iters: The number of L-BFGS training iterations.
    :param kernel_type: The covariance kernel type for the SVGP model. We use a type, instead of an object
                        so that we can reinitialise in the case of an error.
    :param init_method: The inducing point initialisation method, if using a sparse model.
    :param optimise_Z: Allow gradient-based optimisation of the inducing inputs, if using a sparse model.
    :param X_test: [N_test,D], the test feature data.
    :param Y_test: [N_test,1], the test label data.
    :return: The final model and best evidence lower bound (or full set of metrics).
    """
    return _try_train_svgp(X, Y, M, train_iters, kernel_type, init_method, optimise_Z, X_test, Y_test)


def _try_train_svgp(X, Y, M, train_iters, kernel_type, init_method, optimise_Z, X_test, Y_test):
    """
    Train an SVGP model until completion.
    If we error, keep retrying until success - this is due to a spurious Cholesky/inversion error.
    """
    # Try to run the full optimisation cycle.
    try:
        model = gpflow.models.SVGP(
            kernel=gpflow.kernels.SquaredExponential(),
            likelihood=gpflow.likelihoods.Bernoulli(invlink=tf.sigmoid),
            inducing_variable=init_method(X, M).copy()
        )
        gpflow.set_trainable(model.inducing_variable, optimise_Z)
        gpflow.optimizers.Scipy().minimize(
            model.training_loss_closure((X, Y)),
            variables=model.trainable_variables,
            options=dict(maxiter=train_iters)
        )
        return result(model, model.elbo((X, Y)), X_test, Y_test)
    # If we fail due to a (spurious) Cholesky/inversion error, restart.
    except tf.errors.InvalidArgumentError as error:
        msg = error.message
        if "Cholesky" not in msg and "invertible" not in msg:
            raise error
        else:
            if "Cholesky" in msg:
                print("Cholesky error caught, retrying...")
            else:
                print("Inversion error caught, retrying...")
            return _try_train_svgp(X, Y, M, train_iters, kernel_type, init_method, optimise_Z, X_test, Y_test)
