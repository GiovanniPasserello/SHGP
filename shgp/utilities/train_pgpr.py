from typing import Callable, Optional

import gpflow
import numpy as np
import tensorflow as tf

from gpflow.models.util import inducingpoint_wrapper
from gpflow.kernels import Kernel

from shgp.data.metadata_reinit import ReinitMetaDataset
from shgp.robustness.contrained_kernels import ConstrainedExpSEKernel
from shgp.models.pgpr import PGPR


def train_pgpr(
    X: np.ndarray,
    Y: np.ndarray,
    inner_iters: int,
    opt_iters: int,
    ci_iters: int,
    *,
    kernel: Kernel = ConstrainedExpSEKernel(),  # increased stability, with minor performance detriment.
    M: Optional[int] = None,
    init_method: Optional[Callable] = None,
    reinit_metadata: Optional[ReinitMetaDataset] = None,
    optimise_Z: bool = False
):
    """
        Train a PGPR model with the following parameters.
        This is the only function in this file that should be externally called.

        :param X: [N,D], the training feature data.
        :param Y: [N,1], the training label data.
        :param inner_iters: The number of iterations of the inner optimisation loop.
        :param opt_iters: The number of iterations of gradient-based optimisation of the kernel hyperparameters.
        :param ci_iters: The number of iterations of update for the local variational parameters.
        :param kernel: The covariance kernel for the PGPR model.
        :param M: The number of inducing points, if using a sparse model.
        :param init_method: The inducing point initialisation method, if using a sparse model.
        :param reinit_metadata: A dataclass containing training hyperparameters, if using reinitialisation.
        :param optimise_Z: Allow gradient-based optimisation of the inducing inputs, if using a sparse model.
        :return: The final model and best evidence lower bound.
    """
    return _try_train_pgpr(X, Y, inner_iters, opt_iters, ci_iters, kernel, M, init_method, reinit_metadata, optimise_Z)


def _try_train_pgpr(X, Y, inner_iters, opt_iters, ci_iters, kernel, M, init_method, reinit_metadata, optimise_Z):
    """
    Train a PGPR model until completion.
    If we error, keep retrying until success - this is due to a spurious Cholesky error.
    """
    model = PGPR(
        data=(X, Y),
        kernel=kernel,
        inducing_variable=X.copy()
    )
    gpflow.set_trainable(model.inducing_variable, False)

    # Try to run the full optimisation cycle.
    try:
        if M is None or M == len(X):
            return _train_full_pgpr(model, inner_iters, opt_iters, ci_iters)
        else:
            assert init_method is not None, "initialisation_method must not be None, if M < N."
            if reinit_metadata:
                return _train_sparse_reinit_pgpr(model, inner_iters, opt_iters, ci_iters, M, init_method, reinit_metadata, optimise_Z)
            else:
                return _train_sparse_pgpr(model, inner_iters, opt_iters, ci_iters, M, init_method, optimise_Z)
    # If we fail due to a (spurious) Cholesky error, restart.
    except tf.errors.InvalidArgumentError as error:
        msg = error.message
        if "Cholesky" not in msg and "invertible" not in msg:
            raise error
        else:
            print("Cholesky error caught, retrying...")
            return _try_train_pgpr(X, Y, inner_iters, opt_iters, ci_iters, kernel, M, init_method, reinit_metadata, optimise_Z)


def _train_full_pgpr(model, inner_iters, opt_iters, ci_iters):
    """
    Train a non-sparse PGPR model.
    """
    opt = gpflow.optimizers.Scipy()
    for _ in range(inner_iters):
        opt.minimize(model.training_loss, variables=model.trainable_variables, options=dict(maxiter=opt_iters))
        model.optimise_ci(num_iters=ci_iters)

    return model, model.elbo()


def _train_sparse_pgpr(model, inner_iters, opt_iters, ci_iters, M, init_method, optimise_Z):
    """
    Train a sparse PGPR model with a fixed initialisation method.
    For example: uniform_subsample() or kmeans().
    """
    inducing_locs, _ = init_method(model.data[0].numpy(), M)
    inducing_vars = gpflow.inducing_variables.InducingPoints(inducing_locs)
    model.inducing_variable = inducingpoint_wrapper(inducing_vars)
    gpflow.set_trainable(model.inducing_variable, optimise_Z)

    opt = gpflow.optimizers.Scipy()
    for _ in range(inner_iters):
        opt.minimize(model.training_loss, variables=model.trainable_variables, options=dict(maxiter=opt_iters))
        model.optimise_ci(num_iters=ci_iters)

    return model, model.elbo()


def _train_sparse_reinit_pgpr(model, inner_iters, opt_iters, ci_iters, M, reinit_method, reinit_metadata, optimise_Z):
    """
    Train a sparse PGPR model with a given reinitialisation method.
    For example: greedy_variance() or h_greedy_variance().
    """
    opt = gpflow.optimizers.Scipy()
    prev_elbo, elbos = model.elbo(), []
    outer_iters = reinit_metadata.outer_iters

    while True:
        # Reinitialise inducing points
        reinit_method(model, M, reinit_metadata.selection_threshold, optimise_Z)

        # Optimise model
        for _ in range(inner_iters):
            opt.minimize(model.training_loss, variables=model.trainable_variables, options=dict(maxiter=opt_iters))
            model.optimise_ci(num_iters=ci_iters)

        # Check convergence
        next_elbo = model.elbo()
        elbos.append(next_elbo)
        if np.abs(next_elbo - prev_elbo) <= 1e-3:  # if ELBO fails to significantly improve, finish.
            break
        elif outer_iters == 0:  # it is likely that M is too low, and we will not further converge.
            print("PGPR ELBO failed to converge: prev {}, next {}.".format(prev_elbo, next_elbo))
            break
        prev_elbo = next_elbo
        outer_iters -= 1

    return model, np.max(elbos)  # return the highest ELBO
