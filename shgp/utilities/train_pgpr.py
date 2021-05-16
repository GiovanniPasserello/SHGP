from typing import Callable, Optional, Type

import gpflow
import numpy as np
import tensorflow as tf

from gpflow.models.util import inducingpoint_wrapper
from gpflow.kernels import Kernel

from shgp.data.metadata_reinit import ReinitMetaDataset
from shgp.robustness.contrained_kernels import ConstrainedExpSEKernel
from shgp.models.pgpr import PGPR
from shgp.utilities.metrics import compute_test_metrics, ExperimentResults, ExperimentResult


def train_pgpr(
    X: np.ndarray,
    Y: np.ndarray,
    inner_iters: int,
    opt_iters: int,
    ci_iters: int,
    *,
    kernel_type: Type[Kernel] = ConstrainedExpSEKernel,  # increased stability, with minor performance detriment.
    M: Optional[int] = None,
    init_method: Optional[Callable] = None,
    reinit_metadata: Optional[ReinitMetaDataset] = None,
    optimise_Z: bool = False,
    X_test: Optional[np.ndarray] = None,
    Y_test: Optional[np.ndarray] = None
):
    """
        Train a PGPR model with the following parameters.
        This is the only function in this file that should be externally called.

        :param X: [N,D], the training feature data.
        :param Y: [N,1], the training label data.
        :param inner_iters: The number of iterations of the inner optimisation loop.
        :param opt_iters: The number of iterations of gradient-based optimisation of the kernel hyperparameters.
        :param ci_iters: The number of iterations of update for the local variational parameters.
        :param kernel_type: The covariance kernel type for the PGPR model. We use a type, instead of an object
                            so that we can reinitialise in the case of an error.
        :param M: The number of inducing points, if using a sparse model.
        :param init_method: The inducing point initialisation method, if using a sparse model.
        :param reinit_metadata: A dataclass containing training hyperparameters, if using reinitialisation.
        :param optimise_Z: Allow gradient-based optimisation of the inducing inputs, if using a sparse model.
        :param X_test: [N_test,D], the test feature data.
        :param Y_test: [N_test,1], the test label data.
        :return: The final model and best evidence lower bound (or full set of metrics).
    """
    return _try_train_pgpr(X, Y, inner_iters, opt_iters, ci_iters, kernel_type, M, init_method, reinit_metadata, optimise_Z, X_test, Y_test)


def _try_train_pgpr(X, Y, inner_iters, opt_iters, ci_iters, kernel_type, M, init_method, reinit_metadata, optimise_Z, X_test, Y_test):
    """
    Train a PGPR model until completion.
    If we error, keep retrying until success - this is due to a spurious Cholesky error.
    """
    model = PGPR(
        data=(X, Y),
        kernel=kernel_type(),
        inducing_variable=X.copy()
    )
    gpflow.set_trainable(model.inducing_variable, False)

    # Try to run the full optimisation cycle.
    try:
        if M is None or M == len(X):
            return result(*_train_full_pgpr(model, inner_iters, opt_iters, ci_iters), X_test, Y_test)
        else:
            assert init_method is not None, "initialisation_method must not be None, if M < N."
            if reinit_metadata:
                return _train_sparse_reinit_pgpr(model, inner_iters, opt_iters, ci_iters, M, init_method, reinit_metadata, optimise_Z, X_test, Y_test)
            else:
                return result(*_train_sparse_pgpr(model, inner_iters, opt_iters, ci_iters, M, init_method, optimise_Z), X_test, Y_test)
    # If we fail due to a (spurious) Cholesky error, restart.
    except tf.errors.InvalidArgumentError as error:
        msg = error.message
        if "Cholesky" not in msg and "invertible" not in msg:
            raise error
        else:
            print("Cholesky error caught, retrying...")
            return _try_train_pgpr(X, Y, inner_iters, opt_iters, ci_iters, kernel_type, M, init_method, reinit_metadata, optimise_Z, X_test, Y_test)


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


def _train_sparse_reinit_pgpr(model, inner_iters, opt_iters, ci_iters, M, reinit_method, reinit_metadata, optimise_Z, X_test, Y_test):
    """
    Train a sparse PGPR model with a given reinitialisation method.
    For example: greedy_variance() or h_greedy_variance().
    """
    opt = gpflow.optimizers.Scipy()
    prev_elbo, elbos = model.elbo(), []
    outer_iters = reinit_metadata.outer_iters

    return_metrics = X_test is not None
    if return_metrics:  # track ELBO, ACC, NLL
        results = ExperimentResults()

    while True:
        # Reinitialise inducing points
        reinit_method(model, M, reinit_metadata.selection_threshold, optimise_Z)

        # Optimise model
        for _ in range(inner_iters):
            opt.minimize(model.training_loss, variables=model.trainable_variables, options=dict(maxiter=opt_iters))
            model.optimise_ci(num_iters=ci_iters)

        # Evaluate metrics
        next_elbo = model.elbo()
        elbos.append(next_elbo)
        if return_metrics:  # track ELBO, ACC, NLL
            results.add_result(ExperimentResult(next_elbo, *compute_test_metrics(model, X_test, Y_test)))

        # Check convergence
        if np.abs(next_elbo - prev_elbo) <= reinit_metadata.conv_threshold:  # if ELBO fails to significantly improve.
            break
        elif outer_iters == 0:  # it is likely that M is too low, and we will not further converge.
            if reinit_metadata.conv_threshold > 0:
                print("PGPR ELBO failed to converge: prev {}, next {}.".format(prev_elbo, next_elbo))
            break
        prev_elbo = next_elbo
        outer_iters -= 1

    if return_metrics:
        return model, np.max(results.results)   # return the metrics with the highest ELBO
    else:
        return model, np.max(elbos)   # return the highest ELBO


def result(model, elbo, X_test, Y_test):
    """
    If a test set is not provided, return the model and the elbo.
    If a test set is provided, return the model and a set of test metrics.
    """
    if X_test is None:
        return model, elbo
    else:
        return model, ExperimentResult(elbo, *compute_test_metrics(model, X_test, Y_test))
