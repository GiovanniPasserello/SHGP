from typing import Callable, Optional, Type

import gpflow
import numpy as np
import tensorflow as tf

from gpflow.base import Parameter
from gpflow.config import default_float
from gpflow.models.util import inducingpoint_wrapper
from gpflow.kernels import Kernel
from gpflow.utilities import triangular

from shgp.data.metadata_reinit import ReinitMetaDataset
from shgp.models.pgpr import PGPR
from shgp.inducing.initialisation_methods import greedy_variance
from shgp.utilities.metrics import compute_test_metrics, ExperimentResults, ExperimentResult
from shgp.utilities.train_pgpr import result, train_pgpr
from shgp.utilities.train_svgp import train_svgp


def train_pgpr_svgp(
    X: np.ndarray,
    Y: np.ndarray,
    opt_iters: int,
    *,
    kernel_type: Type[Kernel] = gpflow.kernels.SquaredExponential,
    M: Optional[int] = None,
    init_method: Optional[Callable] = None,
    reinit_metadata: Optional[ReinitMetaDataset] = None,
    optimise_Z: bool = False,
    X_test: Optional[np.ndarray] = None,
    Y_test: Optional[np.ndarray] = None
):
    """
        Train a PGPR-SVGP model with the following parameters.
        This is the only function in this file that should be externally called.

        :param X: [N,D], the training feature data.
        :param Y: [N,1], the training label data.
        :param opt_iters: The number of iterations of gradient-based optimisation of the kernel hyperparameters.
        :param kernel_type: The covariance kernel type for the PGPR-SVGP model. We use a type, instead of an object
                            so that we can reinitialise in the case of an error.
        :param M: The number of inducing points, if using a sparse model.
        :param init_method: The inducing point initialisation method, if using a sparse model.
        :param reinit_metadata: A dataclass containing training hyperparameters, if using reinitialisation.
        :param optimise_Z: Allow gradient-based optimisation of the inducing inputs, if using a sparse model.
        :param X_test: [N_test,D], the test feature data.
        :param Y_test: [N_test,1], the test label data.
        :return: The final model and best evidence lower bound (or full set of metrics).
    """
    return _try_train_pgpr_svgp(X, Y, M, opt_iters, kernel_type, init_method, reinit_metadata, optimise_Z, X_test, Y_test)


def _try_train_pgpr_svgp(X, Y, M, opt_iters, kernel_type, init_method, reinit_metadata, optimise_Z, X_test, Y_test):
    """
    Train a PGPR model until completion.
    If we error, keep retrying until success - this is due to a spurious Cholesky error.
    """
    # Try to run the full optimisation cycle.
    try:
        if M is None or M == len(X):
            return train_svgp(X, Y, M, train_iters=opt_iters, kernel_type=kernel_type, init_method=init_method, optimise_Z=optimise_Z, X_test=X_test, Y_test=Y_test)
        else:
            assert init_method is not None, "initialisation_method must not be None, if M < N."
            return _train_sparse_pgpr_svgp(X, Y, M, opt_iters, kernel_type, init_method, reinit_metadata, optimise_Z, X_test, Y_test)
    # If we fail due to a (spurious) Cholesky error, restart.
    except tf.errors.InvalidArgumentError as error:
        msg = error.message
        if "Cholesky" not in msg and "invertible" not in msg:
            raise error
        else:
            if "Cholesky" in msg:
                print("Cholesky error caught, retrying...")
            else:
                print("Inversion error caught, retrying...")
            return _try_train_pgpr_svgp(X, Y, M, opt_iters, kernel_type, init_method, reinit_metadata, optimise_Z, X_test, Y_test)


# def _train_sparse_pgpr_svgp(X, Y, M, opt_iters, kernel_type, reinit_method, reinit_metadata, optimise_Z, X_test, Y_test):
#     """
#     Train a sparse PGPR-SVGP model with a given reinitialisation method.
#     For example: greedy_variance() or h_greedy_variance().
#     """
#
#     # Train PGPR till convergence
#     pgpr, _ = train_pgpr(
#         X, Y,
#         10, opt_iters, 10,  # TODO: Add proper parameters
#         kernel_type=kernel_type,
#         M=M,
#         init_method=reinit_method,
#         reinit_metadata=reinit_metadata,
#         optimise_Z=optimise_Z,
#         X_test=X_test,
#         Y_test=Y_test
#     )
#     # TODO: Training SVGP after this, with pgpr.inducing_variable
#     #       gives worse results than the previously trained PGPR.
#     # _, res = result(pgpr, pgpr.elbo(), X_test, Y_test)
#     # print("here", res.elbo, res.accuracy, res.nll)
#
#     # Train SVGP till convergence
#     svgp = gpflow.models.SVGP(
#         kernel=kernel_type(),
#         likelihood=gpflow.likelihoods.Bernoulli(invlink=tf.sigmoid),
#         inducing_variable=pgpr.inducing_variable
#     )
#     gpflow.set_trainable(svgp.inducing_variable, optimise_Z)
#     gpflow.optimizers.Scipy().minimize(
#         svgp.training_loss_closure((X, Y)),
#         variables=svgp.trainable_variables,
#         options=dict(maxiter=opt_iters)  # TODO: Separate SVGP opt iters
#     )
#
#     return result(svgp, svgp.elbo((X, Y)), X_test, Y_test)


# TODO: Interleaved training (good, but slow and *very* non-monotonic unreliable)
# def _train_sparse_pgpr_svgp(X, Y, M, opt_iters, kernel_type, reinit_method, reinit_metadata, optimise_Z, X_test, Y_test):
#     """
#     Train a sparse PGPR-SVGP model with a given reinitialisation method.
#     For example: greedy_variance() or h_greedy_variance().
#     """
#     return_metrics = X_test is not None
#     if return_metrics:  # track ELBO, ACC, NLL
#         results = ExperimentResults()
#
#     # Initialise PGPR
#     pgpr = PGPR(data=(X, Y), kernel=kernel_type())
#     reinit_method(pgpr, M, reinit_metadata.selection_threshold)
#     q_mu, q_var = pgpr.compute_qu()
#     q_sqrt = tf.expand_dims(tf.linalg.cholesky(q_var), axis=0)
#
#     # Initialise SVGP
#     pgpr_svgp = gpflow.models.SVGP(
#         kernel=kernel_type(),
#         likelihood=gpflow.likelihoods.Bernoulli(tf.sigmoid),
#         inducing_variable=pgpr.inducing_variable,
#         q_mu=q_mu,
#         q_sqrt=q_sqrt
#     )
#     gpflow.set_trainable(pgpr_svgp.inducing_variable, optimise_Z)
#
#     opt = gpflow.optimizers.Scipy()
#     outer_iters = reinit_metadata.outer_iters
#     prev_elbo, elbos = pgpr_svgp.elbo((X, Y)), []
#     while True:
#         # Optimise SVGP
#         opt.minimize(
#             pgpr_svgp.training_loss_closure((X, Y)),
#             variables=pgpr_svgp.trainable_variables,
#             options=dict(maxiter=opt_iters)
#         )
#
#         # Evaluate metrics
#         next_elbo = pgpr_svgp.elbo((X, Y))
#         #print("PGPR-SVGP ELBO:", next_elbo)
#         elbos.append(next_elbo)
#         if return_metrics:  # track ELBO, ACC, NLL
#             results.add_result(ExperimentResult(next_elbo, *compute_test_metrics(pgpr_svgp, X_test, Y_test)))
#
#         # Check convergence
#         outer_iters -= 1
#         if np.abs(next_elbo - prev_elbo) <= reinit_metadata.conv_threshold:  # if ELBO fails to significantly improve.
#             break
#         elif outer_iters == 0:  # it is likely that M is too low, and we will not further converge.
#             if reinit_metadata.conv_threshold > 0:
#                 print("PGPR-SVGP ELBO failed to converge: prev {}, next {}.".format(prev_elbo, next_elbo))
#             break
#         prev_elbo = next_elbo
#
#         # Update PGPR
#         f_mu, f_var = pgpr_svgp.predict_f(X)
#         pgpr.likelihood.c_i = tf.math.sqrt(tf.math.square(f_mu) + f_var)
#
#         # Reinitialised Z
#         pgpr.kernel = pgpr_svgp.kernel
#         reinit_method(pgpr, M, reinit_metadata.selection_threshold)
#
#         # Compute optimal q_mu, q_sqrt
#         q_mu, q_var = pgpr.compute_qu()
#         q_sqrt = tf.expand_dims(tf.linalg.cholesky(q_var), axis=0)
#
#         # Restart SVGP with optimal q_mu, q_sqrt and reinitialised Z from PGPR
#         pgpr_svgp = gpflow.models.SVGP(
#             kernel=kernel_type(),
#             likelihood=gpflow.likelihoods.Bernoulli(tf.sigmoid),
#             inducing_variable=pgpr.inducing_variable,
#             q_mu=q_mu,
#             q_sqrt=q_sqrt
#         )
#         gpflow.set_trainable(pgpr_svgp.inducing_variable, optimise_Z)
#
#     if return_metrics:
#         return pgpr_svgp, np.max(results.results)   # return the metrics with the highest ELBO
#     else:
#         return pgpr_svgp, np.max(elbos)   # return the highest ELBO


# TODO: With greedy variance reinit (no PGPR)
def _train_sparse_pgpr_svgp(X, Y, M, opt_iters, kernel_type, reinit_method, reinit_metadata, optimise_Z, X_test, Y_test):
    """
    Train a sparse PGPR-SVGP model with a given reinitialisation method.
    For example: greedy_variance() or h_greedy_variance().
    """
    return_metrics = X_test is not None
    if return_metrics:  # track ELBO, ACC, NLL
        results = ExperimentResults()

    # Initialise SVGP
    svgp = gpflow.models.SVGP(
        kernel=kernel_type(),
        likelihood=gpflow.likelihoods.Bernoulli(tf.sigmoid),
        inducing_variable=X.copy()
    )

    opt = gpflow.optimizers.Scipy()
    outer_iters = reinit_metadata.outer_iters
    prev_elbo, elbos = -float("inf"), []
    while True:
        # Reinitialise Z
        inducing_locs, _ = greedy_variance(X, M, svgp.kernel)
        # Create new model as M may be different, so q_mu & q_var need reinitialising
        svgp = gpflow.models.SVGP(
            kernel=kernel_type(),
            likelihood=gpflow.likelihoods.Bernoulli(tf.sigmoid),
            inducing_variable=inducingpoint_wrapper(inducing_locs)
        )
        gpflow.set_trainable(svgp.inducing_variable, optimise_Z)

        # Optimise SVGP
        opt.minimize(
            svgp.training_loss_closure((X, Y)),
            variables=svgp.trainable_variables,
            options=dict(maxiter=opt_iters)
        )

        # Evaluate metrics
        next_elbo = svgp.elbo((X, Y))
        print("PGPR-SVGP ELBO:", next_elbo)
        elbos.append(next_elbo)
        if return_metrics:  # track ELBO, ACC, NLL
            results.add_result(ExperimentResult(next_elbo, *compute_test_metrics(svgp, X_test, Y_test)))

        # Check convergence
        outer_iters -= 1
        if np.abs(next_elbo - prev_elbo) <= reinit_metadata.conv_threshold:  # if ELBO fails to significantly improve.
            break
        elif outer_iters == 0:  # it is likely that M is too low, and we will not further converge.
            if reinit_metadata.conv_threshold > 0:
                print("PGPR-SVGP ELBO failed to converge: prev {}, next {}.".format(prev_elbo, next_elbo))
            break
        prev_elbo = next_elbo

    if return_metrics:
        return svgp, np.max(results.results)   # return the metrics with the highest ELBO
    else:
        return svgp, np.max(elbos)   # return the highest ELBO
