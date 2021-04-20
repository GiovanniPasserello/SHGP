import gpflow
import numpy as np
import tensorflow as tf
import warnings

from gpflow.models.util import inducingpoint_wrapper
from typing import Callable, Optional

from shgp.models.pgpr import PGPR


def uniform_subsample(training_inputs: np.ndarray, M: int):
    """
    Uniformly subsample inducing points from the training_inputs.

    Complexity: O(NM) memory, O(NM^2) time
    :param training_inputs: [N,D] np.ndarray, the training data.
    :param M: int, number of inducing points. If threshold is None actual number returned may be less than M.
    :return: inducing inputs, indices
    [M,D] np.ndarray containing inducing inputs, [M] np.ndarray containing indices of selected points in training_inputs
    """

    N = training_inputs.shape[0]
    if M is None:
        M = N

    assert M <= N, 'Cannot set M > N'

    indices = np.random.choice(N, M, replace=False)
    return training_inputs[indices], indices


def h_greedy_variance(
    training_inputs: np.ndarray,
    lmbda: np.ndarray,
    M: int,
    kernel: Callable[[np.ndarray, Optional[np.ndarray], Optional[bool]], np.ndarray],
    threshold: Optional[float] = 0.0
):
    """
    Heteroscedastic implementation of the greedy variance inducing point initialisation
    procedure suggested in https://jmlr.org/papers/volume21/19-1015/19-1015.pdf.
    Adapted from https://github.com/markvdw/RobustGP/blob/master/robustgp/init_methods/methods.py.

    Complexity: O(NM) memory, O(NM^2) time
    :param training_inputs: [N,D] np.ndarray, the training data.
    :param lmbda: [N] np.ndarray, diagonal of the likelihood covariance.
    :param M: int, number of inducing points. If threshold is None actual number returned may be less than M.
    :param kernel: kernelwrapper object.
    :param threshold: convergence threshold at which to stop selection of inducing points early.
    :return: inducing inputs, indices,
    [M,D] np.ndarray containing inducing inputs, [M] np.ndarray containing indices of selected points in training_inputs
    """

    N = training_inputs.shape[0]
    assert M <= N, 'Cannot set M > N'

    perm = np.random.permutation(N)  # permute entries so tie-breaking is random
    training_inputs = training_inputs[perm]
    lmbda = tf.gather(lmbda, perm)
    lmbda_inv = tf.squeeze(tf.math.reciprocal(lmbda))

    # Take inducing first point
    indices = np.zeros(M, dtype=int)
    di = kernel(training_inputs, None, full_cov=False) + 1e-12  # Kff + jitter
    # Nystrom difference from di, heteroscedasticity from lmbda_inv
    indices[0] = np.argmax(lmbda_inv * di)  # select first point

    # Rank-one update cycle to avoid repetitive inversions
    ci = np.zeros((M - 1, N))  # [M,N]
    for m in range(M - 1):
        j = indices[m]
        new_Z = training_inputs[j]  # [1,D]
        dj = tf.sqrt(di[j])
        cj = ci[:m, j]  # [M,1]
        Lraw = np.array(kernel(training_inputs, new_Z, full_cov=True))  # Kfu rank-one
        L = Lraw.squeeze().round(20)  # [N]
        L[j] += 1e-12  # jitter
        ei = (L - cj.dot(ci[:m])) / dj
        ci[m, :] = ei
        try:
            di -= ei ** 2
        except FloatingPointError:
            pass
        di = np.clip(di, 0, None)

        # Nystrom difference from di, heteroscedasticity from lmbda_inv
        criterion = lmbda_inv * di
        indices[m + 1] = np.argmax(criterion)  # select next point

        # terminate if tr(lambda^-1(Kff-Qff)) is small (implies posterior KL is small)
        # criterion.sum() allows fewer points and works well in larger inducing point constraints (20-100s)
        # di.sum() behaves more sensibly in low inducing point constraints (5-20)
        if np.clip(criterion, 0, None).sum() < threshold:
            indices = indices[:m]
            warnings.warn("ConditionalVariance: Terminating selection of inducing points early.")
            break

    # Z, indices of Z in training_inputs
    return training_inputs[indices], perm[indices]


def h_reinitialise_PGPR(
    model: PGPR,
    training_inputs: np.ndarray,
    M: int,
    threshold: Optional[float] = 0.0
):
    """
    Reinitialise inducing points of PGPR model using h_greedy_variance.

    :param model: PGPR, the model to reinitialise.
    :param training_inputs: [N,D] np.ndarray, the training data.
    :param M: int, number of inducing points. If threshold is None actual number returned may be less than M.
    :param threshold: convergence threshold at which to stop selection of inducing points early.
    """
    theta_inv = tf.math.reciprocal(model.likelihood.compute_theta())
    inducing_locs, inducing_idx = h_greedy_variance(training_inputs, theta_inv, M, model.kernel, threshold)
    inducing_vars = gpflow.inducing_variables.InducingPoints(inducing_locs)
    model.inducing_variable = inducingpoint_wrapper(inducing_vars)
    gpflow.set_trainable(model.inducing_variable, False)
    return inducing_locs, inducing_idx


def greedy_variance(
    training_inputs: np.ndarray,
    M: int,
    kernel: Callable[[np.ndarray, Optional[np.ndarray], Optional[bool]], np.ndarray],
    threshold: Optional[float] = 0.0
):
    """
    Homoscedastic greedy variance inducing point initialisation procedure without noise augmentation.

    Complexity: O(NM) memory, O(NM^2) time
    :param training_inputs: [N,D] np.ndarray, the training data.
    :param M: int, number of inducing points. If threshold is None actual number returned may be less than M.
    :param kernel: kernelwrapper object
    :param threshold: convergence threshold at which to stop selection of inducing points early.
    :return: inducing inputs, indices,
    [M,D] np.ndarray containing inducing inputs, [M] np.ndarray containing indices of selected points in training_inputs
    """
    lmbda = np.ones(len(training_inputs))
    return h_greedy_variance(training_inputs, lmbda, M, kernel, threshold)


def reinitialise_PGPR(
    model: PGPR,
    training_inputs: np.ndarray,
    M: int,
    threshold: Optional[float] = 0.0
):
    """
    Reinitialise inducing points of PGPR model using greedy_variance.

    :param model: PGPR, the model to reinitialise.
    :param training_inputs: [N,D] np.ndarray, the training data.
    :param M: int, number of inducing points. If threshold is None actual number returned may be less than M.
    :param threshold: convergence threshold at which to stop selection of inducing points early.
    """
    inducing_locs, inducing_idx = greedy_variance(training_inputs, M, model.kernel, threshold)
    inducing_vars = gpflow.inducing_variables.InducingPoints(inducing_locs)
    model.inducing_variable = inducingpoint_wrapper(inducing_vars)
    gpflow.set_trainable(model.inducing_variable, False)
    return inducing_locs, inducing_idx


def greedy_bound_increase(
    training_inputs: np.ndarray,
    lmbda: np.ndarray,
    M: int,
    kernel: Callable[[np.ndarray, Optional[np.ndarray], Optional[bool]], np.ndarray]
):
    """
    Heteroscedastic implementation of an O(MN^2) greedy ELBO inducing point initialisation.
    This chooses the inducing point that maximises a trace term from the rank-1 difference
    of the ELBO before and after adding that inducing points.
    Note that due to being O(MN^2) this is not a preferable method, it is simply for comparisons,
    however it does generally produce superior inducing point locations which yield a higher ELBO.

    Complexity: O(N^2) memory, O(MN^2) time
    :param training_inputs: [N,D] np.ndarray, the training data.
    :param M: int, number of inducing points. If threshold is None actual number returned may be less than M.
    :param kernel: kernelwrapper object.
    :return: inducing inputs, indices,
    [M,D] np.ndarray containing inducing inputs, [M] np.ndarray containing indices of selected points in training_inputs
    """

    N = training_inputs.shape[0]
    assert M <= N, 'Cannot set M > N'

    perm = np.random.permutation(N)  # permute entries so tie-breaking is random
    training_inputs = training_inputs[perm]
    lmbda = tf.gather(lmbda, perm)
    Kff = tf.reshape(kernel(training_inputs, full_cov=False), (-1, 1))

    indices = np.zeros(M, dtype=int)
    selected = np.argmax(lmbda * Kff)
    indices[0] = selected  # select first point, add to index 0
    Z = np.array([training_inputs[selected]])

    for m in range(1, M):
        kuf = kernel(Z, training_inputs)
        kuu = kernel(Z, Z) + 1e-12  # jitter
        L = tf.linalg.cholesky(kuu)

        ku = kernel(Z, training_inputs)
        L_ku = tf.linalg.triangular_solve(L, ku)
        c = tf.squeeze(Kff) - np.einsum('ij,ij->j', L_ku, L_ku)
        L_kuf = tf.linalg.triangular_solve(L, kuf)
        kf = kernel(training_inputs, training_inputs)  # computes full outer product -> expensive!
        b = tf.matmul(L_kuf, L_ku, transpose_a=True) - kf
        scores = (tf.reduce_sum(lmbda * tf.math.square(b), axis=0) / c).numpy()

        # doesn't account for duplicate inputs (can remove in preprocessing)
        scores[indices[:m]] = float("-inf")

        # need max across points
        selected = np.argmax(scores)
        indices[m] = selected
        Z = np.append(Z, [training_inputs[selected]], axis=0)

    indices = indices.astype(int)
    Z = training_inputs[indices]
    indices = perm[indices]
    return Z, indices


def bound_max_reinitialise_PGPR(
    model: PGPR,
    training_inputs: np.ndarray,
    M: int
):
    """
    Reinitialise inducing points of PGPR model using greedy_bound_increase.

    :param model: PGPR, the model to reinitialise.
    :param training_inputs: [N,D] np.ndarray, the training data.
    :param M: int, number of inducing points. If threshold is None actual number returned may be less than M.
    """
    theta_inv = tf.math.reciprocal(model.likelihood.compute_theta())
    inducing_locs, inducing_idx = greedy_bound_increase(training_inputs, theta_inv, M, model.kernel)
    inducing_vars = gpflow.inducing_variables.InducingPoints(inducing_locs)
    model.inducing_variable = inducingpoint_wrapper(inducing_vars)
    gpflow.set_trainable(model.inducing_variable, False)
    return inducing_locs, inducing_idx
