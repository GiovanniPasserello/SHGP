import warnings
from typing import Callable, Optional

import numpy as np
import tensorflow as tf


def h_greedy_variance(
    training_inputs: np.ndarray,
    lmbda: np.ndarray,
    M: int,
    kernel: Callable[[np.ndarray, Optional[np.ndarray], Optional[bool]], np.ndarray],
    threshold: Optional[int] = 0.0
):
    """
    Heteroscedastic implementation of the greedy variance inducing point initialisation
    procedure suggested in https://jmlr.org/papers/volume21/19-1015/19-1015.pdf.
    Adapted from https://github.com/markvdw/RobustGP/blob/master/robustgp/init_methods/methods.py.

    :param training_inputs: [N,D] np.ndarray, the training data.
    :param lmbda: [N] np.ndarray, diagonal of the likelihood covariance.
    :param M: int, number of inducing points. If threshold is None actual number returned may be less than M.
    :param kernel: kernelwrapper object
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
        indices[m + 1] = np.argmax(lmbda_inv * di)  # select next point

        # sum of di is tr(lambda^-1(Kff-Qff)), if this is small things are ok
        if np.clip(di, 0, None).sum() < threshold:
            indices = indices[:m]
            warnings.warn("ConditionalVariance: Terminating selection of inducing points early.")
            break

    # Z, indices of Z in training_inputs
    return training_inputs[indices], perm[indices]
