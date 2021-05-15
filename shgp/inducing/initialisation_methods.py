import gpflow
import numpy as np
import scipy.cluster
import tensorflow as tf

from gpflow.models.util import inducingpoint_wrapper
from typing import Callable, Optional

from shgp.models.pgpr import PGPR

# TODO: Add sparsity test: GV vs. HGV vs. (non-grad-optim) US vs. (non-grad-optim) KM.


def uniform_subsample(training_inputs: np.ndarray, M: int):
    """
    Uniformly subsample inducing points from training_inputs.

    Complexity: O(NM) memory, O(NM^2) time
    :param training_inputs: [N,D] np.ndarray, the training data.
    :param M: int, number of inducing points.
    :return: inducing_inputs, indices
    [M,D] np.ndarray containing inducing inputs, [M] np.ndarray containing indices of selected points in training_inputs
    """
    N = training_inputs.shape[0]
    if M is None:
        M = N
    else:
        assert M <= N, 'Cannot set M > N'

    indices = np.random.choice(N, M, replace=False)
    return training_inputs[indices], indices


def k_means(training_inputs: np.ndarray, M: int):
    """
    Initialize inducing points from training_inputs using k-means++.
    Adapted from https://github.com/markvdw/RobustGP/blob/master/robustgp/init_methods/methods.py.

    :param training_inputs: [N,D] np.ndarray, the training data.
    :param M: int, number of inducing points ("k" in k-means).
    :return: inducing_inputs [M,D] np.ndarray containing inducing inputs
    """
    N = training_inputs.shape[0]
    assert M <= N, 'Cannot set M > N'

    # If N is large, take a uniform subset
    if N > 20000:
        training_inputs, _ = uniform_subsample(training_inputs, 20000)

    # Scipy k-means++
    centroids, _ = scipy.cluster.vq.kmeans(training_inputs, M)

    # Sometimes k-means returns fewer than k centroids, in this case we sample remaining point from data
    # These may overlap, but the chances are low.
    if len(centroids) < M:
        num_extra_points = M - len(centroids)
        indices = np.random.choice(N, size=num_extra_points, replace=False)
        additional_points = training_inputs[indices]
        centroids = np.concatenate([centroids, additional_points], axis=0)

    return centroids


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

        # We could either select the next inducing point if the below convergence
        # check fails, or we can accept it now.
        # We might as well take one extra point with the knowledge we have.
        indices[m + 1] = np.argmax(criterion)  # select next point

        # terminate if tr(lambda^-1(Kff-Qff)) is small (implies posterior KL is small)
        # criterion.sum() allows fewer points and works well in larger inducing point constraints (20-100s)
        # di.sum() behaves more sensibly in low inducing point constraints (5-20)
        if np.clip(criterion, 0, None).sum() < threshold:
            index_M = m + 2
            indices = indices[:index_M]
            print("Terminating inducing point selection with {}/{} points.".format(index_M, N))
            break

    # Z, indices of Z in training_inputs
    return training_inputs[indices], perm[indices]


def h_reinitialise_PGPR(model: PGPR, M: int, threshold: Optional[float] = 0.0, optimise_Z: bool = False):
    """
    Reinitialise the inducing points of a PGPR model using h_greedy_variance.

    :param model: PGPR, the model to reinitialise.
    :param M: int, number of inducing points. If threshold is None actual number returned may be less than M.
    :param threshold: convergence threshold at which to stop selection of inducing points early.
    :param optimise_Z: Allow gradient-based optimisation of the inducing inputs.
    """
    X = model.data[0].numpy()
    theta_inv = tf.math.reciprocal(model.likelihood.compute_theta())
    inducing_locs, inducing_idx = h_greedy_variance(X, theta_inv, M, model.kernel, threshold)
    inducing_vars = gpflow.inducing_variables.InducingPoints(inducing_locs)
    model.inducing_variable = inducingpoint_wrapper(inducing_vars)
    gpflow.set_trainable(model.inducing_variable, optimise_Z)
    return inducing_locs, inducing_idx


def greedy_variance(
    training_inputs: np.ndarray,
    M: int,
    kernel: Callable[[np.ndarray, Optional[np.ndarray], Optional[bool]], np.ndarray],
    threshold: Optional[float] = 0.0
):
    """
    Homoscedastic greedy variance inducing point initialisation procedure without noise augmentation.
    This is the original procedure suggested in https://jmlr.org/papers/volume21/19-1015/19-1015.pdf.

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


def reinitialise_PGPR(model: PGPR, M: int, threshold: Optional[float] = 0.0, optimise_Z: bool = False):
    """
    Reinitialise the inducing points of a PGPR model using greedy_variance.

    :param model: PGPR, the model to reinitialise.
    :param M: int, number of inducing points. If threshold is None actual number returned may be less than M.
    :param threshold: convergence threshold at which to stop selection of inducing points early.
    :param optimise_Z: Allow gradient-based optimisation of the inducing inputs.
    """
    X = model.data[0].numpy()
    inducing_locs, inducing_idx = greedy_variance(X, M, model.kernel, threshold)
    inducing_vars = gpflow.inducing_variables.InducingPoints(inducing_locs)
    model.inducing_variable = inducingpoint_wrapper(inducing_vars)
    gpflow.set_trainable(model.inducing_variable, optimise_Z)
    return inducing_locs, inducing_idx
