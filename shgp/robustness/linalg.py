import numpy as np
import tensorflow as tf

from gpflow.utilities import to_default_float

# TODO: Doesn't work without eager execution in Tensorflow 2.4.
#       Once Tensorflow 2.5 is released, this can be removed.
tf.config.run_functions_eagerly(True)


# From TF experimental
def robust_cholesky(matrix):
    """
    Attempts a standard Cholesky decomposition, and then tries with increasing jitter upon failure.
    After a maximum jitter is attempted, raises an exception.
    # TF experimental comments:
    Note this will NOT work under a gradient tape until b/177365178 is resolved.
    Also this uses XLA compilation, which is necessary until b/144845034 is resolved.
    :param matrix: The matrix to attempt a Cholesky decomposition on.
    :returns: The lower triangular Cholesky matrix, unless an exception was raised.
    """

    jitter = to_default_float(np.logspace(-9, 0, 10))

    def attempt_cholesky(mat):
        """Return a Cholesky factor and boolean success."""
        try:
            cholesky = tf.linalg.cholesky(mat)
            return cholesky, tf.reduce_all(tf.math.is_finite(cholesky))
        except tf.errors.InvalidArgumentError:
            return mat, False

    # TODO: Once Tensorflow 2.5 is released, this can be removed.
    attempt_cholesky = tf.function(attempt_cholesky, autograph=False)  # , jit_compile=True)

    # Run a backoff algorithm with dynamic jitter increase.
    # In the first iteration no jitter is added.
    # Subsequent iterations try jitter increasing on a logarithmic scale.
    def _run_backoff(mat, index=0):
        if index >= len(jitter):
            message = 'Cholesky failed with maximum jitter.'
            print(message)
            raise tf.errors.InvalidArgumentError(
                node_def=None,
                message=message,
                op=None
            )

        cholesky, ok = attempt_cholesky(mat)
        return tf.cond(
            ok,
            lambda: cholesky,
            lambda: _run_backoff(
                matrix + to_default_float(tf.eye(mat.shape[0])) * jitter[index],
                index + 1
            )
        )

    return _run_backoff(matrix)
