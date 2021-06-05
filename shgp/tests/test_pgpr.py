import numpy as np
import tensorflow as tf
import gpflow

from gpflow.utilities import to_default_float
from shgp.models.pgpr import PGPR
from shgp.models.wenzel import Wenzel
from shgp.robustness.linalg import robust_cholesky

np.random.seed(42)
tf.random.set_seed(42)

"""
Test suite to check the mathematical correctness of PGPR.
"""


def test_pgpr_qu():
    """
    Test if the closed form q(u) is the same as the predictive distribution of the
    GP when evaluated at the inducing points.
    """

    rng = np.random.RandomState(0)
    X = to_default_float(rng.randn(100, 1))
    Z = to_default_float(rng.randn(20, 1))
    Y = to_default_float(np.random.rand(100, 1).round())

    model = PGPR((X, Y), kernel=gpflow.kernels.SquaredExponential(), inducing_variable=Z)
    gpflow.optimizers.Scipy().minimize(model.training_loss, variables=model.trainable_variables)

    qu_mean, qu_cov = model.compute_qu()
    f_at_Z_mean, f_at_Z_cov = model.predict_f(model.inducing_variable.Z, full_cov=True)

    np.testing.assert_allclose(qu_mean, f_at_Z_mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(tf.reshape(qu_cov, (1, 20, 20)), f_at_Z_cov, rtol=1e-5, atol=1e-5)

    print("Test passed.")


def test_wenzel_converges_to_pgpr():
    """
    Test that the Wenzel ELBO converges to the PGPR ELBO.
    """

    rng = np.random.RandomState(0)
    X = to_default_float(rng.randn(100, 1))
    Z = to_default_float(rng.randn(20, 1))
    Y = to_default_float(np.random.rand(100, 1).round())

    pgpr = PGPR((X, Y), kernel=gpflow.kernels.SquaredExponential(), inducing_variable=Z)
    opt = gpflow.optimizers.Scipy()
    gpflow.set_trainable(pgpr.inducing_variable, False)
    for _ in range(5):
        opt.minimize(pgpr.training_loss, variables=pgpr.trainable_variables, options=dict(maxiter=250))
        pgpr.optimise_ci()

    wenzel = Wenzel((X, Y), kernel=gpflow.kernels.SquaredExponential(), inducing_variable=Z)
    opt = gpflow.optimizers.Scipy()
    gpflow.set_trainable(wenzel.inducing_variable, False)
    for _ in range(5):
        opt.minimize(wenzel.training_loss, variables=wenzel.trainable_variables, options=dict(maxiter=250))
        wenzel.optimise_ci()

    np.testing.assert_allclose(pgpr.elbo(), wenzel.elbo(), rtol=1e-3, atol=1e-3)

    print("Test passed.")


def test_optimal_wenzel_is_same_as_pgpr():
    """
    Test that the Wenzel ELBO is the same as the PGPR ELBO if given the optimal q(u).
    """

    rng = np.random.RandomState(0)
    X = to_default_float(rng.randn(100, 1))
    Z = to_default_float(rng.randn(20, 1))
    Y = to_default_float(np.random.rand(100, 1).round())

    pgpr = PGPR((X, Y), kernel=gpflow.kernels.SquaredExponential(), inducing_variable=Z)
    q_mu, q_var = pgpr.compute_qu()
    q_sqrt = tf.expand_dims(robust_cholesky(q_var), axis=0)

    wenzel = Wenzel(
        (X, Y),
        kernel=gpflow.kernels.SquaredExponential(),
        inducing_variable=Z,
        q_mu=q_mu,
        q_sqrt=q_sqrt
    )

    np.testing.assert_allclose(pgpr.elbo(), wenzel.elbo(), rtol=1e-3, atol=1e-3)

    print("Test passed.")


if __name__ == '__main__':
    test_pgpr_qu()
    test_wenzel_converges_to_pgpr()
    test_optimal_wenzel_is_same_as_pgpr()
