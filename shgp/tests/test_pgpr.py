from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import gpflow

from gpflow.utilities import to_default_float
from shgp.models.pgpr import PGPR


def test_hgpr_qu():
    rng = np.random.RandomState(0)
    X = to_default_float(rng.randn(100, 1))
    Z = to_default_float(rng.randn(20, 1))
    Y = to_default_float(np.sin(X * -1.4) + 0.5 * rng.randn(len(X), 1))

    model = PGPR((X, Y), kernel=gpflow.kernels.SquaredExponential(), inducing_variable=Z)
    gpflow.optimizers.Scipy().minimize(model.training_loss, variables=model.trainable_variables)

    qu_mean, qu_cov = model.compute_qu()
    f_at_Z_mean, f_at_Z_cov = model.predict_f(model.inducing_variable.Z, full_cov=True)

    np.testing.assert_allclose(qu_mean, f_at_Z_mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(tf.reshape(qu_cov, (1, 20, 20)), f_at_Z_cov, rtol=1e-5, atol=1e-5)

    print("Test passed.")


test_hgpr_qu()
