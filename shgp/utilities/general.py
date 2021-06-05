import gpflow

from tensorflow import sigmoid

"""
General utilities for use throughout the repository.
"""


def invlink(f):
    """
    Transforms a set of GP mean values into pseudo-probabilities in [0, 1].
    Used for classification tasks in order to assign predictive labels from the GP.

    :param f: An array of GP predictive mean values.
    :return: The inverse-linked (squashed) predictive probabilities.
    """
    return gpflow.likelihoods.Bernoulli(invlink=sigmoid).invlink(f).numpy()
