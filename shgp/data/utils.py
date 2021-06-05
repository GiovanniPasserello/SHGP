import numpy as np

"""
General dataset utilities.
"""


def standardise_features(data: np.ndarray):
    """
    Standardise all features to 0 mean and unit variance.

    :param: data - the input data.
    :return: the normalised data.
    """
    data_means = data.mean(axis=0)  # mean value per feature
    data_stds = data.std(axis=0)  # standard deviation per feature

    # standardise each feature
    return (data - data_means) / data_stds


def generate_polynomial_noise_data(N):
    """
    Generate N datapoints with polynomial (heteroscedastic) noise variance.

    :param N: int, the number of datapoints to generate.
    :return:
        np.ndarray, N input locations in [-5, 5].
        np.ndarray, N noisy function values with polynomial noise.
        np.ndarray, N values of the known polynomial noise variance.
    """
    X = np.random.rand(N)[:, None] * 10 - 5  # Inputs, shape N x 1
    X = np.sort(X.flatten()).reshape(N, 1)
    F = 2.5 * np.sin(6 * X) + np.cos(3 * X)  # Mean function values
    NoiseVar = np.abs(0.25 * X**2 + 0.1 * X)  # Quadratic noise variances
    Y = F + np.random.randn(N, 1) * np.sqrt(NoiseVar)  # Noisy data
    return X, Y, NoiseVar
