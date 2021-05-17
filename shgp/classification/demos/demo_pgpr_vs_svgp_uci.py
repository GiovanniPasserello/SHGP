import numpy as np
import tensorflow as tf

np.random.seed(0)
tf.random.set_seed(0)


# TODO: Add experiment contrasting PGPR HGV to PGPR GO initialised at HGV - how much performance are we missing?


def load_fertility():
    # https://archive.ics.uci.edu/ml/datasets/Fertility
    dataset = "../../data/fertility.txt"

    data = np.loadtxt(dataset, delimiter=",")
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1)

    # TODO: This might be a good example to discuss sparsity (9 inducing points PGPR at 1e-6)
    # The comparison here may be that:
    # Yes, we can use 100 inducing points and Bernoulli is almost as quick as PGPR,
    # but why use 100 when we can use 9? A large downside is that if we use 9 points
    # with Bernoulli, it takes about 25 seconds as opposed to 2.38 with PGPR or 2.39
    # with Bernoulli 100 points.

    # TODO: The effect of using a constrained SE kernel slightly affects optimal values.
    # Is there a good way to ensure we are not overly constraining the system?

    NUM_INDUCING = 100  # quicker with 100 than with 9
    BERN_ITERS = 500  # best with 100: -38.982949 (with 9: -38.991571)
    PGPR_ITERS = (5, 25, 5)  # best with 100: -39.354674
    GREEDY_THRESHOLD = 1e-6  # (early stops at 9): -39.354684

    return X, Y, NUM_INDUCING, BERN_ITERS, PGPR_ITERS, GREEDY_THRESHOLD


def load_magic():
    # https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope
    dataset = "../../data/magic.txt"

    data = np.loadtxt(dataset, delimiter=",")
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1)

    NUM_INDUCING = 100  # 200
    BERN_ITERS = 200  # 100
    PGPR_ITERS = (5, 25, 10)  # This slightly benefits from being larger than this, but becomes slow
    GREEDY_THRESHOLD = 0  # 100

    return X, Y, NUM_INDUCING, BERN_ITERS, PGPR_ITERS, GREEDY_THRESHOLD


def load_electricity():
    # https://datahub.io/machine-learning/electricity
    dataset = "../../data/electricity.csv"

    data = np.loadtxt(dataset, delimiter=",", skiprows=1)  # skip headers
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1)

    NUM_INDUCING = 100  # 100
    BERN_ITERS = 200  # best with 100: -20891.910067, accuracy: 0.789217
    PGPR_ITERS = (5, 25, 10)  # best with 100: -21090.760171, accuracy: 0.784936
    GREEDY_THRESHOLD = 0

    # 200 - very close performance
    # bernoulli: ELBO=-20638.401637, ACC=0.795043, NLL=0.441826, TIME=213.12
    # pgpr: ELBO=-20646.108759, ACC=0.791159, NLL=0.444059, TIME=764.12 (but this cycled - converged ~150 seconds)
    # 10 - worse performance
    # bernoulli: ELBO=-21872.580557, ACC=0.778469, NLL=0.467994, TIME=25.71
    # pgpr: ELBO=-22990.888027, ACC=0.755032, NLL=0.505194, TIME=51.28 (this cycled - converged ~15 seconds)

    return X, Y, NUM_INDUCING, BERN_ITERS, PGPR_ITERS, GREEDY_THRESHOLD
