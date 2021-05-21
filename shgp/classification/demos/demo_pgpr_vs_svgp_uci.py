import numpy as np
import tensorflow as tf

np.random.seed(0)
tf.random.set_seed(0)


# TODO: Run fertility metric experiment
def load_fertility():
    # https://archive.ics.uci.edu/ml/datasets/Fertility
    dataset = "../../data/fertility.txt"

    data = np.loadtxt(dataset, delimiter=",")
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1)

    NUM_INDUCING = 100  # quicker with 100 than with 9
    BERN_ITERS = 500  # best with 100: -38.982949 (with 9: -38.991571)
    PGPR_ITERS = (5, 25, 5)  # best with 100: -39.354674
    GREEDY_THRESHOLD = 1e-6  # (early stops at 9): -39.354684

    return X, Y, NUM_INDUCING, BERN_ITERS, PGPR_ITERS, GREEDY_THRESHOLD


# TODO: Try fix memory errors (maybe I can run PGPR, but not svgp?)
#       This is why it won't fit in memory???
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

    return X, Y, NUM_INDUCING, BERN_ITERS, PGPR_ITERS, GREEDY_THRESHOLD
