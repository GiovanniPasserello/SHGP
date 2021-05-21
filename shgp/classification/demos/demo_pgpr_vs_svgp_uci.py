import numpy as np
import tensorflow as tf

np.random.seed(0)
tf.random.set_seed(0)


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
