import gpflow

from tensorflow import sigmoid


# Logistic sigmoid inverse link
def invlink(f):
    return gpflow.likelihoods.Bernoulli(invlink=sigmoid).invlink(f).numpy()
