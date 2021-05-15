from dataclasses import dataclass
from typing import List

import numpy as np

from shgp.utilities.general import invlink


@dataclass
class ExperimentResult:
    """
        A dataclass to store a single set of metric results from an experiment iteration.

        :param elbo: The variational lower bound on the train set (lower bound to log marginal likelihood).
        :param accuracy: The accuracy on a held-out test set.
        :param nll: The negative log likelihood on a held-out test set.
    """
    elbo: float
    accuracy: float
    nll: float

    def __le__(self, other):
        return self.elbo <= other.elbo


class ExperimentResults:
    """
        A dataclass to store a collection of metric results over multiple experiment iterations.

        :param results: The collection of metric results.
    """
    def __init__(self):
        self.results: List[ExperimentResult] = []

    def add_result(self, result: ExperimentResult):
        self.results.append(result)

    def _extract_attributes(self):
        elbos = [r.elbo for r in self.results]
        accs = [r.accuracy for r in self.results]
        nlls = [r.nll for r in self.results]
        return elbos, accs, nlls

    def compute_distribution(self):
        elbos, accs, nlls = self._extract_attributes()
        maximum = ExperimentResult(np.max(elbos), np.max(accs), np.max(nlls))
        minimum = ExperimentResult(np.min(elbos), np.min(accs), np.min(nlls))
        median = ExperimentResult(np.median(elbos), np.median(accs), np.median(nlls))
        mean = ExperimentResult(np.mean(elbos), np.mean(accs), np.mean(nlls))
        std = ExperimentResult(np.std(elbos), np.std(accs), np.std(nlls))
        return maximum, minimum, median, mean, std


# Predictive Accuracy
def compute_accuracy(Y, F):
    preds = np.round(invlink(F))
    return np.sum(Y == preds) / len(Y)


# Average NLL
def compute_nll(Y, F):
    P = invlink(F)
    return -np.log(np.where(Y, P, 1 - P)).mean()


# Compute metrics on a held-out test set
def compute_test_metrics(model, X_test, Y_test):
    F, _ = model.predict_f(X_test)
    return compute_accuracy(Y_test, F), compute_nll(Y_test, F)
