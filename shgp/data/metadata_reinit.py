from dataclasses import dataclass


@dataclass
class ReinitMetaDataset:
    """
    A dataset utilities class for reinitialisation training hyperparameters.

    :param outer_iters: The maximum number of times to attempt reinitialisation (i.e., the outer loop).
    :param selection_threshold: The threshold on the trace term which tells us when we have enough inducing points.
    :param conv_threshold: The threshold on the improvement of the ELBO to check for convergence.
        Having this threshold is not perfect and can cause the model to stop early.
        Sometimes setting this to zero and running the total number of outer_iters is more reliable, but slower.
    """
    outer_iters: int = 10
    selection_threshold: float = 0.0
    conv_threshold: float = 1e-3
