from dataclasses import dataclass

"""
An alternative to storing this metadata per dataset would be to devise some method
of automatically determining suitable hyperparameter settings for any dataset.
This is a difficult task as they vary wildly depending on dataset.
"""


@dataclass
class ReinitMetaDataset:
    """
        A dataset utilities class for reinitialisation training hyperparameters.

        :param outer_iters: The maximum number of times to attempt reinitialisation (i.e., the outer loop).
        :param selection_threshold: The threshold on the trace term which tells us when we have enough inducing points.
    """
    outer_iters: int = 10
    selection_threshold: int = 0
