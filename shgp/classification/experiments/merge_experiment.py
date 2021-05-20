import gpflow
import numpy as np
import tensorflow as tf

from shgp.data.metadata_metrics import BreastCancerMetricsMetaDataset
from shgp.data.metadata_reinit import ReinitMetaDataset
from shgp.inducing.initialisation_methods import k_means, h_reinitialise_PGPR
from shgp.utilities.metrics import ExperimentResults
from shgp.utilities.train_pgpr_svgp import train_pgpr_svgp
from shgp.utilities.train_svgp import train_svgp


np.random.seed(42)
tf.random.set_seed(42)

"""
An attempt to merge heteroscedastic greedy variance reinitialisation with SVGP Bernoulli.

This may be beneficial for two reasons:
    A) Heteroscedastic greedy variance offers a strong inducing point initialisation procedure.
    B) SVGP obtains a better ELBO than PGPR, as PGPR uses a lower bound to the Bernoulli likelihood. 

This method works in 7 steps:
    1. Initialise PGPR
    2. Select Z using heteroscedastic greedy variance
    3. Pass Z and PGPR q(u) to SVGP
    4. Train SVGP
    5. Pass SVGP kernel hyperparameters to PGPR
    6. Reinitialise Z using heteroscedastic greedy variance
    7. Repeat 2-6 until convergence of the SVGP ELBO
"""


def run_metrics_experiment(metadata):
    pgpr_svgp_results, svgp_results = ExperimentResults(), ExperimentResults()

    for c in range(metadata.num_cycles):
        # Different train_test_split for each iteration
        X, Y, X_test, Y_test = metadata.load_train_test_split()

        print("Beginning cycle {}...".format(c + 1))
        pgpr_svgp_result, svgp_result = run_iteration(metadata, X, Y, X_test, Y_test)
        pgpr_svgp_results.add_result(pgpr_svgp_result)
        svgp_results.add_result(svgp_result)
        print("PGPR-SVGP: ELBO = {:.6f}, ACC = {:.6f}, NLL = {:.6f}.".format(pgpr_svgp_result.elbo, pgpr_svgp_result.accuracy, pgpr_svgp_result.nll))
        print("SVGP: ELBO = {:.6f}, ACC = {:.6f}, NLL = {:.6f}.".format(svgp_result.elbo, svgp_result.accuracy, svgp_result.nll))

    pgpr_svgp_dist = pgpr_svgp_results.compute_distribution()
    svgp_dist = svgp_results.compute_distribution()

    print("\nPGPR-SVGP Distribution:")
    print("ELBO - max: {:.6f}, min: {:.6f}, median: {:.6f}, mean: {:.6f}, std: {:.6f}.".format(
        pgpr_svgp_dist[0].elbo, pgpr_svgp_dist[1].elbo, pgpr_svgp_dist[2].elbo, pgpr_svgp_dist[3].elbo, pgpr_svgp_dist[4].elbo)
    )
    print("ACC  - max: {:.6f}, min: {:.6f}, median: {:.6f}, mean: {:.6f}, std: {:.6f}.".format(
        pgpr_svgp_dist[0].accuracy, pgpr_svgp_dist[1].accuracy, pgpr_svgp_dist[2].accuracy, pgpr_svgp_dist[3].accuracy, pgpr_svgp_dist[4].accuracy)
    )
    print("NLL  - max: {:.6f}, min: {:.6f}, median: {:.6f}, mean: {:.6f}, std: {:.6f}.".format(
        pgpr_svgp_dist[0].nll, pgpr_svgp_dist[1].nll, pgpr_svgp_dist[2].nll, pgpr_svgp_dist[3].nll, pgpr_svgp_dist[4].nll)
    )

    print("\nSVGP Distribution:")
    print("ELBO - max: {:.6f}, min: {:.6f}, median: {:.6f}, mean: {:.6f}, std: {:.6f}.".format(
        svgp_dist[0].elbo, svgp_dist[1].elbo, svgp_dist[2].elbo, svgp_dist[3].elbo, svgp_dist[4].elbo)
    )
    print("ACC  - max: {:.6f}, min: {:.6f}, median: {:.6f}, mean: {:.6f}, std: {:.6f}.".format(
        svgp_dist[0].accuracy, svgp_dist[1].accuracy, svgp_dist[2].accuracy, svgp_dist[3].accuracy, svgp_dist[4].accuracy)
    )
    print("NLL  - max: {:.6f}, min: {:.6f}, median: {:.6f}, mean: {:.6f}, std: {:.6f}.".format(
        svgp_dist[0].nll, svgp_dist[1].nll, svgp_dist[2].nll, svgp_dist[3].nll, svgp_dist[4].nll)
    )


def run_iteration(metadata, X, Y, X_test, Y_test):
    kernel_type = gpflow.kernels.SquaredExponential

    #############
    # PGPR-SVGP #
    #############

    _, pgpr_svgp_result = train_pgpr_svgp(
        X, Y,
        kernel_type=kernel_type,
        M=metadata.M,
        opt_iters=metadata.svgp_iters,
        init_method=h_reinitialise_PGPR,
        reinit_metadata=ReinitMetaDataset(),
        X_test=X_test, Y_test=Y_test
    )

    ########
    # SVGP #
    ########

    _, svgp_result = train_svgp(
        X, Y,
        kernel_type=kernel_type,
        M=metadata.M,
        train_iters=metadata.svgp_iters,
        init_method=k_means,
        X_test=X_test, Y_test=Y_test
    )

    return pgpr_svgp_result, svgp_result


if __name__ == '__main__':
    run_metrics_experiment(BreastCancerMetricsMetaDataset())
