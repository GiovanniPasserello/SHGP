import gpflow
import numpy as np
import tensorflow as tf

from shgp.data.metadata_metrics import BananaMetricsMetaDataset, BreastCancerMetricsMetaDataset, \
    CrabsMetricsMetaDataset, ElectricityMetricsMetaDataset, HeartMetricsMetaDataset, IonosphereMetricsMetaDataset, \
    MagicMetricsMetaDataset, PimaMetricsMetaDataset, TwonormMetricsMetaDataset, RingnormMetricsMetaDataset
from shgp.data.metadata_reinit import ReinitMetaDataset
from shgp.inducing.initialisation_methods import h_reinitialise_PGPR
from shgp.utilities.metrics import ExperimentResults
from shgp.utilities.train_pgpr import train_pgpr
from shgp.utilities.train_svgp import train_svgp

np.random.seed(42)
tf.random.set_seed(42)

"""
A comparison of PGPR with heteroscedastic greedy variance reinitialisation and SVGP with a Bernoulli
likelihood and fixed inducing points initialised with kmeans. 
These are the 'best' case scenarios of best method used for a thorough evaluation of PGPR vs SVGP. 
In particular we evaluate the performance of very sparse models to see whether HGV is beneficial.
"""


def run_metrics_experiment(metadata):
    svgp_results, pgpr_results = ExperimentResults(), ExperimentResults()

    for c in range(metadata.num_cycles):
        # Different train_test_split for each iteration
        X, Y, X_test, Y_test = metadata.load_train_test_split()

        print("Beginning cycle {}...".format(c + 1))
        svgp_result, pgpr_result = run_iteration(metadata, X, Y, X_test, Y_test)
        svgp_results.add_result(svgp_result)
        pgpr_results.add_result(pgpr_result)
        print("SVGP: ELBO = {:.6f}, ACC = {:.6f}, NLL = {:.6f}.".format(svgp_result.elbo, svgp_result.accuracy, svgp_result.nll))
        print("PGPR: ELBO = {:.6f}, ACC = {:.6f}, NLL = {:.6f}.".format(pgpr_result.elbo, pgpr_result.accuracy, pgpr_result.nll))

    svgp_dist = svgp_results.compute_distribution()
    pgpr_dist = pgpr_results.compute_distribution()

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

    print("\nPGPR Distribution:")
    print("ELBO - max: {:.6f}, min: {:.6f}, median: {:.6f}, mean: {:.6f}, std: {:.6f}.".format(
        pgpr_dist[0].elbo, pgpr_dist[1].elbo, pgpr_dist[2].elbo, pgpr_dist[3].elbo, pgpr_dist[4].elbo)
    )
    print("ACC  - max: {:.6f}, min: {:.6f}, median: {:.6f}, mean: {:.6f}, std: {:.6f}.".format(
        pgpr_dist[0].accuracy, pgpr_dist[1].accuracy, pgpr_dist[2].accuracy, pgpr_dist[3].accuracy, pgpr_dist[4].accuracy)
    )
    print("NLL  - max: {:.6f}, min: {:.6f}, median: {:.6f}, mean: {:.6f}, std: {:.6f}.".format(
        pgpr_dist[0].nll, pgpr_dist[1].nll, pgpr_dist[2].nll, pgpr_dist[3].nll, pgpr_dist[4].nll)
    )


def run_iteration(metadata, X, Y, X_test, Y_test):
    ########
    # SVGP #
    ########

    _, svgp_result = train_svgp(
        X, Y, metadata.M,
        train_iters=metadata.svgp_iters,
        X_test=X_test, Y_test=Y_test
    )

    ########
    # PGPR #
    ########

    _, pgpr_result = train_pgpr(
        X, Y,
        metadata.inner_iters, metadata.opt_iters, metadata.ci_iters,
        kernel_type=gpflow.kernels.SquaredExponential,
        M=metadata.M,
        init_method=h_reinitialise_PGPR,
        reinit_metadata=ReinitMetaDataset(),
        X_test=X_test,
        Y_test=Y_test
    )

    return svgp_result, pgpr_result


if __name__ == '__main__':
    run_metrics_experiment(RingnormMetricsMetaDataset())
