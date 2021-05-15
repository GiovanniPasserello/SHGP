import gpflow
import numpy as np
import tensorflow as tf

from shgp.data.metadata_metrics import BananaMetricsMetaDataset
from shgp.data.metadata_reinit import ReinitMetaDataset
from shgp.inducing.initialisation_methods import h_reinitialise_PGPR, k_means
from shgp.robustness.contrained_kernels import ConstrainedExpSEKernel
from shgp.utilities.metrics import compute_test_metrics, ExperimentResult, ExperimentResults
from shgp.utilities.train_pgpr import train_pgpr

np.random.seed(42)
tf.random.set_seed(42)


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

    svgp = gpflow.models.SVGP(
        kernel=ConstrainedExpSEKernel(),
        likelihood=gpflow.likelihoods.Bernoulli(invlink=tf.sigmoid),
        inducing_variable=k_means(X, metadata.M).copy()
    )
    gpflow.set_trainable(svgp.inducing_variable, False)  # we don't want to optimize - the comparison is fixed methods
    gpflow.optimizers.Scipy().minimize(
        svgp.training_loss_closure((X, Y)),
        variables=svgp.trainable_variables,
        options=dict(maxiter=metadata.svgp_iters)
    )
    svgp_result = ExperimentResult(svgp.elbo((X, Y)), *compute_test_metrics(svgp, X_test, Y_test))

    ########
    # PGPR #
    ########

    pgpr, pgpr_result = train_pgpr(
        X, Y,
        metadata.inner_iters, metadata.opt_iters, metadata.ci_iters,
        M=metadata.M,
        init_method=h_reinitialise_PGPR,
        reinit_metadata=ReinitMetaDataset(),
        X_test=X_test,
        Y_test=Y_test
    )

    return svgp_result, pgpr_result


if __name__ == '__main__':
    run_metrics_experiment(BananaMetricsMetaDataset())
