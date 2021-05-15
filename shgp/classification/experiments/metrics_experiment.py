from datetime import datetime

import gpflow
import numpy as np
import tensorflow as tf

from shgp.data.metadata_metrics import BananaMetricsMetaDataset
from shgp.data.metadata_reinit import ReinitMetaDataset
from shgp.inducing.initialisation_methods import uniform_subsample, h_reinitialise_PGPR
from shgp.robustness.contrained_kernels import ConstrainedExpSEKernel
from shgp.utilities.metrics import compute_accuracy, compute_nll, ExperimentResult, ExperimentResults
from shgp.utilities.train_pgpr import train_pgpr

"""
... May need multiple of these classes - one for each experiment I want to run
"""

np.random.seed(0)
tf.random.set_seed(0)


def run_metrics_experiment(metadata):
    X, Y = metadata.load_data()

    svgp_results, pgpr_results = ExperimentResults(), ExperimentResults()
    for c in range(metadata.num_cycles):
        print("Beginning cycle {}...".format(c + 1))
        svgp_result, pgpr_result = run_iteration(metadata, X, Y)
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


# TODO: Make this return the results corresponding to the best PGPR ELBO during the train loop
def run_iteration(metadata, X, Y):
    ########
    # SVGP #
    ########

    svgp = gpflow.models.SVGP(
        kernel=ConstrainedExpSEKernel(),
        likelihood=gpflow.likelihoods.Bernoulli(invlink=tf.sigmoid),
        inducing_variable=uniform_subsample(X, metadata.M)[0].copy()
    )
    gpflow.set_trainable(svgp.inducing_variable, False)
    gpflow.optimizers.Scipy().minimize(
        svgp.training_loss_closure((X, Y)),
        variables=svgp.trainable_variables,
        options=dict(maxiter=metadata.svgp_iters)
    )
    svgp_F, _ = svgp.predict_f(X)
    svgp_elbo, svgp_acc, svgp_nll = svgp.elbo((X, Y)), compute_accuracy(Y, svgp_F), compute_nll(Y, svgp_F)
    svgp_result = ExperimentResult(svgp_elbo, svgp_acc, svgp_nll)

    ########
    # PGPR #
    ########

    pgpr, pgpr_elbo = train_pgpr(
        X, Y,
        metadata.inner_iters, metadata.opt_iters, metadata.ci_iters,
        M=metadata.M,
        init_method=h_reinitialise_PGPR,
        reinit_metadata=ReinitMetaDataset()
    )
    pgpr_F, _ = pgpr.predict_f(X)
    pgpr_elbo, pgpr_acc, pgpr_nll = pgpr.elbo(), compute_accuracy(Y, pgpr_F), compute_nll(Y, pgpr_F)
    pgpr_result = ExperimentResult(pgpr_elbo, pgpr_acc, pgpr_nll)

    return svgp_result, pgpr_result


if __name__ == '__main__':
    run_metrics_experiment(BananaMetricsMetaDataset())
