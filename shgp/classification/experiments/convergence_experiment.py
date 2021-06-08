import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from shgp.data.metadata_reinit import ReinitMetaDataset
from shgp.data.metadata_convergence import BananaConvergenceMetaDataset
from shgp.inducing.initialisation_methods import h_reinitialise_PGPR, k_means, uniform_subsample
from shgp.utilities.train_pgpr import train_pgpr
from shgp.utilities.train_svgp import train_svgp

np.random.seed(42)
tf.random.set_seed(42)

"""
A comparison of PGPR with two different inducing point initialisation procedures. Here we investigate
the convergence of the ELBO using Adam vs BFGS on both k_means and hgv. This allows us to analyse 
whether the inducing points of hgv are initialised closer than k_means to the gradient-optimised result.
To evaluate other inducing point selection methods, change the model definitions below.

Observations:
    HGV is always better than K-means for both Adam & BFGS.
    BFGS learns quickest initially, but Adam often overtakes at the end.
    SVGP requires much larger M to converge to its optimal elbo than PGPR requires to converge to its optimal ELBO.
"""


def run_convergence_experiment(X, Y, M, opt_iters):
    ##############################################
    # SVGP with Full Inducing Points ('Optimal') #
    ##############################################

    print("Training non-sparse model...")
    _, elbo_svgp = train_svgp(X, Y, M=len(X), train_iters=250, init_method=uniform_subsample)
    optimal = np.full(opt_iters + 1, elbo_svgp)

    print("Non-sparse result:")
    print("optimal = {}".format(elbo_svgp))

    ############################################
    # Z for Different Reinitialisation Methods #
    ############################################

    # Get the initial inducing points for hgv and kmeans
    print("Selecting inducing points...")
    pgpr_hgv, _ = train_pgpr(
        X, Y,
        10, 500, 10,
        M=M, init_method=h_reinitialise_PGPR, reinit_metadata=ReinitMetaDataset()
    )
    hgv_Z = pgpr_hgv.inducing_variable.Z.variables[0].numpy()
    kmeans_Z = k_means(X, M)
    print("Inducing points selected.")

    ####################
    # Convergence test #
    ####################

    results_hgv_bfgs, results_hgv_adam, results_kmeans_bfgs, results_kmeans_adam = \
        test_convergence((X, Y), opt_iters, hgv_Z, kmeans_Z)

    results = list(zip(results_hgv_bfgs, results_hgv_adam, results_kmeans_bfgs, results_kmeans_adam))

    print("Sparse results:")
    print("results_hgv_bfgs = {}".format(results_hgv_bfgs))
    print("results_hgv_adam = {}".format(results_hgv_adam))
    print("results_kmeans_bfgs = {}".format(results_kmeans_bfgs))
    print("results_kmeans_adam = {}".format(results_kmeans_adam))

    return results, optimal


# Test the performance of a model with a given number of inducing points, M.
def test_convergence(data, opt_iters, hgv_Z, kmeans_Z):
    print("Training HGV BFGS...")
    bfgs_opt = gpflow.optimizers.Scipy()
    results_hgv_bfgs = train_convergence_svgp(data, opt_iters, hgv_Z, bfgs_opt)
    print("HGV BFGS trained: ELBO = {}".format(results_hgv_bfgs[-1]))
    print("Training HGV Adam...")
    adam_opt = tf.optimizers.Adam(beta_1=0.5, beta_2=0.5)
    results_hgv_adam = train_convergence_svgp(data, opt_iters, hgv_Z, adam_opt)
    print("HGV Adam trained: ELBO = {}".format(results_hgv_adam[-1]))

    print("Training K-means BFGS...")
    bfgs_opt = gpflow.optimizers.Scipy()
    results_kmeans_bfgs = train_convergence_svgp(data, opt_iters, kmeans_Z, bfgs_opt)
    print("K-means BFGS trained: ELBO = {}".format(results_kmeans_bfgs[-1]))
    print("Training K-means BFGS...")
    adam_opt = tf.optimizers.Adam(beta_1=0.5, beta_2=0.5)
    results_kmeans_adam = train_convergence_svgp(data, opt_iters, kmeans_Z, adam_opt)
    print("K-means Adam trained: ELBO = {}".format(results_kmeans_adam[-1]))

    return results_hgv_bfgs, results_hgv_adam, results_kmeans_bfgs, results_kmeans_adam


def train_convergence_svgp(data, opt_iters, Z, opt):
    """
    Train an SVGP model until completion, recording the ELBO every 'elbo_period' iterations.
    If we error, keep retrying until success - this is due to a spurious Cholesky error.
    """
    model = gpflow.models.SVGP(
        kernel=gpflow.kernels.SquaredExponential(),
        likelihood=gpflow.likelihoods.Bernoulli(tf.sigmoid),
        inducing_variable=Z.copy()
    )
    gpflow.set_trainable(model.inducing_variable, True)

    # Try to run the full optimisation cycle.
    try:
        results = [model.elbo(data).numpy()]
        if isinstance(opt, gpflow.optimizers.Scipy):
            # Optimise kernel hyperparameters, recording the ELBO after each step.
            for _ in range(opt_iters):
                opt.minimize(model.training_loss_closure(data), variables=model.trainable_variables, options=dict(maxiter=1))
                results.append(model.elbo(data).numpy())
        else:
            # Optimise kernel hyperparameters, recording the ELBO after each step.
            for step in range(opt_iters):
                opt.lr.assign(decayed_learning_rate(step, opt_iters))
                opt.minimize(model.training_loss_closure(data), var_list=model.trainable_variables)
                results.append(model.elbo(data).numpy())

    # If we fail due to a (spurious) Cholesky error, restart.
    except tf.errors.InvalidArgumentError as error:
        msg = error.message
        if "Cholesky" not in msg and "invertible" not in msg:
            raise error
        else:
            print("Cholesky error caught, retrying...")
            return train_convergence_svgp(data, opt_iters, Z, opt)

    return results


def decayed_learning_rate(step, decay_steps, initial_learning_rate=0.5, alpha=0.05):
    step = min(step, decay_steps)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return initial_learning_rate * decayed


def plot_results(name, opt_iters, results, optimal):
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.tick_params(labelright=True)

    # Axis labels
    plt.title('Convergence of Inducing Point Methods - {} Dataset'.format(name))
    plt.ylabel('ELBO')
    plt.xlabel('Iterations')
    # Axis limits
    plt.xlim(0, opt_iters)
    plt.xticks(np.arange(0, opt_iters + 1, 50))
    plt.yscale('symlog')

    # Setup each subplot
    iter_array = np.arange(opt_iters + 1)
    plt.plot(iter_array, optimal, color='k', linestyle='dashed', label='Optimal', zorder=101)
    plt.plot(iter_array, results[:, 0], label='HGV GO - BFGS', zorder=100)
    plt.plot(iter_array, results[:, 1], label='HGV GO - Adam', zorder=98)
    plt.plot(iter_array, results[:, 2], label='K-means GO - BFGS', zorder=99)
    plt.plot(iter_array, results[:, 3], label='K-means GO - Adam', zorder=97)

    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    # Load data
    dataset = BananaConvergenceMetaDataset()  # to test another dataset, just change this definition
    X, Y = dataset.load_data()

    # Get convergence results
    results, optimal = run_convergence_experiment(X, Y, dataset.M, dataset.opt_iters)

    # Plot results
    plot_results(dataset.name, dataset.opt_iters, np.array(results), optimal)
