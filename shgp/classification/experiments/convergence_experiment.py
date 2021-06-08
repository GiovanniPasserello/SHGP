import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from shgp.data.metadata_reinit import ReinitMetaDataset
from shgp.data.metadata_convergence import IonosphereConvergenceMetaDataset
from shgp.inducing.initialisation_methods import h_reinitialise_PGPR, k_means
from shgp.models.pgpr import PGPR
from shgp.robustness.contrained_kernels import ConstrainedExpSEKernel
from shgp.utilities.train_pgpr import train_pgpr

np.random.seed(42)
tf.random.set_seed(42)

"""
A comparison of PGPR with two different inducing point initialisation procedures. Here we investigate
the convergence of the ELBO using Adam vs BFGS on both k_means and hgv. This allows us to analyse 
whether the inducing points of hgv are initialised closer than k_means to the gradient-optimised result.
To evaluate other inducing point selection methods, change the model definitions below.
"""


def run_convergence_experiment(X, Y, M, inner_iters, opt_iters, ci_iters):
    ################################################
    # PGPR with Different Reinitialisation Methods #
    ################################################

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

    results_hgv_bfgs, results_hgv_adam, results_kmeans_bfgs, results_kmeans_adam = \
        test_convergence(X, Y, inner_iters, opt_iters, ci_iters, hgv_Z, kmeans_Z)

    results = list(zip(results_hgv_bfgs, results_hgv_adam, results_kmeans_bfgs, results_kmeans_adam))

    print("Sparse results:")
    print("results_hgv_bfgs = {}".format(results_hgv_bfgs))
    print("results_hgv_adam = {}".format(results_hgv_adam))
    print("results_kmeans_bfgs = {}".format(results_kmeans_bfgs))
    print("results_kmeans_adam = {}".format(results_kmeans_adam))

    ##############################################
    # PGPR with Full Inducing Points ('Optimal') #
    ##############################################

    print("Training non-sparse model...")
    _, elbo_pgpr = train_pgpr(X, Y, inner_iters, opt_iters, ci_iters)
    optimal = np.full(len(results), elbo_pgpr)

    print("Non-sparse result:")
    print("optimal = {}".format(elbo_pgpr))

    return results, optimal


# Test the performance of a model with a given number of inducing points, M.
def test_convergence(X, Y, inner_iters, opt_iters, ci_iters, hgv_Z, kmeans_Z):
    print("Training HGV BFGS...")
    bfgs_opt = gpflow.optimizers.Scipy()
    results_hgv_bfgs = train_convergence_pgpr(X, Y, inner_iters, opt_iters, ci_iters, hgv_Z, bfgs_opt)
    print("HGV BFGS trained: ELBO = {}".format(results_hgv_bfgs[-1]))
    print("Training HGV Adam...")
    adam_opt = tf.optimizers.Adam(beta_1=0.5, beta_2=0.5)
    results_hgv_adam = train_convergence_pgpr(X, Y, inner_iters, opt_iters, ci_iters, hgv_Z, adam_opt)
    print("HGV Adam trained: ELBO = {}".format(results_hgv_adam[-1]))

    print("Training K-means BFGS...")
    bfgs_opt = gpflow.optimizers.Scipy()
    results_kmeans_bfgs = train_convergence_pgpr(X, Y, inner_iters, opt_iters, ci_iters, kmeans_Z, bfgs_opt)
    print("K-means BFGS trained: ELBO = {}".format(results_kmeans_bfgs[-1]))
    print("Training K-means BFGS...")
    adam_opt = tf.optimizers.Adam(beta_1=0.5, beta_2=0.5)
    results_kmeans_adam = train_convergence_pgpr(X, Y, inner_iters, opt_iters, ci_iters, kmeans_Z, adam_opt)
    print("K-means Adam trained: ELBO = {}".format(results_kmeans_adam[-1]))

    return results_hgv_bfgs, results_hgv_adam, results_kmeans_bfgs, results_kmeans_adam


def train_convergence_pgpr(X, Y, inner_iters, opt_iters, ci_iters, Z, opt):
    """
    Train a PGPR model until completion, recording the ELBO every 'elbo_period' iterations.
    If we error, keep retrying until success - this is due to a spurious Cholesky error.
    """
    model = PGPR(
        data=(X, Y),
        kernel=ConstrainedExpSEKernel(),
        inducing_variable=Z  # gradient-optimised inducing points, initialised at Z
    )
    gpflow.set_trainable(model.inducing_variable, True)
    results = [model.elbo().numpy()]

    # Try to run the full optimisation cycle.
    try:
        if isinstance(opt, gpflow.optimizers.Scipy):
            for _ in range(inner_iters):
                # Optimise kernel hyperparameters, recording the ELBO after each step.
                for _ in range(opt_iters):
                    opt.minimize(model.training_loss, variables=model.trainable_variables, options=dict(maxiter=1))
                    results.append(model.elbo().numpy())
                # Update local variational parameters.
                model.optimise_ci(num_iters=ci_iters)
        else:
            step = 0
            decay_steps = inner_iters * opt_iters
            for _ in range(inner_iters):
                # Optimise kernel hyperparameters, recording the ELBO after each step.
                for i in range(opt_iters):
                    opt.lr.assign(decayed_learning_rate(step, decay_steps))
                    opt.minimize(model.training_loss, var_list=model.trainable_variables)
                    results.append(model.elbo().numpy())
                    step += 1
                # Update local variational parameters.
                model.optimise_ci(num_iters=ci_iters)

    # If we fail due to a (spurious) Cholesky error, restart.
    except tf.errors.InvalidArgumentError as error:
        msg = error.message
        if "Cholesky" not in msg and "invertible" not in msg:
            raise error
        else:
            print("Cholesky error caught, retrying...")
            return train_convergence_pgpr(X, Y, inner_iters, opt_iters, ci_iters, Z, opt)

    return results


def decayed_learning_rate(step, decay_steps, initial_learning_rate=0.5, alpha=0.05):
    step = min(step, decay_steps)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return initial_learning_rate * decayed


def plot_results(name, max_iters, results, optimal):
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.tick_params(labelright=True)

    # Axis labels
    plt.title('Convergence of Inducing Point Methods - {} Dataset'.format(name))
    plt.ylabel('ELBO')
    plt.xlabel('Iterations')
    # Axis limits
    plt.xlim(0, max_iters)

    # Setup each subplot
    iter_array = np.arange(max_iters + 1)
    plt.plot(iter_array, optimal, color='k', linestyle='dashed', label='Optimal', zorder=101)
    plt.plot(iter_array, results[:, 0], label='HGV GO - BFGS', zorder=100)
    plt.plot(iter_array, results[:, 1], label='HGV GO - Adam', zorder=98)
    plt.plot(iter_array, results[:, 2], label='K-means GO - BFGS', zorder=99)
    plt.plot(iter_array, results[:, 3], label='K-means GO - Adam', zorder=97)

    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    # Load data
    dataset = IonosphereConvergenceMetaDataset()  # to test another dataset, just change this definition
    X, Y = dataset.load_data()

    results, optimal = run_convergence_experiment(
        X, Y,
        dataset.M, dataset.inner_iters, dataset.opt_iters, dataset.ci_iters
    )

    max_iters = dataset.inner_iters * dataset.opt_iters

    plot_results(dataset.name, max_iters, np.array(results), optimal)
