import gpflow
import numpy as np
import tensorflow as tf

from datetime import datetime
from tensorflow import sigmoid

from shgp.inducing.initialisation_methods import uniform_subsample, h_reinitialise_PGPR
from shgp.kernels.ConstrainedSEKernel import ConstrainedSEKernel
from shgp.models.pgpr import PGPR

np.random.seed(0)
tf.random.set_seed(0)


# TODO: Better convergence guarantees of training PGPR
# TODO: Experiments - each dataset with 100 inducing points, ACCURACY & ELBO.
#       Run 5-10 times and average. Bern GO, PGPR GO, PGPR GV, PGPR HGV.
# TODO: Use test sets for evaluation


# TODO: Move to utils
# Polya-Gamma uses logit link / sigmoid
def invlink(f):
    return gpflow.likelihoods.Bernoulli(invlink=sigmoid).invlink(f).numpy()


def standardise_features(data):
    """
    Standardise all features to 0 mean and unit variance.

    :param: data - the input data.
    :return: the normalised data.
    """
    data_means = data.mean(axis=0)  # mean value per feature
    data_stds = data.std(axis=0)  # standard deviation per feature

    # standardise each feature
    return (data - data_means) / data_stds


def load_fertility():
    dataset = "../data/classification/fertility.txt"

    data = np.loadtxt(dataset, delimiter=",")
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1)

    # TODO: This might be a good example to discuss sparsity (9 inducing points PGPR at 1e-6)
    # The comparison here may be that:
    # Yes, we can use 100 inducing points and Bernoulli is almost as quick as PGPR,
    # but why use 100 when we can use 9? A large downside is that if we use 9 points
    # with Bernoulli, it takes about 25 seconds as opposed to 2.38 with PGPR or 2.39
    # with Bernoulli 100 points.

    # TODO: The effect of using a constrained SE kernel slightly affects optimal values.
    # Is there a good way to ensure we are not overly constraining the system?

    NUM_INDUCING = 100  # quicker with 100 than with 9
    BERN_ITERS = 500  # best with 100: -38.982949 (with 9: -38.991571)
    PGPR_ITERS = (5, 25, 5)  # best with 100: -39.354674
    GREEDY_THRESHOLD = 1e-6  # (early stops at 9): -39.354684

    return X, Y, NUM_INDUCING, BERN_ITERS, PGPR_ITERS, GREEDY_THRESHOLD


def load_breast_cancer():
    dataset = "../data/classification/breast_cancer.txt"

    data = np.loadtxt(dataset, delimiter=",")
    X = data[:, 2:]
    Y = data[:, 1].reshape(-1, 1) - 1

    # TODO: This might be a good example to discuss PG beating Bernoulli with full inducing points.
    # TODO: Add sparsity experiment?
    # The comparison here may be that:
    # With 194 inducing points fixed for Bernoulli, the ELBO is -116.46. Interestingly,
    # the PGPR ELBO is -103.88 even though both models receive the same fixed set of data
    # points - the full training set. Why is this? This is very interesting to explore!
    # With fewer inducing points, Bernoulli is able to reach an ELBO of -100, but is very
    # slow to converge. On the other hand PGPR is much quicker and the ELBO is only about 3-4
    # nats lower. To reinforce this, with 59 inducing points Bernoulli takes 23.35
    # seconds and converges to an ELBO of -100.71, whereas PGPR takes 10.35 seconds for an ELBO
    # of -104.19. If we reduce the threshold it can converge even quicker though - for example
    # with a threshold at 2e-1 we converge in 8.79 and get a better ELBO! (perhaps we need a better
    # convergence setup than the current absolute difference check - the number of cycles matters).

    # In a way, the fact that Bernoulli achieves -116.46 (a suboptimal value) with the full training set
    # emphasises the unreliability of gradient-based methods (changing the random seed allows Bernoulli
    # to converge). One large benefit of greedy variance is that there is no stochasticity in the
    # selection process. The same points will be chosen no matter the setup. The only thing that does
    # change when using greedy variance is the optimisation of hyperparameters (which is often stochastic).
    # The added randomness of SVGP Bernoulli makes it quite unreliable - some seeds converge in 2 seconds,
    # other seeds take 25 seconds. Or is this more to do with the kernel being constrained? see below:

    # TODO: The effect of using a constrained SE kernel is large. It makes the difference between
    # Bernoulli converging to its optimal values or not. The problem of -116.46 Bernoulli is removed
    # when we use the standard SE kernel with NUM_INDUCING=194. But also when using the unconstrained
    # kernel we get worse values when NUM_INDUCING=59 for Bernoulli - is one definitely better? For
    # consistency we should probably use constrained SE kernels everywhere to avoid Cholesky errors
    # and make sure to mention this in the report. PGPR isn't affected in this example, but Bernoulli
    # is significantly affected - is this a special case?

    NUM_INDUCING = 194  # quicker with 194 than with 59
    BERN_ITERS = 500  # best with 194: -116.463033 (with 59: -100.710896)
    PGPR_ITERS = (5, 10, 5)  # best with 194: -103.882682
    GREEDY_THRESHOLD = 5e-1  # (early stops at 59): -104.194101

    return X, Y, NUM_INDUCING, BERN_ITERS, PGPR_ITERS, GREEDY_THRESHOLD


def load_ionosphere():
    dataset = "../data/classification/ionosphere.txt"

    data = np.loadtxt(dataset, delimiter=",")
    X = data[:, :-1]
    X = np.delete(X, 1, axis=1)  # remove a column of zeros (33 dimensions)
    Y = data[:, -1].reshape(-1, 1)

    # In this case, Bernoulli GO seems to perform much better than PGPR.
    # This is likely because of the small dataset size.

    NUM_INDUCING = 156  # quicker with 156 than with 351
    BERN_ITERS = 100  # best with 351: -100.021138 (with 156: -107.370414)
    PGPR_ITERS = (5, 25, 5)  # best with 351: -126.962021
    GREEDY_THRESHOLD = 1  # (early stops at 156): -127.549833

    return X, Y, NUM_INDUCING, BERN_ITERS, PGPR_ITERS, GREEDY_THRESHOLD


def load_pima():
    dataset = "../data/classification/pima-diabetes.csv"

    data = np.loadtxt(dataset, delimiter=",")
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1)

    # As the size of the datasets grow, we begin to see the benefits of PGPR. Where Bernoulli becomes
    # infeasible, PGPR is able to find sparser and more efficient solutions.

    # with 768 bern took 43.24 seconds, with 123 bern took 5.67 seconds
    # with 768 PGPR took 26.59 seconds, with 123 PGPR took 9.85 seconds
    # (accounting for looping convergence, PGPR 768 took 18.48 seconds and 123 took 2.63 seconds)
    # TODO: Sort out the convergence criterion: Perhaps it's better to just check if the ELBO increases?
    #       Perhaps this doesn't matter as speed is not the focus on this project.
    #       It would be an interesing point for future work to look into.

    NUM_INDUCING = 768  # quicker with 123 than with 768
    BERN_ITERS = 100  # best with 768: -372.840412 (with 123: -374.059495)
    PGPR_ITERS = (5, 25, 5)  # best with 768: -377.604798
    GREEDY_THRESHOLD = 1  # (early stops at 123): -378.048993

    return X, Y, NUM_INDUCING, BERN_ITERS, PGPR_ITERS, GREEDY_THRESHOLD


def load_magic():
    dataset = "../data/classification/magic.txt"

    data = np.loadtxt(dataset, delimiter=",")
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1)

    # TODO: Large comparison, how do we reliably find results when this takes so long to run?
    # Maybe smaller datasets contain more feasible insight - too many variables to consider?
    # Here it is very important to have good convergence guarantees - the current absolute
    # difference check is not good enough. Perhaps we can check if the ELBO increases, but
    # is that reliable - what if we reach a local minimum which we might leave later?

    # TODO: Could we just prune the dataset?

    NUM_INDUCING = 100  # 200
    BERN_ITERS = 200  # 100
    PGPR_ITERS = (5, 25, 10)  # This benefits from being larger than this, but becomes slow
    GREEDY_THRESHOLD = 0  # 100

    return X, Y, NUM_INDUCING, BERN_ITERS, PGPR_ITERS, GREEDY_THRESHOLD


def compute_accuracy(Y, F):
    preds = np.round(invlink(F))
    return np.sum(Y == preds) / len(Y)


# Average NLL
def compute_nll(Y, F):
    P = invlink(F)
    return -np.log(np.where(Y, P, 1 - P)).mean()


def run_experiment():
    initial_inducing_inputs, _ = uniform_subsample(X, NUM_INDUCING)

    ########
    # SVGP #
    ########
    svgp = gpflow.models.SVGP(
        kernel=ConstrainedSEKernel(),
        likelihood=gpflow.likelihoods.Bernoulli(invlink=sigmoid),
        inducing_variable=initial_inducing_inputs.copy()
    )
    svgp_start = datetime.now()
    gpflow.optimizers.Scipy().minimize(
        svgp.training_loss_closure((X, Y)),
        variables=svgp.trainable_variables,
        options=dict(maxiter=SVGP_ITERS)
    )
    svgp_time = datetime.now() - svgp_start
    print("svgp trained in {:.2f} seconds".format(svgp_time.total_seconds()))
    print("ELBO = {:.6f}".format(svgp.elbo((X,Y))))
    print("Accuracy = {:.6f}".format(compute_accuracy(Y, svgp.predict_f(X)[0])))
    print("NLL = {:.6f}".format(compute_nll(Y, svgp.predict_f(X)[0])))

    ########
    # PGPR #
    ########
    # Define model
    pgpr = PGPR(
        data=(X, Y),
        kernel=ConstrainedSEKernel(),
        inducing_variable=initial_inducing_inputs.copy()
    )

    # Begin optimisation
    pgpr_start = datetime.now()
    prev_elbo = pgpr.elbo()
    opt = gpflow.optimizers.Scipy()
    iter_limit = 10  # to avoid infinite loops
    while True:
        h_reinitialise_PGPR(pgpr, X, NUM_INDUCING, GREEDY_THRESHOLD)

        # Optimize model
        for _ in range(PGPR_ITERS[0]):
            opt.minimize(pgpr.training_loss, variables=pgpr.trainable_variables, options=dict(maxiter=PGPR_ITERS[1]))
            pgpr.optimise_ci(PGPR_ITERS[2])

        next_elbo = pgpr.elbo()
        print("Previous ELBO: {}, Next ELBO: {}".format(prev_elbo, next_elbo))
        if np.abs(next_elbo - prev_elbo) <= 1e-3 or iter_limit == 0:
            if iter_limit == 0:
                print("PGPR failed to converge.")
            break
        prev_elbo = next_elbo
        iter_limit -= 1

    pgpr_time = datetime.now() - pgpr_start

    print("pgpr trained in {:.2f} seconds".format(pgpr_time.total_seconds()))
    print("Final number of inducing points: {}".format(pgpr.inducing_variable.num_inducing))
    print("ELBO = {:.6f}".format(pgpr.elbo()))
    print("Accuracy = {:.6f}".format(compute_accuracy(Y, pgpr.predict_f(X)[0])))
    print("NLL = {:.6f}".format(compute_nll(Y, pgpr.predict_f(X)[0])))


if __name__ == '__main__':
    X, Y, NUM_INDUCING, SVGP_ITERS, PGPR_ITERS, GREEDY_THRESHOLD = load_ionosphere()
    X = standardise_features(X)

    run_experiment()
