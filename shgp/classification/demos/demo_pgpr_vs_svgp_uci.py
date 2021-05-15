import numpy as np
import tensorflow as tf

np.random.seed(0)
tf.random.set_seed(0)


# TODO: Better convergence guarantees of training PGPR
# TODO: Experiments - choose M from sparsity, or cap at 500 for large datasets. Report ACCURACY, ELBO and NLL.
#       Run 5-10 times and average. Bern GO, PGPR GO, PGPR GV, PGPR HGV.
# TODO: Add experiment contrasting PGPR HGV to PGPR GO initialised at HGV - how much performance are we missing?


def load_fertility():
    # https://archive.ics.uci.edu/ml/datasets/Fertility
    dataset = "../../data/fertility.txt"

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


def load_crabs():
    # https://datarepository.wolframcloud.com/resources/Sample-Data-Crab-Measures
    dataset = "../../data/datasets/crabs.csv"

    data = np.loadtxt(dataset, delimiter=",", skiprows=1)
    X = data[:, 1:]
    X = np.delete(X, 1, axis=1)  # remove a column of indices (6 dimensions)
    Y = data[:, 0].reshape(-1, 1)

    # TODO: Sparsity experiment?
    # Interesting that Bernoulli GO has a significantly worse ELBO than PGPR. With
    # an unconstrained kernel Bernoulli GO gets -43.075003 and 1.00 accuracy, which
    # is still a lower ELBO than PGPR - why is this?
    # With an unconstrained kernel PGPR achieves an ELBO of -30.343163

    # TODO: Important
    # This is one of the key datasets where PGPR completely outperforms SVGP!
    # Even with different seeds or more optimisation steps, PGPR always outperforms
    # SVGP with Bern. Why is this?

    # TODO: Maybe put an asterisk next to this experiment and mark as unconstrained.
    #       Because SVGP fails to converge with any constrained kernel.

    NUM_INDUCING = 28  # quicker with 28 than with 200
    BERN_ITERS = 200  # best with 200: -112.733273, acc: 0.875000 (with 28: -112.174307, acc: 0.895000)
    PGPR_ITERS = (5, 25, 5)  # best with 200: -37.638322, acc: 1.00
    GREEDY_THRESHOLD = 1e-1  # (early stops at 28): -37.665957, acc: 1.00 (can move to 5e-1, still acc: 1.00)

    return X, Y, NUM_INDUCING, BERN_ITERS, PGPR_ITERS, GREEDY_THRESHOLD


def load_ionosphere():
    # https://archive.ics.uci.edu/ml/datasets/ionosphere
    dataset = "../../data/ionosphere.txt"

    # TODO: Sparisity experiment?
    data = np.loadtxt(dataset, delimiter=",")
    X = data[:, :-1]
    X = np.delete(X, 1, axis=1)  # remove a column of zeros (33 dimensions)
    Y = data[:, -1].reshape(-1, 1)

    # In this case, Bernoulli GO seems to perform much better than PGPR.
    # This is likely because of the small dataset size.

    NUM_INDUCING = 351  # quicker with 156 than with 351
    BERN_ITERS = 100  # best with 351: -100.021138 (with 156: -107.370414)
    PGPR_ITERS = (5, 25, 5)  # best with 351: -126.962021
    GREEDY_THRESHOLD = 1  # (early stops at 156): -127.549833

    return X, Y, NUM_INDUCING, BERN_ITERS, PGPR_ITERS, GREEDY_THRESHOLD


def load_breast_cancer():
    # https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
    dataset = "../../data/breast-cancer-diagnostic.txt"

    data = np.loadtxt(dataset, delimiter=",")
    X = data[:, 2:]
    Y = data[:, 1].reshape(-1, 1)

    # TODO: This might be a good example to discuss PG sometimes beating Bernoulli.
    # TODO: Sparisity experiment?
    # The comparison here may be that:
    # With 569 inducing points fixed for Bernoulli, the ELBO is -178.680281. Interestingly,
    # the PGPR ELBO is -74.876002 even though both models receive the same fixed set of data
    # points - the full training set. Why is this? This is very interesting to explore!
    # SVGP convergence is very noisy and sometimes converges rapidly, sometimes slowly.
    # PGPR is quick to converge, but due to my convergence criterion it can cycle for a while.
    # (perhaps we need a better convergence setup than the current absolute difference check - the number of cycles matters).

    # In a way, the fact that Bernoulli achieves -178.680281 (a suboptimal value) with the full training set
    # emphasises the unreliability of gradient-based methods (changing the random seed allows Bernoulli
    # to converge). One large benefit of greedy variance is that there is no stochasticity in the
    # selection process. The same points will be chosen no matter the setup. The only thing that does
    # change when using greedy variance is the optimisation of hyperparameters (which is often stochastic).
    # The added randomness of SVGP Bernoulli makes it quite unreliable - some seeds converge in 1 second,
    # other seeds take 15 seconds. Or is this more to do with the kernel being constrained? see below:

    # TODO: The effect of using a constrained SE kernel is large. It makes the difference between
    # Bernoulli converging to its optimal values or not. The problem of -178.680281 Bernoulli is removed
    # when we use the standard SE kernel with NUM_INDUCING=569. In fact it achieves -59.074076 but also
    # takes 45.37 seconds. Also when using the unconstrained kernel for NUM_INDUCING=59 for Bernoulli
    # we get an ELBO of -66.578607.

    # TODO: Important thoughts:
    # So when do we constrain the kernel, and when do we not??? Is one definitely better? For consistency
    # we should probably use constrained SE kernels everywhere to avoid Cholesky errors and make sure to
    # mention this in the report. PGPR isn't affected in this example, but Bernoulli is significantly affected.

    # TODO: Other thoughts
    # if we change the seed to 42 with a constrained kernel however:
    #       Bernoulli achieves an ELBO of -86.431603 with 54 datapoints and -60.998253 with 569
    # Interestingly inconsistent performance, and sometimes significantly worse than PGPR - why is this?
    # Could this perhaps be due to a very noisy dataset - is this an observed downside to SVGP/gradient methods?

    # TODO: These results definitely need to be averaged over many runs (see above)
    NUM_INDUCING = 569  # quicker with 54 than with 569
    BERN_ITERS = 200  # best with 569: -178.680281 (with 54): -253.199324)  # why such catastrophic performance?
    PGPR_ITERS = (5, 25, 5)  # best with 569: -74.876002
    GREEDY_THRESHOLD = 1  # (early stops at 54): -75.173682

    return X, Y, NUM_INDUCING, BERN_ITERS, PGPR_ITERS, GREEDY_THRESHOLD


def load_pima():
    # http://networkrepository.com/pima-indians-diabetes.php
    dataset = "../../data/pima-diabetes.csv"

    data = np.loadtxt(dataset, delimiter=",")
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1)

    # TODO: Sparisity experiment?
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


def load_twonorm():
    # https://www.openml.org/d/1507
    dataset = "../../data/twonorm.csv"

    data = np.loadtxt(dataset, delimiter=",", skiprows=1)  # skip headers
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1) - 1

    # TODO: Reiteration from 'magic'
    # TODO: Large comparison, how do we reliably find results when this takes so long to run?
    # Maybe smaller datasets contain more feasible insight - too many variables to consider?

    # TODO: We can prune the dataset if needed
    # TODO: Maybe we should do full runs if possible - but the compute isn't possible if M is large?
    #       Probably possible for small-scale tests, e.g. 100 inducing points

    # TODO: Only computationally feasible to try a small number of inducing points.
    # This means that we cannot realistically do a sparsity experiment, but perhaps we can
    # do a metric experiment with 100 inducing points. (e.g. error bars on ELBO/ACC)

    # TODO: Important
    # This is a very interesting comparison of Bern vs PGPR.
    # It seems that in general, especially for small M, Bern outperforms PGPR (especially M=10)
    # The ELBO of SVGP however is far lower than the ELBO of PGPR for M=200. Why is it in this case
    # that the accuracy is almost identical, but the ELBO is so wildly different. In general we would
    # expect similar or worse ELBOs from PGPR.
    NUM_INDUCING = 100
    BERN_ITERS = 200  # best with 100: -5031.590838, accuracy: 0.971757 (in 1.79 seconds)
    PGPR_ITERS = (5, 25, 10)  # best with 100: -5129.289144, accuracy: 0.961081 (in 10.85 seconds)
    GREEDY_THRESHOLD = 0

    # TODO: This might be a good dataset for small-scale sparsity experiments

    # 200 - mixed performance - far better ELBO, slightly better ACC, far better NLL, worse time
    # bernoulli: ELBO=-5062.441466, ACC=0.975000, NLL=0.683428, TIME=3.50
    # pgpr: ELBO=-505.188846, ACC=0.979459, NLL=0.056077, TIME=90.51 (but it cycled and converged ~25 seconds)
    # 10 - worse performance
    # bernoulli: ELBO=-5128.682356, ACC=0.918784, NLL=0.693055, TIME=1.12
    # pgpr: ELBO=-5129.289144, ACC=0.778649, NLL=0.693147, TIME=3.20

    return X, Y, NUM_INDUCING, BERN_ITERS, PGPR_ITERS, GREEDY_THRESHOLD


def load_ringnorm():
    # https://www.openml.org/d/1496
    dataset = "../../data/ringnorm.csv"

    data = np.loadtxt(dataset, delimiter=",", skiprows=1)  # skip headers
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1) - 1

    # TODO: Reiteration from 'magic'
    # TODO: Large comparison, how do we reliably find results when this takes so long to run?
    # Maybe smaller datasets contain more feasible insight - too many variables to consider?

    # TODO: We can prune the dataset if needed
    # TODO: Maybe we should do full runs if possible - but the compute isn't possible if M is large?
    #       Probably possible for small-scale tests, e.g. 100 inducing points

    # TODO: Only computationally feasible to try a small number of inducing points.
    # This means that we cannot realistically do a sparsity experiment, but perhaps we can
    # do a metric experiment with 100 inducing points. (e.g. error bars on ELBO/ACC)

    # TODO: Important
    # This is one of the key datasets where PGPR almost always outperforms SVGP!
    # Training of SVGP is very unstable and depends on the random seed
    # - Bern often barely gets above 50% accuracy and sometimes unconstraining the kernel doesn't make a difference!
    # - This is another downside showing the inconsistency/fragility of grad optim Bern & SVGP
    # - When Bernoulli works, it works very well!
    NUM_INDUCING = 100
    BERN_ITERS = 200  # best with 100: -4539.416618, accuracy: 0.504865  # catastophic failure again?
    PGPR_ITERS = (5, 25, 10)  # best with 100: -2716.269190, accuracy: 0.919459 (but one iteration was -2284.628780 - better convergence criterion, or just report max?)
    GREEDY_THRESHOLD = 0

    # 200 - far superior performance
    # bernoulli: ELBO=-4872.606096, ACC=0.505946, NLL=0.656005, TIME=3.84 (with different seed: ELBO=-853.335810, ACC=0.984730)
    # pgpr: ELBO=-1421.169011, ACC=0.978108, NLL=0.076769, TIME=170.78 (but it cycled for a long time)
    # 20 - far superior performance (10 had inversion errors for Bernoulli)
    # bernoulli: ELBO=-5000.689376, ACC=0.505270, NLL=0.674485, TIME=0.81
    # pgpr: ELBO=-4176.587485, ACC=0.736216, NLL=0.551947, TIME=25.40 (but it cycled for a long time)

    return X, Y, NUM_INDUCING, BERN_ITERS, PGPR_ITERS, GREEDY_THRESHOLD


def load_magic():
    # https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope
    dataset = "../../data/magic.txt"

    data = np.loadtxt(dataset, delimiter=",")
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1)

    # TODO: Large comparison, how do we reliably find results when this takes so long to run?
    # Maybe smaller datasets contain more feasible insight - too many variables to consider?
    # Here it is very important to have good convergence guarantees - the current absolute
    # difference check is not good enough. Perhaps we can check if the ELBO increases, but
    # is that reliable - what if we reach a local minimum which we might leave later?

    # TODO: We can prune the dataset if needed
    # TODO: Maybe we should do full runs if possible - but the compute is not possible?
    #       Probably possible for small-scale tests, e.g. 100 inducing points

    # TODO: Only computationally feasible to try a small number of inducing points.
    # This means that we cannot realistically do a sparsity experiment (unless we prune).
    # We can do a metric experiment with 100 inducing points. (e.g. error bars on ELBO/ACC)
    NUM_INDUCING = 100  # 200
    BERN_ITERS = 200  # 100
    PGPR_ITERS = (5, 25, 10)  # This slightly benefits from being larger than this, but becomes slow
    GREEDY_THRESHOLD = 0  # 100

    return X, Y, NUM_INDUCING, BERN_ITERS, PGPR_ITERS, GREEDY_THRESHOLD


def load_electricity():
    # https://datahub.io/machine-learning/electricity
    dataset = "../../data/electricity.csv"

    data = np.loadtxt(dataset, delimiter=",", skiprows=1)  # skip headers
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1, 1)

    # TODO: Reiteration from 'magic'
    # TODO: Large comparison, how do we reliably find results when this takes so long to run?
    # Maybe smaller datasets contain more feasible insight - too many variables to consider?

    # TODO: We can prune the dataset if needed
    # TODO: Maybe we should do full runs if possible - but the compute isn't possible if M is large?
    #       Probably possible for small-scale tests, e.g. 100 inducing points

    # TODO: Only computationally feasible to try a small number of inducing points.
    # This means that we cannot realistically do a sparsity experiment, but perhaps we can
    # do a metric experiment with 100 inducing points. (e.g. error bars on ELBO/ACC)
    NUM_INDUCING = 100  # 100
    BERN_ITERS = 200  # best with 100: -20891.910067, accuracy: 0.789217
    PGPR_ITERS = (5, 25, 10)  # best with 100: -21090.760171, accuracy: 0.784936
    GREEDY_THRESHOLD = 0

    # 200 - very close performance
    # bernoulli: ELBO=-20638.401637, ACC=0.795043, NLL=0.441826, TIME=213.12
    # pgpr: ELBO=-20646.108759, ACC=0.791159, NLL=0.444059, TIME=764.12 (but this cycled - converged ~150 seconds)
    # 10 - worse performance
    # bernoulli: ELBO=-21872.580557, ACC=0.778469, NLL=0.467994, TIME=25.71
    # pgpr: ELBO=-22990.888027, ACC=0.755032, NLL=0.505194, TIME=51.28 (this cycled - converged ~15 seconds)

    return X, Y, NUM_INDUCING, BERN_ITERS, PGPR_ITERS, GREEDY_THRESHOLD
