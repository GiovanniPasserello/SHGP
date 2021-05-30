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


# TODO: Worst candidate
# Results from full train PGPR, then pass Z for full train SVGP (no outer reinit):
#   It appears that this is a very adversarial initialisation for SVGP - it performs badly on most datasets.
# Crabs
# PGPR-SVGP Distribution: (beats pgpr hgv)
# ELBO - max: -22.474055, min: -24.872674, median: -23.607021, mean: -23.531741, std: 0.756435.
# ACC  - max: 1.000000, min: 1.000000, median: 1.000000, mean: 1.000000, std: 0.000000.
# NLL  - max: 0.020949, min: 0.000498, median: 0.004298, mean: 0.007684, std: 0.007415.
# SVGP Distribution:
# ELBO - max: -22.886637, min: -28.304666, median: -25.128148, mean: -25.541968, std: 1.726059.
# ACC  - max: 1.000000, min: 1.000000, median: 1.000000, mean: 1.000000, std: 0.000000.
# NLL  - max: 0.027319, min: 0.002898, median: 0.005168, mean: 0.010180, std: 0.008736.

# Banana
# PGPR-SVGP Distribution: (beats pgpr hgv)
# ELBO - max: -92.571297, min: -126.921554, median: -98.976606, mean: -101.615531, std: 8.905149.
# ACC  - max: 0.975000, min: 0.850000, median: 0.912500, mean: 0.910000, std: 0.040620.
# NLL  - max: 0.361546, min: 0.060645, median: 0.206494, mean: 0.202408, std: 0.084123.
# SVGP Distribution:
# ELBO - max: -94.009353, min: -103.955948, median: -99.347811, mean: -99.675996, std: 2.875932.
# ACC  - max: 0.975000, min: 0.875000, median: 0.912500, mean: 0.912500, std: 0.032113.
# NLL  - max: 0.353792, min: 0.059819, median: 0.206814, mean: 0.195779, std: 0.077877.

# Heart
# PGPR-SVGP Distribution: (loses to pgpr hgv)
# ELBO - max: -137.371980, min: -167.637171, median: -164.491385, mean: -160.074340, std: 9.183553.
# ACC  - max: 0.962963, min: 0.703704, median: 0.796296, mean: 0.807407, std: 0.077336.
# NLL  - max: 0.683176, min: 0.384606, median: 0.621359, mean: 0.594673, std: 0.089176.
# SVGP Distribution:
# ELBO - max: -101.100542, min: -148.365596, median: -104.327670, mean: -113.113321, std: 17.620058.
# ACC  - max: 1.000000, min: 0.740741, median: 0.870370, mean: 0.874074, std: 0.068693.
# NLL  - max: 0.554373, min: 0.135649, median: 0.356916, mean: 0.378400, std: 0.127785.

# Ionosphere
# PGPR-SVGP Distribution: (loses to pgpr hgv)
# ELBO - max: -110.604867, min: -218.341364, median: -213.777281, mean: -182.978958, std: 47.016947.
# ACC  - max: 0.972222, min: 0.305556, median: 0.722222, mean: 0.733333, std: 0.193330.
# NLL  - max: 0.693147, min: 0.114627, median: 0.672774, mean: 0.510393, std: 0.234660.
# SVGP Distribution:
# ELBO - max: -103.506418, min: -186.125799, median: -108.915725, mean: -116.489050, std: 23.317403.
# ACC  - max: 0.972222, min: 0.777778, median: 0.916667, mean: 0.902778, std: 0.057265.
# NLL  - max: 0.583838, min: 0.117098, median: 0.244106, mean: 0.271088, std: 0.123693.

# Breast Cancer
# PGPR-SVGP Distribution: (loses to pgpr hgv)
# ELBO - max: -52.929865, min: -338.086797, median: -308.095943, mean: -250.572178, std: 108.847563.
# ACC  - max: 0.982456, min: 0.631579, median: 0.771930, mean: 0.789474, std: 0.123557.
# NLL  - max: 0.614071, min: 0.086798, median: 0.480590, mean: 0.408090, std: 0.188616.
# SVGP Distribution:
# ELBO - max: -48.775009, min: -235.309456, median: -55.791680, mean: -76.249760, std: 54.154673.
# ACC  - max: 1.000000, min: 0.877193, median: 0.982456, mean: 0.961404, std: 0.039072.
# NLL  - max: 0.371899, min: 0.025845, median: 0.088319, mean: 0.128202, std: 0.117569.

# Pima
# PGPR-SVGP Distribution: (loses to pgpr hgv)
# ELBO - max: -339.236903, min: -438.105658, median: -346.264417, mean: -367.624296, std: 32.415953.
# ACC  - max: 0.844156, min: 0.688312, median: 0.779221, mean: 0.771429, std: 0.050699.
# NLL  - max: 0.599243, min: 0.376020, median: 0.461824, mean: 0.468597, std: 0.064765.
# SVGP Distribution:
# ELBO - max: -337.697290, min: -346.814701, median: -341.609822, mean: -342.150919, std: 2.985189.
# ACC  - max: 0.844156, min: 0.753247, median: 0.798701, mean: 0.800000, std: 0.029156.
# NLL  - max: 0.491475, min: 0.364078, median: 0.441640, mean: 0.431878, std: 0.040019.

# TODO: Second best candidate
# Results from SVGP gv reinit (very slow training SVGP multiple times):
# Crabs
# PGPR-SVGP Distribution: (beats pgpr hgv)
# ELBO - max: -21.543007, min: -23.815687, median: -22.786379, mean: -22.667314, std: 0.607973.
# ACC  - max: 1.000000, min: 1.000000, median: 1.000000, mean: 1.000000, std: 0.000000.
# NLL  - max: 0.024279, min: 0.000652, median: 0.002922, mean: 0.005936, std: 0.006953.
# SVGP Distribution:
# ELBO - max: -23.827066, min: -41.535403, median: -26.978224, mean: -29.290871, std: 5.669795.
# ACC  - max: 1.000000, min: 1.000000, median: 1.000000, mean: 1.000000, std: 0.000000.
# NLL  - max: 0.057917, min: 0.003611, median: 0.017666, mean: 0.024588, std: 0.020781.

# Banana
# PGPR-SVGP Distribution: (beats pgpr hgv)
# ELBO - max: -93.828072, min: -103.638061, median: -99.580822, mean: -98.977024, std: 3.031143.
# ACC  - max: 0.975000, min: 0.850000, median: 0.900000, mean: 0.905000, std: 0.041533.
# NLL  - max: 0.334486, min: 0.058948, median: 0.164676, mean: 0.185157, std: 0.084736.
# SVGP Distribution:
# ELBO - max: -95.237144, min: -103.654768, median: -100.773131, mean: -100.080934, std: 2.701968.
# ACC  - max: 0.975000, min: 0.825000, median: 0.900000, mean: 0.905000, std: 0.047170.
# NLL  - max: 0.324890, min: 0.060560, median: 0.164710, mean: 0.189113, std: 0.086033.

# Heart
# PGPR-SVGP Distribution: (draws with pgpr hgv)
# ELBO - max: -97.146215, min: -107.591011, median: -103.481362, mean: -103.441737, std: 3.134346.
# ACC  - max: 0.962963, min: 0.703704, median: 0.851852, mean: 0.848148, std: 0.071146.
# NLL  - max: 0.637126, min: 0.220092, median: 0.368424, mean: 0.374192, std: 0.124435.
# SVGP Distribution:
# ELBO - max: -97.055776, min: -131.976801, median: -103.310292, mean: -105.957719, std: 9.180429.
# ACC  - max: 0.962963, min: 0.703704, median: 0.870370, mean: 0.851852, std: 0.072198.
# NLL  - max: 0.637369, min: 0.220284, median: 0.393066, mean: 0.387377, std: 0.120376.

# Ionosphere
# PGPR-SVGP Distribution: (loses to pgpr hgv)
# ELBO - max: -101.459150, min: -148.837753, median: -109.091908, mean: -111.581688, std: 13.078458.
# ACC  - max: 0.972222, min: 0.805556, median: 0.875000, mean: 0.877778, std: 0.051520.
# NLL  - max: 0.539254, min: 0.112306, median: 0.369333, mean: 0.348497, std: 0.153782.
# SVGP Distribution:
# ELBO - max: -96.647093, min: -186.125799, median: -107.442977, mean: -114.076699, std: 24.391957.
# ACC  - max: 0.972222, min: 0.777778, median: 0.875000, mean: 0.886111, std: 0.054786.
# NLL  - max: 0.583838, min: 0.089269, median: 0.360997, mean: 0.332865, std: 0.155977.

# Breast Cancer
# PGPR-SVGP Distribution: (loses to pgpr hgv)
# ELBO - max: -44.181989, min: -243.949769, median: -56.997358, mean: -92.489327, std: 74.650825.
# ACC  - max: 1.000000, min: 0.807018, median: 0.982456, mean: 0.961404, std: 0.055367.
# NLL  - max: 0.426944, min: 0.017149, median: 0.069778, mean: 0.146026, std: 0.139730.
# SVGP Distribution:
# ELBO - max: -43.302989, min: -285.304134, median: -58.468338, mean: -96.990956, std: 81.910065.
# ACC  - max: 1.000000, min: 0.842105, median: 0.982456, mean: 0.959649, std: 0.048397.
# NLL  - max: 0.504728, min: 0.021801, median: 0.080290, mean: 0.170945, std: 0.175113.

# Pima
# PGPR-SVGP Distribution: (loses to pgpr hgv)
# ELBO - max: -328.662462, min: -347.229481, median: -339.846917, mean: -339.005043, std: 5.008193.
# ACC  - max: 0.844156, min: 0.675325, median: 0.785714, mean: 0.771429, std: 0.054854.
# NLL  - max: 0.618986, min: 0.365728, median: 0.463492, mean: 0.476286, std: 0.067422.
# SVGP Distribution:
# ELBO - max: -328.613480, min: -346.972261, median: -339.729831, mean: -338.769933, std: 5.115560.
# ACC  - max: 0.857143, min: 0.675325, median: 0.785714, mean: 0.774026, std: 0.055769.
# NLL  - max: 0.622665, min: 0.365662, median: 0.462153, mean: 0.476838, std: 0.068235.


# TODO: Best candidate - this appears to perform the best of all three options.
# Results from hgv reinit (passing SVGP kernel params to PGPR, and initialising SVGP at q(u) from PGPR)
# Crabs
# PGPR-SVGP Distribution: (beats pgpr hgv)
# ELBO - max: -22.023821, min: -25.010794, median: -23.196518, mean: -23.417317, std: 1.019762.
# ACC  - max: 1.000000, min: 1.000000, median: 1.000000, mean: 1.000000, std: 0.000000.
# NLL  - max: 0.020154, min: 0.000588, median: 0.006871, mean: 0.010130, std: 0.007919.
# SVGP Distribution:
# ELBO - max: -23.386512, min: -41.535403, median: -26.648172, mean: -28.626344, std: 5.617346.
# ACC  - max: 1.000000, min: 1.000000, median: 1.000000, mean: 1.000000, std: 0.000000.
# NLL  - max: 0.051004, min: 0.004487, median: 0.023362, mean: 0.024276, std: 0.015086.

# Banana
# PGPR-SVGP Distribution: (beats pgpr hgv)
# ELBO - max: -94.899695, min: -103.286065, median: -98.565541, mean: -98.653077, std: 2.744259.
# ACC  - max: 0.975000, min: 0.850000, median: 0.900000, mean: 0.905000, std: 0.040000.
# NLL  - max: 0.331383, min: 0.061284, median: 0.195053, mean: 0.195380, std: 0.082114.
# SVGP Distribution:
# ELBO - max: -95.778227, min: -103.654768, median: -99.521598, mean: -99.599943, std: 2.600769.
# ACC  - max: 0.975000, min: 0.850000, median: 0.887500, mean: 0.902500, std: 0.041003.
# NLL  - max: 0.341103, min: 0.060560, median: 0.195393, mean: 0.196742, std: 0.083398.

# Heart
# PGPR-SVGP Distribution: (loses to pgpr hgv)
# ELBO - max: -95.941532, min: -106.995744, median: -103.814808, mean: -102.859724, std: 3.132896.
# ACC  - max: 0.962963, min: 0.666667, median: 0.851852, mean: 0.837037, std: 0.081481.
# NLL  - max: 0.676240, min: 0.233359, median: 0.360904, mean: 0.397462, std: 0.124621.
# SVGP Distribution:
# ELBO - max: -95.723071, min: -148.470315, median: -105.605896, mean: -119.624871, std: 21.714117.
# ACC  - max: 0.962963, min: 0.666667, median: 0.833333, mean: 0.825926, std: 0.089274.
# NLL  - max: 0.685729, min: 0.233735, median: 0.491596, mean: 0.470618, std: 0.124970.

# Ionosphere
# PGPR-SVGP Distribution: (beats pgpr hgv)
# ELBO - max: -105.065745, min: -109.864717, median: -108.337373, mean: -107.984972, std: 1.498420.
# ACC  - max: 0.916667, min: 0.861111, median: 0.888889, mean: 0.897222, std: 0.017786.
# NLL  - max: 0.527850, min: 0.203504, median: 0.297755, mean: 0.312896, std: 0.082535.
# SVGP Distribution:
# ELBO - max: -104.496760, min: -109.150004, median: -106.618944, mean: -106.517940, std: 1.224817.
# ACC  - max: 0.944444, min: 0.861111, median: 0.888889, mean: 0.900000, std: 0.022222.
# NLL  - max: 0.464050, min: 0.160188, median: 0.284794, mean: 0.286295, std: 0.072999.

# Breast Cancer
# PGPR-SVGP Distribution: (beats pgpr hgv)
# ELBO - max: -53.092124, min: -58.996227, median: -56.068862, mean: -56.175013, std: 1.990107.
# ACC  - max: 1.000000, min: 0.929825, median: 0.973684, mean: 0.971930, std: 0.021053.
# NLL  - max: 0.129703, min: 0.021079, median: 0.046592, mean: 0.065499, std: 0.038660.
# SVGP Distribution:
# ELBO - max: -52.797070, min: -290.736351, median: -57.312985, mean: -101.327688, std: 89.319011.
# ACC  - max: 1.000000, min: 0.859649, median: 0.973684, mean: 0.963158, std: 0.041107.
# NLL  - max: 0.504664, min: 0.015748, median: 0.066597, mean: 0.143335, std: 0.164690.

# Pima
# PGPR-SVGP Distribution: (loses to pgpr hgv)
# ELBO - max: -327.529158, min: -346.009926, median: -335.378425, mean: -336.743647, std: 5.481372.
# ACC  - max: 0.870130, min: 0.675325, median: 0.720779, mean: 0.738961, std: 0.057334.
# NLL  - max: 0.661164, min: 0.387636, median: 0.524282, mean: 0.510168, std: 0.078058.
# SVGP Distribution:
# ELBO - max: -327.005077, min: -345.421797, median: -335.347458, mean: -336.421096, std: 5.545441.
# ACC  - max: 0.870130, min: 0.662338, median: 0.714286, mean: 0.737662, std: 0.060302.
# NLL  - max: 0.658934, min: 0.388002, median: 0.527262, mean: 0.509732, std: 0.077754.
