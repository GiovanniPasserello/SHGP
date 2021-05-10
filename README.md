# SHGP

SHGP is a GPflow-based software package for Sparse Heteroscedastic Gaussian Processes. It was developed as an accompaniment to my final-year master's thesis at Imperial College London - 'Heteroscedastic Inducing Point Selection for Gaussian Process Classification'. The focus of this thesis is to compute an SGPR-style model for Gaussian process classification, in which gradient-based optimisation of the variational distribution over inducing points is not required.

### Contributions

* An implementation of a heteroscedastic Gaussian process regression model in which the variational distribution is analytically optimised. This is an extension of the collapsed model derived in [Titsias 2009].
* An implementation of a novel heterscedastic Gaussian process classification model in which the variational distribution is analytically optimised. This model employs Pólya-Gamma data augmentation to form a conditionally conjugate lower bound to the likelihood. Importantly, the effective likelihood is an unnormalised Gaussian distribution with heteroscedastic variance.
* An implementation of a novel heteroscedastic inducing point selection method for both Gaussian process regression and classification. The method is a heteroscedastic extension of [Burt et al. 2020], but its application to classification is entirely novel and is made possible by our collapsed model.
* A robust procedure for stable Cholesky decompositions with dynamic jitter selection.
* Multiple experiments which investigate the benefits of the above contributions. In particular, we show that our collapsed Gaussian process classification model paired with our inducing point selection method permits much sparser models than alternative schemes. We also show that the selection method allows the number of inducing points required for a sufficiently accurate approximation to be automatically determined by the model.

### License

This project is licensed under the MIT License - see [LICENSE](https://github.com/GiovanniPasserello/SHGP/blob/main/LICENSE) for details.
