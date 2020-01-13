# mlr3learners.lightgbm (!!!under development!!!)

<!-- badges: start -->
[![pipeline status](https://gitlab.com/kapsner/mlr3learners-lightgbm/badges/master/pipeline.svg)](https://gitlab.com/kapsner/mlr3learners-lightgbm/commits/master)
[![coverage report](https://gitlab.com/kapsner/mlr3learners-lightgbm/badges/master/coverage.svg)](https://gitlab.com/kapsner/mlr3learners-lightgbm/commits/master)
<!-- badges: end -->
 
[mlr3learners.lightgbm](https://github.com/kapsner/mlr3learners.lightgbm) brings the [LightGBM gradient booster](https://lightgbm.readthedocs.io) to the [mlr3](https://github.com/mlr-org/mlr3) framework by using the [official lightgbm R implementation](https://github.com/microsoft/LightGBM/tree/master/R-package). 

# Features 

* integrated native CV before the actual model training to find the optimal `nrounds` for the given training data and parameter set  
* GPU support  

# Installation 

Before you can install the `mlr3learners.lightgbm` package, you need to install the lightgbm R package according to [its documentation](https://github.com/microsoft/LightGBM/blob/master/R-package/README.md) (this is necessary since lightgbm is neither on CRAN nor installable via `devtools::install_github`).  

```bash
git clone --recursive --branch stable --depth 1 https://github.com/microsoft/LightGBM
cd LightGBM && \
Rscript build_r.R
```

If the lightgbm R package is installed, you can continue and install the [mlr3learners.lightgbm](https://github.com/kapsner/mlr3learners.lightgbm) R package:

```r
install.packages("devtools")
devtools::install_github("kapsner/mlr3learners.lightgbm")
```

# Example

```r
library(mlr3)
task = mlr3::tsk("iris")
learner = mlr3::lrn("classif.lightgbm")

learner$early_stopping_rounds <- 1000
learner$num_boost_round <- 5000

learner$param_set$values <- list(
  "objective" = "multiclass",
  "learning_rate" = 0.01,
  "seed" = 17L
)

learner$train(task, row_ids = 1:120)
predictions <- learner$predict(task, row_ids = 121:150)
```

For further information and examples, please view the `mlr3learners.lightgbm` package vignettes and the [mlr3book](https://mlr3book.mlr-org.com/index.html).  

# GPU acceleration

The `mlr3learners.lightgbm` can also be used with lightgbm's GPU compiled version.

To install the lightgbm R package with GPU support, execute the following commands ([lightgbm manual](https://github.com/microsoft/LightGBM/blob/master/R-package/README.md)):

```bash
git clone --recursive --branch stable --depth 1 https://github.com/microsoft/LightGBM
cd LightGBM && \
sed -i -e 's/use_gpu <- FALSE/use_gpu <- TRUE/g' R-package/src/install.libs.R && \
Rscript build_r.R
```

In order to use the GPU acceleration, the parameter `device_type = "gpu"` (default: "cpu") needs to be set. According to the [LightGBM parameter manual](https://lightgbm.readthedocs.io/en/latest/Parameters.html), 'it is recommended to use the smaller `max_bin` (e.g. 63) to get the better speed up'. 

```r
learner$param_set$values <- list(
  "objective" = "multiclass",
  "learning_rate" = 0.01,
  "seed" = 17L,
  "device_type" = "gpu",
  "max_bin" = 63L
)
```

All other steps are similar to the workflow without GPU support. 

The GPU support has been tested in a [Docker container](https://github.com/kapsner/docker_images/blob/master/Rdatascience/rdsc_gpu/Dockerfile) running on a Linux 19.10 host, Intel i7, 16 GB RAM, an NVIDIA(R) RTX 2060, CUDA(R) 10.2 and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker). 

# More Infos:

- Microsoft's LightGBM: https://lightgbm.readthedocs.io/en/latest/
- mlr3: https://github.com/mlr-org/mlr3

