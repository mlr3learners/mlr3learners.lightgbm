# mlr3learners.lightgbm

<!-- badges: start -->
[![R CMD Check via {tic}](https://github.com/mlr3learners/mlr3learners.lightgbm/workflows/R%20CMD%20Check%20via%20{tic}/badge.svg?branch=master)](https://github.com/mlr3learners/mlr3learners.lightgbm/actions)
[![Parameter Check](https://github.com/mlr3learners/mlr3learners.lightgbm/workflows/Parameter%20Check/badge.svg?branch=master)](https://github.com/mlr3learners/mlr3learners.lightgbm/actions)
[![linting](https://github.com/mlr3learners/mlr3learners.lightgbm/workflows/lint/badge.svg?branch=master)](https://github.com/mlr3learners/mlr3learners.lightgbm/actions)
[![codecov](https://codecov.io/gh/mlr3learners/mlr3learners.lightgbm/branch/master/graph/badge.svg)](https://codecov.io/gh/mlr3learners/mlr3learners.lightgbm)
[![pipeline status](https://gitlab.com/kapsner/mlr3learners-lightgbm/badges/master/pipeline.svg)](https://gitlab.com/kapsner/mlr3learners-lightgbm/commits/master)
[![coverage report](https://gitlab.com/kapsner/mlr3learners-lightgbm/badges/master/coverage.svg)](https://gitlab.com/kapsner/mlr3learners-lightgbm/commits/master)
<!-- badges: end -->
 
[mlr3learners.lightgbm](https://github.com/kapsner/mlr3learners.lightgbm) brings the [LightGBM gradient booster](https://lightgbm.readthedocs.io) to the [mlr3](https://github.com/mlr-org/mlr3) framework by using the [official lightgbm R implementation](https://github.com/microsoft/LightGBM/tree/master/R-package). 

# Features 

* integrated learner-native cross-validation (CV) using `lgb.cv` before the actual model training to find the optimal `num_iterations` for the given training data and parameter set  
* GPU support  

# Installation 

As of lightgbm version 3.0.0, you can install the [mlr3learners.lightgbm](https://github.com/mlr3learners/mlr3learners.lightgbm) R package with:

```r
install.packages("remotes")
# stable version:
remotes::install_github("mlr3learners/mlr3learners.lightgbm")
# dev version:
# remotes::install_github("mlr3learners/mlr3learners.lightgbm@development")
```

# Example

```r
library(mlr3)
library(mlr3learners.lightgbm)
task = mlr3::tsk("iris")
learner = mlr3::lrn("classif.lightgbm", objective = "multiclass")

learner$param_set$values = mlr3misc::insert_named(
  learner$param_set$values,
    list(
    "early_stopping_round" = 10,
    "learning_rate" = 0.1,
    "seed" = 17L,
    "metric" = "multi_logloss",
    "num_iterations" = 100,
    "num_class" = 3
  )
)

learner$train(task, row_ids = 1:120)
predictions = learner$predict(task, row_ids = 121:150)
```

For further information and examples, please view the `mlr3learners.lightgbm` [package vignettes](vignettes/) and the [mlr3book](https://mlr3book.mlr-org.com/index.html).  

# GPU acceleration

The `mlr3learners.lightgbm` can also be used with lightgbm's GPU compiled version.

To install the lightgbm R package with GPU support, execute the following commands ([lightgbm manual](https://github.com/microsoft/LightGBM/blob/master/R-package/README.md)):

```bash
git clone --recursive --branch stable --depth 1 https://github.com/microsoft/LightGBM
cd LightGBM && \
Rscript build_r.R --use-gpu
```

In order to use the GPU acceleration, the parameter `device_type = "gpu"` (default: "cpu") needs to be set. According to the [LightGBM parameter manual](https://lightgbm.readthedocs.io/en/latest/Parameters.html), 'it is recommended to use the smaller `max_bin` (e.g. 63) to get the better speed up'. 

```r
learner$param_set$values = mlr3misc::insert_named(
  learner$param_set$values,
  list(
    "objective" = "multiclass",
    "device_type" = "gpu",
    "max_bin" = 63L,
    "early_stopping_round" = 10,
    "learning_rate" = 0.1,
    "seed" = 17L,
    "metric" = "multi_logloss",
    "num_iterations" = 100,
    "num_class" = 3
  )
)
```

All other steps are similar to the workflow without GPU support. 

The GPU support has been tested in a [Docker container](https://github.com/kapsner/docker_images/blob/master/Rdatascience/image_rdsc_gpu/Dockerfile) running on a Linux 19.10 host, Intel i7, 16 GB RAM, an NVIDIA(R) RTX 2060, CUDA(R) 10.2 and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker). 

# More Infos:

- Microsoft's LightGBM: https://lightgbm.readthedocs.io/en/latest/
- mlr3: https://github.com/mlr-org/mlr3

