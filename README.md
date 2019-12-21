# mlr3learners.lightgbm (!!!under development!!!)

<!-- badges: start -->
[![pipeline status](https://gitlab.com/kapsner/mlr3learners-lightgbm/badges/master/pipeline.svg)](https://gitlab.com/kapsner/mlr3learners-lightgbm/commits/master)
[![coverage report](https://gitlab.com/kapsner/mlr3learners-lightgbm/badges/master/coverage.svg)](https://gitlab.com/kapsner/mlr3learners-lightgbm/commits/master)
<!-- badges: end -->
 
[mlr3learners.lightgbm](https://github.com/kapsner/mlr3learners.lightgbm) brings the [LightGBM gradient booster](https://lightgbm.readthedocs.io) to the [mlr3](https://github.com/mlr-org/mlr3) framework by using the [lightgbm.py](https://github.com/kapsner/lightgbm.py) R implementation. 

# Installation

Install the [mlr3learners.lightgbm](https://github.com/kapsner/mlr3learners.lightgbm) R package:

```r
install.packages("devtools")
devtools::install_github("kapsner/mlr3learners.lightgbm")
```

Before you can use the `mlr3learners.lightgbm` package, you need to install the lightgbm R package according to [its documentation](https://github.com/microsoft/LightGBM/blob/master/R-package/README.md) (this is necessary since lightgbm is neither on CRAN nor installable via `devtools::install_github`).  

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

# More Infos:

- Microsoft's LightGBM: https://lightgbm.readthedocs.io/en/latest/
- mlr3: https://github.com/mlr-org/mlr3

