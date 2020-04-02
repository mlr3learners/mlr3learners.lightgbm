context("LearnerRegrLightGBM")

# https://github.com/mlr-org/mlr3/blob/master/inst/testthat/helper_autotest.R

test_that(
  desc = "LearnerRegrLightGBM",
  code = {

    learner = LearnerRegrLightGBM$new()
    expect_learner(learner)
    learner$early_stopping_rounds = 5
    learner$nrounds = 10
    result = run_autotest(learner)
    expect_true(result, info = result$error)
  }
)
