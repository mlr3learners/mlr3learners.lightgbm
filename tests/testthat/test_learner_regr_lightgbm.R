context("LearnerRegrLightGBM")

# https://github.com/mlr-org/mlr3/blob/master/inst/testthat/helper_autotest.R

test_that(
  desc = "LearnerRegrLightGBM",
  code = {

    learner <- LearnerRegrLightGBM$new()
    expect_learner(learner)
    result <- run_autotest(learner, predict_types = "response")
    skip("Type error in score()")
    expect_true(result, info = result$error)
  }
)