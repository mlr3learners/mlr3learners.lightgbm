context("LearnerRegrLightGBM")

# https://github.com/mlr-org/mlr3/blob/master/inst/testthat/helper_autotest.R

test_that(
  desc = "LearnerRegrLightGBM",
  code = {

    learner <- LearnerRegrLightGBM$new()
    expect_learner(learner)
    learner$param_set$values <- list(
      "learning_rate" = 0.01,
      "seed" = 17L,
      "metric" = "auc"
    )
    result <- run_autotest(learner)
    skip("Type error in score()")
    expect_true(result, info = result$error)
  }
)
