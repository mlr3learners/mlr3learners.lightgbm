context("LearnerRegrLightGBM")

# https://github.com/mlr-org/mlr3/blob/master/inst/testthat/helper_autotest.R

test_that(
  desc = "LearnerRegrLightGBM",
  code = {
    learner = LearnerRegrLightGBM$new()
    expect_learner(learner)
    learner$param_set$values = mlr3misc::insert_named(
      learner$param_set$values,
      list(
        "early_stopping_round" = 3,
        "learning_rate" = 0.1,
        "num_iterations" = 10
      )
    )
    result = run_autotest(learner, exclude = "sanity")
    expect_true(result, info = result$error)
  }
)
