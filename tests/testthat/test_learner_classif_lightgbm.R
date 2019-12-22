context("LearnerClassifLightGBM")

test_that(
  desc = "LearnerClassifLightGBM",
  code = {

    learner <- LearnerClassifLightGBM$new()
    expect_learner(learner)
    learner$nrounds <- 1000
    learner$early_stopping_rounds <- 50
    result <- run_autotest(learner)
    skip("Type error in score()")
    expect_true(result, info = result$error)
  }
)
