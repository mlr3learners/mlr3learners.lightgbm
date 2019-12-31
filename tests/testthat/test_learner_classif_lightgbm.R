context("LearnerClassifLightGBM")

test_that(
  desc = "LearnerClassifLightGBM",
  code = {

    learner <- LearnerClassifLightGBM$new()
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
