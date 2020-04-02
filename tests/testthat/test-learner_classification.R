context("Test Classification")

test_that(
  desc = "Learner Classification",
  code = {

    task = mlr3::tsk("pima")

    set.seed(17)
    split = list(
      train_index = sample(seq_len(task$nrow), size = 0.7 * task$nrow)
    )
    split$test_index = setdiff(seq_len(task$nrow), split$train_index)

    learner = mlr3::lrn("classif.lightgbm")
    learner$early_stopping_rounds = 3
    learner$nrounds = 10

    learner$param_set$values = list(
      "learning_rate" = 0.1,
      "seed" = 17L,
      "metric" = "auc"
    )

    learner$train(task, row_ids = split$train_index)

    expect_equal(learner$model$current_iter(), 10L)
    expect_known_hash(learner$lgb_learner$train_label, "9d63ff0583")

    predictions = learner$predict(task, row_ids = split$test_index)

    expect_known_hash(predictions$response, "b8b6d3c1bd")
    importance = learner$importance()

    expect_equal(importance[["glucose"]], 0.45657835120582451749)
  }
)
