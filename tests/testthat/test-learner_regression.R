context("Test Regression")

test_that(
  desc = "Learner Regression",
  code = {
    library(mlbench)
    data("BostonHousing2")
    dataset = data.table::as.data.table(BostonHousing2)
    target_col = "medv"

    dataset = lightgbm::lgb.prepare(dataset)

    task = mlr3::TaskRegr$new(
      id = "bostonhousing",
      backend = dataset,
      target = target_col
    )

    set.seed(17)
    split = list(
      train_index = sample(seq_len(task$nrow), size = 0.7 * task$nrow)
    )
    split$test_index = setdiff(seq_len(task$nrow), split$train_index)

    learner = mlr3::lrn("regr.lightgbm", objective = "regression")

    learner$param_set$values = mlr3misc::insert_named(
      learner$param_set$values,
      list(
        "early_stopping_round" = 3,
        "learning_rate" = 0.1,
        "seed" = 17L,
        "num_iterations" = 10
      )
    )

    learner$train(task, row_ids = split$train_index)

    expect_equal(learner$model$current_iter(), 10L)

    predictions = learner$predict(task, row_ids = split$test_index)

    expect_known_hash(predictions$response, "27b55df5ab")
    importance = learner$importance()

    expect_equal(importance[["cmedv"]], 0.99991534830393857813)
  }
)
