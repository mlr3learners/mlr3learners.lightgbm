context("Test custom metrics")

test_that(
  desc = "eval_prauc",
  code = {

    # currently (06.06.2020) fails due to
    # https://github.com/microsoft/LightGBM/issues/3112


    library(mlbench)
    data("PimaIndiansDiabetes2")
    dataset = data.table::as.data.table(PimaIndiansDiabetes2)
    target_col = "diabetes"

    dataset = lightgbm::lgb.prepare(dataset)
    dataset[, (target_col) := factor(get(target_col) - 1L)]

    task = mlr3::TaskClassif$new(
      id = "pima",
      backend = dataset,
      target = target_col,
      positive = "1"
    )

    set.seed(17)
    split = list(
      train_index = sample(seq_len(task$nrow), size = 0.7 * task$nrow)
    )
    split$test_index = setdiff(seq_len(task$nrow), split$train_index)

    learner = mlr3::lrn("classif.lightgbm", objective = "binary")
    learner$param_set$values = mlr3misc::insert_named(
      learner$param_set$values,
      list(
        "early_stopping_round" = 3,
        "learning_rate" = 0.1,
        "seed" = 17L,
        "num_iterations" = 10,
        "custom_eval" = lgb_prauc
      )
    )
    learner$train(task, row_ids = split$train_index)

    expect_equal(learner$model$current_iter(), 10L)
  }
)


test_that(
  desc = "eval_rmsle",
  code = {

    # currently (06.06.2020) fails due to
    # https://github.com/microsoft/LightGBM/issues/3112


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
        "num_iterations" = 10,
        "custom_eval" = lgb_rmsle
      )
    )

    learner$train(task, row_ids = split$train_index)

    expect_equal(learner$model$current_iter(), 10L)

  }
)
