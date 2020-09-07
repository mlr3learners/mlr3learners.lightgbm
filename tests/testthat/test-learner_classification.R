context("Test Classification")

test_that(
  desc = "Learner Classification",
  code = {
    library(mlbench)
    data("PimaIndiansDiabetes2")
    dataset = data.table::as.data.table(PimaIndiansDiabetes2)
    target_col = "diabetes"

    vec = setdiff(colnames(dataset), target_col)

    dataset = cbind(
      dataset[, c(target_col), with = F],
      lightgbm::lgb.convert_with_rules(dataset[, vec, with = F])[[1]]
    )

    task = mlr3::TaskClassif$new(
      id = "pima",
      backend = dataset,
      target = target_col,
      positive = "pos"
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
        "metric" = "auc",
        "num_iterations" = 10
      )
    )
    learner$train(task, row_ids = split$train_index)

    expect_equal(learner$model$current_iter(), 10L)

    predictions = learner$predict(task, row_ids = split$test_index)

    expect_known_hash(predictions$response, "b8b6d3c1bd") # 264f358a7c
    importance = learner$importance()

    expect_equal(
      round(importance[["glucose"]], 4),
      round(0.45686638496446557722, 4)
    )
  }
)
