context("Test importance_plot")

test_that(
  desc = "Testing plot",
  code = {

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
        "metric" = "auc"
      )
    )
    learner$train(task, row_ids = split$train_index)

    importance  = learner$importance()

    expect_length(importance, 8)

    expect_class(importance_plot(importance), c("gg", "ggplot"))
  }
)
