task <- mlr3::tsk("boston_housing")
set.seed(17)
split <- list(
  train_index = sample(seq_len(task$nrow), size = 0.7 * task$nrow)
)
split$test_index <- setdiff(seq_len(task$nrow), split$train_index)

params <- list(
  "objective" = "regression",
  "learning_rate" = 0.1,
  "metric" = "rmse",
  "seed" = 17
)

data <- task$data(rows = split$train_index)
data_test <- task$data(rows = split$test_index)

dtrain <- lightgbm::lgb.Dataset(
  data = as.matrix(data[, task$feature_names, with = F]),
  label = data[, get(task$target_names)],
  free_raw_data = FALSE
)

model <- lightgbm::lgb.cv(params = params, data = dtrain,
                          early_stopping_rounds = 100,
                          nrounds = 5000)

final_model <- lightgbm::lgb.train(params = params,
                                   data = dtrain,
                                   nrounds = model$best_iter)

predictions <- final_model$predict(as.matrix(data_test[, task$feature_names, with = F]))

MLmetrics::RMSLE(y_pred = predictions, y_true = data_test[, get(task$target_names)])

imp <- lightgbm::lgb.importance(final_model)

imp
