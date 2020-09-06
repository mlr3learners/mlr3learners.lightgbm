library(mlr3learners.lightgbm)

test_that("regr.lightgbm", {
  learner = mlr3::lrn("regr.lightgbm")
  fun = lightgbm::lgb.train

  # Please refer to: https://lightgbm.readthedocs.io/en/latest/Parameters.html
  exclude = c(
    "nrounds", # nrounds is included in the ParamSet as "num_iterations"
    "valids", # a validation dataset is not implemented yet
    "obj", # obj is included in the ParamSet as "objective"
    "record", # record is not implemented
    "eval_freq", # eval_freq is implemented as "metric_freq"
    "colnames", # colnames is not implemented
    "early_stopping_rounds", # implemented as "early_stopping_round"
    "callbacks", # not impolemented
    "reset_data", # not implemented
    # params is defined in lgb.train:
    # https://github.com/mlr3learners/mlr3learners.lightgbm/blob/master/R/
    # LearnerRegrLightGBM.R#L674
    "params",
    # data is defined in lgb.trian:
    # https://github.com/mlr3learners/mlr3learners.lightgbm/blob/master/R/
    # LearnerRegrLightGBM.R#L673
    "data",
    # eval is defined in lgb.train:
    # https://github.com/mlr3learners/mlr3learners.lightgbm/blob/master/R/
    # LearnerRegrLightGBM.R#L675
    "eval"
  )

  ParamTest = run_paramtest(learner, fun, exclude)
  expect_true(ParamTest, info = paste0(
    "Missing parameters:",
    paste0("- '", ParamTest$missing, "'", collapse = "
")))
})
