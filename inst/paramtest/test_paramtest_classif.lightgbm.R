library(mlr3learners.lightgbm)

test_that("classif.lightgbm", {
  learner = mlr3::lrn("classif.lightgbm")
  fun = lightgbm::lgb.train

  # Lorenz: "parameters" in the lightgbm-way are only the learner parameters,
  # passed as ParamSet to the "params" argument.
  # All below "excluded" parameters are arguments to the function "lgb.train".
  # Some of them are already included in "params" but with slightly different
  # namings (I preferred the "main"-name and not aliases.
  # Others arguments are not required (currently).
  # Please refer to: https://lightgbm.readthedocs.io/en/latest/Parameters.html
  exclude = c(
    "params", # params is the ParamSet "ps"
    "data", # data is extracted from the task and passed directly to the function
    "nrounds", # nrounds is included in the ParamSet as "num_iterations"
    "valids", # a validation dataset is not implemented yet
    "obj", # obj is included in the ParamSet as "objective"
    "eval", # eval is the possibiliy to add custom evaluation functions; this is
    # implemented with the new config parameter "custom_eval"
    "record", # record is not implemented
    "eval_freq", # eval_freq is implemented as "metric_freq"
    "colnames", # colnames is not implemented
    "early_stopping_rounds", # implemented as "early_stopping_round"
    "callbacks", # not impolemented
    "reset_data" # not implemented
  )

  ParamTest = run_paramtest(learner, fun, exclude)
  expect_true(ParamTest, info = paste0(
    "Missing parameters:",
    paste0("- '", ParamTest$missing, "'", collapse = "
")))
})
