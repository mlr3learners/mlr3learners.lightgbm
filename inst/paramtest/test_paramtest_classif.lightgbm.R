library(mlr3learners.lightgbm)

test_that("classif.lightgbm", {
  learner = mlr3::lrn("classif.lightgbm")
  fun = lightgbm::lgb.train
  exclude = c("params",
              "data",
              "nrounds",
              "valids",
              "obj",
              "eval",
              "record",
              "eval_freq",
              "colnames",
              "early_stopping_rounds",
              "callbacks",
              "reset_data")

  ParamTest = run_paramtest(learner, fun, exclude)
  expect_true(ParamTest, info = paste0(
    "
Missing parameters:
",
    paste0("- '", ParamTest$missing, "'", collapse = "
")))
})
