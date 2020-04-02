library(mlr3learners.lightgbm)

test_that("classif.lightgbm", {
  learner = lrn("classif.lightgbm")
  fun = lightgbm::lgb.train
  exclude = c()

  ParamTest = run_paramtest(learner, fun, exclude)
  expect_true(ParamTest, info = paste0(
    "
Missing parameters:
",
    paste0("- '", ParamTest$missing, "'", collapse = "
")))
})
