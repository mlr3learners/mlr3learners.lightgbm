context("backend_preprocessing")

test_that(
  desc = "Test the backend_preprocessing function",
  code = {
    library(mlbench)
    data("PimaIndiansDiabetes2")
    dataset = data.table::as.data.table(PimaIndiansDiabetes2)
    target_col = "diabetes"

    backend = backend_preprocessing(
      datatable = dataset,
      target_col = target_col,
      task_type = "class:binary",
      positive = "pos"
    )

    expect_known_hash(backend, "d4474ab2b4")
  }
)
