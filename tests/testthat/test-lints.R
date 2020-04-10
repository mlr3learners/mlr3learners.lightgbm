context("lints")

if (dir.exists("../../00_pkg_src")) {
  prefix <- "../../00_pkg_src/mlr3learners.lightgbm/"
} else if (dir.exists("../../R")) {
  prefix <- "../../"
} else if (dir.exists("./R")) {
  prefix <- "./"
}

test_that(
  desc = "test lints",
  code = {
    skip_on_cran()
    skip_if(dir.exists("../../00_pkg_src"))
    lintr::expect_lint_free(path = prefix)
  })
