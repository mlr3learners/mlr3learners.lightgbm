# R CMD check
if (!ci_has_env("PARAMTEST")) {

  # see https://github.com/microsoft/LightGBM/tree/master/R-package#install
  get_stage("install") %>%
    add_code_step(system2("git",
      args = c("clone", "--recursive", "https://github.com/microsoft/LightGBM"))) %>%
    add_code_step(system("cd LightGBM && Rscript build_r.R"))

  do_package_checks()

  do_drat("mlr3learners/mlr3learners.drat")
} else {
  # PARAMTEST
  get_stage("install") %>%
    add_step(step_install_deps()) %>%
    add_code_step(system2("git",
      args = c("clone", "--recursive", "https://github.com/microsoft/LightGBM"))) %>%
    add_code_step(system("cd LightGBM && Rscript build_r.R"))

  get_stage("script") %>%
    add_code_step(remotes::install_dev("mlr3")) %>%
    add_code_step(testthat::test_dir(system.file("paramtest", package = "mlr3learners.lightgbm"),
      stop_on_failure = TRUE))
}
