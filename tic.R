# R CMD check
if (!ci_has_env("PARAMTEST")) {
  do_package_checks()

  get_stage("install") %>%
    add_step(step_install_deps()) %>%
    add_code_step(git2r::clone("https://github.com/microsoft/LightGBM", "lightgbm")) %>%
    add_code_step(system("Rscript -e 'setwd(paste0(getwd(), \"/lightgbm\")); source(\"build_r.R\")'"))

  #do_drat("mlr3learners/mlr3learners.drat")
} else {
  # PARAMTEST
  get_stage("install") %>%
    add_step(step_install_deps()) %>%
    add_code_step(git2r::clone("https://github.com/microsoft/LightGBM", "lightgbm")) %>%
    add_code_step(system("Rscript -e 'setwd(paste0(getwd(), \"/lightgbm\")); source(\"build_r.R\")'"))

  get_stage("script") %>%
    add_code_step(testthat::test_dir(system.file("paramtest", package = "mlr3learners.lightgbm"),
      stop_on_failure = TRUE))
}
