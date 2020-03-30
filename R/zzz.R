#' @import data.table
#' @import paradox
#' @import mlr3misc
#' @importFrom R6 R6Class
#' @importFrom mlr3 mlr_learners LearnerClassif LearnerRegr
"_PACKAGE"
.onLoad <- function(libname, pkgname) {
  # nocov start
  # get mlr_learners dictionary from the mlr3 namespace
  x <- utils::getFromNamespace("mlr_learners", ns = "mlr3")

  # add the learner to the dictionary
  x$add("classif.lightgbm", LearnerClassifLightGBM)
  x$add("regr.lightgbm", LearnerRegrLightGBM)
} # nocov end



#' @import data.table
#' @import paradox
#' @import mlr3misc
#' @importFrom R6 R6Class
#' @importFrom mlr3 mlr_learners LearnerClassif LearnerRegr
"_PACKAGE"

# nocov start
register_mlr3 = function(libname, pkgname) {
  # get mlr_learners dictionary from the mlr3 namespace
  x = utils::getFromNamespace("mlr_learners", ns = "mlr3")

  # add the learner to the dictionary
  x$add("classif.lightgbm", LearnerClassifLightGBM)
  x$add("regr.lightgbm", LearnerRegrLightGBM)
}

.onLoad = function(libname, pkgname) {
  register_mlr3()
  setHook(packageEvent("mlr3", "onLoad"), function(...) register_mlr3(),
          action = "append")
}

.onUnload = function(libpath) {
  event = packageEvent("mlr3", "onLoad")
  hooks = getHook(event)
  pkgname = vapply(hooks, function(x) environment(x)$pkgname, NA_character_)
  setHook(event, hooks[pkgname != "mlr3learners.lightgbm"],
          action = "replace")
}
# nocov end
