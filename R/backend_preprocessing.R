#' @title backend_preprocessing
#'
#' @description The function transforms the values of a data.table object
#'   so that in can be passed to an mlr3 task to be used with the
#'   \code{mlr3learners.lightgbm} R package. This is necessary because
#'   \code{lightgbm} can only handle numeric and integer features. Furthermore,
#'   for classification tasks the target variable needs to be of type integer
#'   with values starting at '0' for the first class.
#'   The function is a wrapper around \code{lightgbm::prepare}.
#'
#' @param datatable A data.table object, holding the target variable and
#'   features.
#' @param target_col A character. The name of the target variable.
#' @param task_type A character. The type of learning task to prepare
#'   `datatable` for. Can be one of `regression`, `class:binary` or
#'   `class:multiclass`.
#' @param positive A character. If `task_type` = `class:binary`, this argument
#'   is required to specify the positive class of the binary classification
#'   task, which will be replaced by the value `1` in the resulting object.
#'
#' @return The function returns a data.table with the transformed target
#'   variable and feature variables. This object can then be used to create
#'   an mlr3 task to be used with the \code{mlr3learners.lightgbm} R package.
#'
#' @seealso \code{mlr3}, \code{lightgbm::prepare}, \code{mlr3learners.lightgbm},
#'   \code{plyr::revalue}
#'
#' @export
#'
backend_preprocessing = function(
  datatable,
  target_col,
  task_type,
  positive = NULL) {

  stopifnot(
    data.table::is.data.table(datatable)
    , is.character(target_col)
    , target_col %in% colnames(datatable)
    , is.character(task_type)
    , task_type %in% c("regression", "class:binary", "class:multiclass")
  )

  # extract label
  label <- datatable[, get(target_col)]

  if (task_type == "class:binary") {
    stopifnot(
      is.character(positive)
      , positive %in% datatable[, unique(get(target_col))]
    )

    # transform label (revalue it and set "positive" to 1)
    negative = setdiff(
      datatable[, unique(get(target_col))],
      positive
    )

    message(paste0("positive class: ", positive))
    message(paste0("negative class: ", negative))

    # replace values
    repl = c(0, 1)
    # create named vector
    names(repl) = c(negative, positive)
    # revalue
    label = as.integer(plyr::revalue(
      x = as.character(label),
      replace = repl
    ))
  } else if (task_type == "class:multiclass") {
    label = as.integer(factor(label)) - 1L
  } else {
    stopifnot(is.numeric(label))
  }

  # get feature colnames
  vec = setdiff(colnames(datatable), target_col)

  # create backend
  backend = cbind(
    label,
    lightgbm::lgb.convert_with_rules(datatable[, vec, with = F])[[1]]
  )
  colnames(backend)[1] = target_col

  # return backend
  return(backend)
}
