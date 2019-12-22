#' @title Classification LightGBM Learner
#'
#' @aliases mlr_learners_classif.lightgbm
#' @format [R6::R6Class] inheriting from [mlr3::LearnerClassif].
#'
#' @import data.table
#' @import paradox
#' @importFrom mlr3 mlr_learners LearnerClassif
#'
#' @export
LearnerClassifLightGBM <- R6::R6Class(
  "LearnerClassifLightGBM",
  inherit = LearnerClassif,

  private = list(

    # data: train, valid, test
    train_input = NULL,
    valid_input = NULL,
    test_input = NULL,
    input_rules = NULL,

    # the list, passed to the train function
    valid_list = NULL,

    # the label names used as column names for the prediction output
    label_names = NULL,

    # the transfrom_target instance
    trans_tar = NULL,

    # save importance values
    imp = NULL
  ),

  public = list(

    #' @field nrounds Number of training rounds.
    nrounds = NULL,

    #' @field early_stopping_rounds A integer. Activates early stopping.
    #'   Requires at least one validation data and one metric. If there's
    #'   more than one, will check all of them except the training data.
    #'   Returns the model with (best_iter + early_stopping_rounds).
    #'   If early stopping occurs, the model will have 'best_iter' field.
    early_stopping_rounds = NULL,

    #' @field categorical_feature A list of str or int. Type int represents
    #'   index, type str represents feature names.
    categorical_feature = NULL,

    #' @field train_data A data.table object holding the training data.
    train_data = NULL,
    #' @field train_label A vector holding the training labels.
    train_label = NULL,

    #' @field valid_data A data.table object holding the validation data.
    valid_data = NULL,
    #' @field valid_label A vector holding the validation labels.
    valid_label = NULL,

    # define methods
    #' @description The initialize function.
    #'
    initialize = function() {

      self$nrounds <- 10L

      super$initialize(
        # see the mlr3book for a description:
        # https://mlr3book.mlr-org.com/extending-mlr3.html
        id = "classif.lightgbm",
        packages = "lightgbm",
        feature_types = c("numeric", "factor", "ordered"),
        predict_types = "prob",
        param_set = lgbparams(),
        properties = c("twoclass",
                       "multiclass",
                       "missings",
                       "importance")
      )

      private$trans_tar <- TransformTarget$new(
        param_set = self$param_set
      )
    },

    #' @description The train_internal function
    #'
    #' @param task An mlr3 task
    #'
    train_internal = function(task) {

      data <- task$data()

      n <- nlevels(factor(data[, get(task$target_names)]))

      if (is.null(self$param_set$values[["objective"]])) {
        # if not provided, set default objective depending on the
        # number of levels
        message("No objective provided...")
        if (n > 2) {
          self$param_set$values <- c(
            self$param_set$values,
            list("objective" = "multiclass")
          )
          message("Setting objective to 'multiclass'")
        } else if (n == 2) {
          self$param_set$values <- c(
            self$param_set$values,
            list("objective" = "binary")
          )
          message("Setting objective to 'binary'")
        } else {
          stop(paste0("Please provide a target with a least ",
                      "2 levels for classification tasks"))
        }

      } else {
        stopifnot(
          self$param_set$values[["objective"]] %in%
            c("binary", "multiclass", "multiclassova", "lambdarank")
        )
      }

      # create label
      self$train_label <- private$trans_tar$transform_target(
        vector = data[, get(task$target_names)],
        mapping = "dtrain"
      )
      private$label_names <- sort(unique(self$train_label))

      if (!is.null(private$valid_input)) {
        stopifnot(
          identical(
            private$trans_tar$value_mapping_dtrain,
            private$trans_tar$value_mapping_dvalid
          )
        )
      }

      # extract classification classes
      if (length(private$label_names) > 2) {
        stopifnot(
          self$param_set$values[["objective"]] %in%
            c("multiclass", "multiclassova", "lambdarank")
        )
        self$param_set$values[["num_class"]] <- length(private$label_names)
      }

      # create lgb.Datasets
      private$train_input <- lightgbm::lgb.prepare_rules(
        data[, task$feature_names, with = F],
        rules = private$input_rules
      )
      if (is.null(private$input_rules)) {
        private$input_rules <- private$train_input$rules
      }
      self$train_data <- lightgbm::lgb.Dataset(
        data = as.matrix(private$train_input$data),
        label = self$train_label,
        free_raw_data = FALSE
      )

      # switch of lightgbm's parallelization and use the one of mlr3
      if (is.null(self$param_set$values[["num_threads"]])) {
        self$param_set$values <- c(
          self$param_set$values,
          list("num_threads" = 1L)
        )
      } else if (self$param_set$values[["num_threads"]] != 1L) {
        self$param_set$values[["num_threads"]] <- 1L
      }

      mlr3misc::invoke(
        .f = lightgbm::lgb.train,
        params = self$param_set$values,
        data = self$train_data,
        nrounds = self$nrounds,
        valids = private$valid_list,
        categorical_feature = self$categorical_feature,
        eval_freq = 50L,
        early_stopping_rounds = self$early_stopping_rounds
      ) # use the mlr3misc::invoke function (it's similar to do.call())
    },

    #' @description The predict_internal function
    #'
    #' @param task An mlr3 task
    #'
    predict_internal = function(task) {
      newdata <- task$data(cols = task$feature_names) # get newdata

      # create lgb.Datasets
      private$test_input <- lightgbm::lgb.prepare_rules(
        newdata,
        rules = private$input_rules
      )

      test_data <- as.matrix(private$test_input$data)

      p <- mlr3misc::invoke(
        .f = self$model$predict,
        data = test_data,
        reshape = TRUE
      )

      if (self$param_set$values[["objective"]] %in%
          c("multiclass", "multiclassova", "lambdarank")) {
        colnames(p) <- as.character(unique(private$label_names))

        # process target variable
        c_names <- colnames(p)
        c_names <- plyr::revalue(
          x = c_names,
          replace = private$trans_tar$value_mapping_dtrain
        )
        colnames(p) <- c_names

      } else if (self$param_set$values[["objective"]] == "binary") {

        # reshape binary prob to matrix
        p <- cbind(
          "0" = 1 - p,
          "1" = p
        )

        c_names <- colnames(p)
        c_names <- plyr::revalue(
          x = c_names,
          replace = private$trans_tar$value_mapping_dtrain
        )
        colnames(p) <- c_names
      }

      PredictionClassif$new(
        task = task,
        prob = p
      )

    },

    # Add method for importance, if learner supports that.
    # It must return a sorted (decreasing) numerical, named vector.
    #' @description The importance function
    #'
    #' @details A named vector with the learner's variable importances.
    #'
    importance = function() {
      if (is.null(self$model)) {
        stop("No model stored")
      }

      if (is.null(private$imp)) {
        private$imp <- lightgbm::lgb.importance(learner$model)
      }
      ret <- sapply(private$imp$Feature, function(x) {
        return(private$imp[which(private$imp$Feature == x), ]$Gain)
      }, USE.NAMES = TRUE, simplify = TRUE)

      return(unlist(ret))
    },
    #' @description The importance2 function
    #'
    #' @details Returns a list with the learner's variable importance values
    #'   and an importance plot.
    #'
    importance2 = function() {
      if (is.null(self$model)) {
        stop("No model stored")
      }

      plot <- lightgbm::lgb.plot.importance(private$imp)
      return(list(
        importance = private$imp,
        plot = plot
      ))
    },

    #' @description The valids function
    #'
    #' @details The function can be used to provide a subsample to the data
    #'   to the lightgbm's train function's `valids` argument. This is e.g.
    #'   needed, when the argument `early_stopping_rounds` is used.
    #'
    #' @param task The mlr3 task.
    #' @param row_ids An integer vector with the row IDs for the validation
    #'   data.
    #'
    valids = function(task, row_ids) {
      task <- mlr3::assert_task(as_task(task))
      mlr3::assert_learnable(task, self)

      row_ids <- mlr3::assert_row_ids(row_ids)

      mlr3::assert_task(task)

      # subset to test set w/o cloning
      row_ids <- assert_row_ids(row_ids)
      prev_use <- task$row_roles$use
      on.exit({
        task$row_roles$use <- prev_use
      }, add = TRUE)
      task$row_roles$use <- row_ids

      vdata <- task$data()

      # create label
      self$valid_label <- private$trans_tar$transform_target(
        vector = vdata[, get(task$target_names)],
        mapping = "dvalid"
      )

      # create lgb.Datasets
      private$valid_input <- lightgbm::lgb.prepare_rules(
        vdata[, task$feature_names, with = F]
      )
      private$input_rules <- private$valid_input$rules

      self$valid_data <- lightgbm::lgb.Dataset(
        data = as.matrix(private$valid_input$data),
        label = self$valid_label,
        free_raw_data = FALSE
      )

      private$valid_list <- list(validation = self$valid_data)
    }
  )
)
