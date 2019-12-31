#' @title Regression LightGBM Learner
#'
#' @aliases mlr_learners_regr.lightgbm
#' @format [R6::R6Class] inheriting from [mlr3::LearnerRegr].
#'
#' @importFrom mlr3 mlr_learners LearnerRegr
#'
#' @export
LearnerRegrLightGBM <- R6::R6Class(
  "LearnerRegrLightGBM",
  inherit = LearnerRegr,

  private = list(

    # save importance values
    imp = NULL,

    # some pre training checks for this learner
    pre_train_checks = function(task) {
      if (is.null(self$param_set$values[["objective"]])) {
        # if not provided, set default objective to "regression"
        # this is needed for the learner's init_data function
        self$param_set$values <- mlr3misc::insert_named(
          self$param_set$values,
          list("objective" = "regression")
        )
        message("No objective provided... Setting objective to 'regression'")
      } else {
        stopifnot(
          !(self$param_set$values[["objective"]] %in%
              c("binary", "multiclass",
                "multiclassova", "lambdarank"))
        )
      }

      self$lgb_learner$nrounds <- self$nrounds
      self$lgb_learner$early_stopping_rounds <- self$early_stopping_rounds
      self$lgb_learner$categorical_feature <- self$categorical_feature
      self$lgb_learner$param_set <- self$param_set
    }
  ),

  public = list(

    #' @field lgb_learner The lightgbm learner instance
    lgb_learner = NULL,

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

    #' @field cv_model The cross validation model.
    cv_model = NULL,

    # define methods
    #' @description The initialize function.
    #'
    initialize = function() {

      # instantiate the learner
      self$lgb_learner <- LightGBM$new()

      # set default parameters
      self$nrounds <- self$lgb_learner$nrounds
      self$early_stopping_rounds <- self$lgb_learner$early_stopping_rounds
      self$categorical_feature <- self$lgb_learner$categorical_feature

      super$initialize(
        # see the mlr3book for a description:
        # https://mlr3book.mlr-org.com/extending-mlr3.html
        id = "regr.lightgbm",
        packages = "lightgbm",
        feature_types = c(
          "numeric", "factor", "ordered",
          "character", "integer"
        ),
        predict_types = "response",
        param_set = self$lgb_learner$param_set,
        properties = c("missings",
                       "importance")
      )
    },

    #' @description The train_internal function
    #'
    #' @param task An mlr3 task
    #'
    train_internal = function(task) {

      private$pre_train_checks(task)

      mlr3misc::invoke(
        .f = self$lgb_learner$train,
        task = task
      ) # use the mlr3misc::invoke function (it's similar to do.call())
    },

    #' @description The train_cv function
    #'
    #' @param task An mlr3 task
    #' @param row_ids An integer vector with the row IDs for the validation
    #'   data.
    #'
    train_cv = function(task, row_ids) {

      if (is.null(self$model)) {

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

        private$pre_train_checks(task)

        self$lgb_learner$train_cv(task)

        self$cv_model <- self$lgb_learner$cv_model

      } else {

        stop("A final model has already been trained!")
      }
    },

    #' @description The predict_internal function
    #'
    #' @param task An mlr3 task
    #'
    predict_internal = function(task) {

      p <- mlr3misc::invoke(
        .f = self$lgb_learner$predict,
        task = task
      )

      PredictionRegr$new(
        task = task,
        response = p
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
        private$imp <- self$lgb_learner$importance()
      }
      ret <- sapply(private$imp$Feature, function(x) {
        return(private$imp[which(private$imp$Feature == x), ]$Gain)
      }, USE.NAMES = TRUE, simplify = TRUE)

      return(unlist(ret))
    }
  )
)
