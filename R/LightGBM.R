#' @title LightGBM Learner
#'
LightGBM <- R6::R6Class(
  "LightGBM",

  private = list(

    # data: train, valid, test
    train_input = NULL,
    valid_input = NULL,
    test_input = NULL,
    input_rules = NULL,

    # the list, passed to the train function
    valid_list = NULL,

    # save importance values
    imp = NULL,

    #' @description The data_preprocessing function
    #'
    #' @param task An mlr3 task
    #'
    data_preprocessing = function(task) {

      stopifnot(
        !is.null(self$param_set$values[["objective"]])
      )

      # extract data
      data <- task$data()

      # create training label
      self$train_label <- self$trans_tar$transform_target(
        vector = data[, get(task$target_names)],
        mapping = "dtrain"
      )

      # some further special treatments, when we have a classification task
      if (self$param_set$values[["objective"]] %in%
          c("binary", "multiclass", "multiclassova", "lambdarank")) {
        # store the class label names
        self$label_names <- sort(unique(self$train_label))

        # if a validation set is provided, check if value mappings are
        # identical
        if (!is.null(private$valid_input)) {
          stopifnot(
            identical(
              self$trans_tar$value_mapping_dtrain,
              self$trans_tar$value_mapping_dvalid
            )
          )
        }

        # extract classification classes and set num_class
        if (length(self$label_names) > 2) {
          stopifnot(
            self$param_set$values[["objective"]] %in%
              c("multiclass", "multiclassova", "lambdarank")
          )
          self$param_set$values[["num_class"]] <- length(self$label_names)
        }
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
    }
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

    #' @field label_names The unique label names in classification tasks.
    label_names = NULL,

    #' @field trans_tar The transfrom_target instance.
    trans_tar = NULL,

    #' @field param_set The lightgbm parameters.
    param_set = NULL,

    #' @field nrounds_by_cv A logical. Calculate the best nrounds by using
    #'   the `lgb.cv` before the training step
    nrounds_by_cv = TRUE,

    #' @field cv_folds The number of cross validation folds, when setting
    #'   `nrounds_by_cv` = TRUE (default: 5).
    cv_folds = NULL,

    #' @field cv_model The cross validation model.
    cv_model = NULL,

    #' @field model The trained lightgbm model.
    model = NULL,

    # define methods
    #' @description The initialize function.
    #'
    initialize = function() {

      self$nrounds <- 10L

      self$cv_folds <- 5

      self$param_set <- lgbparams()

      self$trans_tar <- TransformTarget$new(
        param_set = self$param_set
      )
    },

    #' @description The train_cv function
    #'
    #' @param task An mlr3 task
    #'
    train_cv = function(task) {
      if (is.null(self$cv_model)) {
        message(
          sprintf(
            paste0("Optimizing nrounds with %s fold CV."),
            self$cv_folds
          )
        )

        private$data_preprocessing(task)

        self$cv_model <- lightgbm::lgb.cv(
          params = self$param_set$values,
          data = self$train_data,
          nrounds = self$nrounds,
          nfold = self$cv_folds,
          categorical_feature = self$categorical_feature,
          eval_freq = 50L,
          early_stopping_rounds = self$early_stopping_rounds,
          stratified = TRUE
        )
        message(
          sprintf(
            paste0("CV results: best iter %s; best score: %s"),
            self$cv_model$best_iter, self$cv_model$best_score
          )
        )
        # set nrounds to best iteration from cv-model
        self$nrounds <- self$cv_model$best_iter
        # if we already have figured out the best nrounds, which are provided
        # to the train function, we don't need early stopping anymore
        self$early_stopping_rounds <- NULL

        self$nrounds_by_cv <- FALSE

      } else {
        stop("A CV model has already been trained!")
      }
    },

    #' @description The train function
    #'
    #' @param task An mlr3 task
    #'
    train = function(task) {
      if (is.null(self$model)) {
        if (is.null(self$cv_model) && self$nrounds_by_cv) {
          self$train_cv(task)
        } else if (is.null(self$cv_model) && isFALSE(self$nrounds_by_cv)) {
          private$data_preprocessing(task)
        }

        self$model <- lightgbm::lgb.train(
          params = self$param_set$values,
          data = self$train_data,
          nrounds = self$nrounds,
          valids = private$valid_list,
          categorical_feature = self$categorical_feature,
          eval_freq = 50L,
          early_stopping_rounds = self$early_stopping_rounds
        )
        message(
          sprintf("Final model: current iter: %s", self$model$current_iter())
        )
        return(self$model)
      } else {
        stop("A model has already been trained!")
      }
    },

    #' @description The predict function
    #'
    #' @param task An mlr3 task
    #'
    predict = function(task) {
      newdata <- task$data(cols = task$feature_names) # get newdata

      # create lgb.Datasets
      private$test_input <- lightgbm::lgb.prepare_rules(
        newdata,
        rules = private$input_rules
      )

      test_data <- as.matrix(private$test_input$data)

      p <- self$model$predict(
        data = test_data,
        reshape = TRUE
      )

      return(p)
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
      return(private$imp)
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

      if (is.null(private$imp)) {
        private$imp <- lightgbm::lgb.importance(learner$model)
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
      self$valid_label <- self$trans_tar$transform_target(
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
