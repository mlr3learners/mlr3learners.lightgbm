#' @title LightGBM Learner
#'
LightGBM <- R6::R6Class(
  "LightGBM",

  private = list(

    valid_state = NULL,

    # data: train, valid, test
    train_input = NULL,
    valid_input = NULL,
    test_input = NULL,
    input_rules = NULL,

    # the list, passed to the train function
    valid_list = NULL,

    # save importance values
    imp = NULL,

    # convert object types
    # this is necessary, since mlr3 tuning does pass wrong types
    convert_types = function() {

      self$nrounds <- as.integer(self$nrounds)
      if (!is.null(self$early_stopping_rounds)) {
        self$early_stopping_rounds <- as.integer(self$early_stopping_rounds)
      }
      self$cv_folds <- as.integer(self$cv_folds)

      # check for user-changed num_iterations here
      if (!is.null(self$param_set$values[["num_iterations"]])) {
        # if yes, pass value to nrounds
        self$nrounds <- self$param_set$values[["num_iterations"]]
      }

      # set correct types for parameters
      for (param in names(self$param_set$values)) {
        value <- self$param_set$values[[param]]
        if (self$param_set$class[[param]] == "ParamInt") {
          self$param_set$values[[param]] <- as.integer(round(value))
        } else if (self$param_set$class[[param]] == "ParamDbl") {
          self$param_set$values[[param]] <- as.numeric(value)
        }
      }
    },

    #' @description The backend_preprocessing function
    #'
    #' @param task An mlr3 task
    #'
    backend_preprocessing = function(task) {

      stopifnot(
        !is.null(self$param_set$values[["objective"]])
      )

      # extract data
      data <- task$data()

      # give param_set to transform target function
      self$trans_tar$param_set <- self$param_set

      # create training label
      self$train_label <- self$trans_tar$transform_target(
        vector = data[, get(task$target_names)],
        positive = task$positive,
        negative = task$negative,
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
        n <- data[, nlevels(factor(get(task$target_names)))]
        if (n > 2) {
          stopifnot(
            self$param_set$values[["objective"]] %in%
              c("multiclass", "multiclassova", "lambdarank")
          )
        }
        # set num_class only in multiclass-objective
        if (self$param_set$values[["objective"]] == "multiclass") {
          self$param_set$values[["num_class"]] <- n
        } else {
          self$param_set$values <-
            self$param_set$values[names(self$param_set$values) != "num_class"]
        }
      }

      if (isFALSE(private$valid_state)) {
        private$input_rules <- NULL
      }

      # create lgb.Datasets
      private$train_input <- lightgbm::lgb.prepare(
        data[, task$feature_names, with = F]
      )
      self$train_data <- lightgbm::lgb.Dataset(
        data = as.matrix(private$train_input),
        label = self$train_label,
        reference = private$input_rules,
        free_raw_data = FALSE
      )

      if ("weights" %in% task$properties) {
        lightgbm::setinfo(self$train_data, "weight", task$weights$weight)
      }

      # if user has not specified categorical_feature, look in data for
      # categorical features
      if (is.null(self$categorical_feature) && self$autodetect_categorical) {
        if (any(task$feature_types$type %in%
                c("factor", "ordered", "character"))) {
          cat_feat <- task$feature_types[
            get("type") %in% c("factor", "ordered", "character"), get("id")
            ]
          self$categorical_feature <- cat_feat
        }
      }

      if (!is.null(self$categorical_feature)) {
        self$train_data$set_colnames(task$feature_names)
      }

      if (is.null(private$input_rules)) {
        private$input_rules <- self$train_data
      }

      # add to training data to validation set:
      if (!is.null(self$valid_data)) {
        if (!is.null(self$categorical_feature)) {
          self$valid_data$set_colnames(task$feature_names)
        }
        private$valid_list <- c(
          list(dvalid = self$valid_data),
          list(dtrain = self$train_data)
        )
      }
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

    #' @field categorical_feature A vector of str or int. Type int represents
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
    nrounds_by_cv = NULL,

    #' @field cv_folds The number of cross validation folds, when setting
    #'   `nrounds_by_cv` = TRUE (default: 5).
    cv_folds = NULL,

    #' @field cv_model The cross validation model.
    cv_model = NULL,

    #' @field autodetect_categorical Automatically detect categorical features.
    autodetect_categorical = NULL,

    #' @field model The trained lightgbm model.
    model = NULL,

    # define methods
    #' @description The initialize function.
    #'
    initialize = function() {

      self$nrounds <- 10L

      self$cv_folds <- 5L

      self$nrounds_by_cv <- TRUE

      self$param_set <- lgbparams()

      self$autodetect_categorical <- TRUE

      self$trans_tar <- TransformTarget$new(
        param_set = self$param_set
      )

      private$valid_state <- FALSE

    },

    #' @description The train_cv function
    #'
    #' @param task An mlr3 task
    #'
    train_cv = function(task) {
      message(
        sprintf(
          paste0("Optimizing nrounds with %s fold CV."),
          self$cv_folds
        )
      )

      private$backend_preprocessing(task)

      private$convert_types()

      self$cv_model <- lightgbm::lgb.cv(
        params = self$param_set$values,
        data = self$train_data,
        nrounds = self$nrounds,
        nfold = self$cv_folds,
        categorical_feature = self$categorical_feature,
        eval_freq = 20L,
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
    },

    #' @description The train function
    #'
    #' @param task An mlr3 task
    #'
    train = function(task) {
      if (self$nrounds_by_cv) {
        self$train_cv(task)
      } else if (isFALSE(self$nrounds_by_cv)) {
        private$backend_preprocessing(task)
        private$convert_types()
      }

      self$model <- lightgbm::lgb.train(
        params = self$param_set$values,
        data = self$train_data,
        nrounds = self$nrounds,
        valids = private$valid_list,
        categorical_feature = self$categorical_feature,
        eval_freq = 20L,
        early_stopping_rounds = self$early_stopping_rounds
      )
      message(
        sprintf("Final model: current iter: %s", self$model$current_iter())
      )
      return(self$model)
    },

    #' @description The predict function
    #'
    #' @param task An mlr3 task
    #'
    predict = function(task) {
      newdata <- task$data(cols = task$feature_names) # get newdata

      data.table::setcolorder(
        newdata,
        self$train_data$get_colnames()
      )

      # create lgb.Datasets
      private$test_input <- lightgbm::lgb.prepare(
        newdata
      )

      test_data <- as.matrix(private$test_input)

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
        private$imp <- lightgbm::lgb.importance(self$model)
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
        private$imp <- lightgbm::lgb.importance(self$model)
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

      private$valid_state <- TRUE

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

      # give param_set to transform target function
      self$trans_tar$param_set <- self$param_set

      # create label
      self$valid_label <- self$trans_tar$transform_target(
        vector = vdata[, get(task$target_names)],
        positive = task$positive,
        negative = task$negative,
        mapping = "dvalid"
      )

      # create lgb.Datasets
      private$valid_input <- lightgbm::lgb.prepare(
        vdata[, task$feature_names, with = F]
      )

      self$valid_data <- lightgbm::lgb.Dataset(
        data = as.matrix(private$valid_input),
        label = self$valid_label,
        free_raw_data = FALSE
      )


      if ("weights" %in% task$properties) {
        lightgbm::setinfo(self$valid_data, "weight", task$weights$weight)
      }

      private$input_rules <- self$valid_data
    }
  )
)
