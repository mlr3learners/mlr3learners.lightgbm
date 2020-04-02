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
LearnerClassifLightGBM = R6::R6Class(
  "LearnerClassifLightGBM",
  inherit = LearnerClassif,

  public = list(

    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {

      ps = ParamSet$new(
        params = list(

          #######################################
          # Core Parameters
          ParamInt$new("nrounds", default = 10L, tags = "train"),
          ParamInt$new("early_stopping_rounds", default = NULL, tags = "train"),
          ParamUty$new("categorical_feature", default = NULL, tags = "train"),
          ParamFct$new(
            id = "task",
            default = "train",
            levels = c("train", "predict", "convert_model", "refit"),
            tags = "train"),
          ParamFct$new(
            id = "objective",
            default = "",
            levels = c(
              "",
              "regression", "regression_l1",
              "huber", "fair", "poisson",
              "quantile", "mape", "gamma",
              "tweedie",
              "binary", "multiclass",
              "multiclassova",
              "cross_entropy",
              "cross_entropy_lambda",
              "lambdarank"
            ),
            tags = "train"),
          ParamFct$new(
            id = "boosting",
            default = "gbdt",
            levels = c("gbdt", "rf", "dart", "goss"),
            tags = "train"),
          # % constraints: num_iterations >= 0
          # Note: internally, LightGBM constructs num_class * num_iterations
          # trees for multi-class classification problems
          ParamInt$new(
            id = "num_iterations",
            default = 100L,
            lower = 0L,
            tags = "train"),
          # % constraints: learning_rate > 0.0
          ParamDbl$new(
            id = "learning_rate",
            default = 0.1,
            lower = 0.0,
            tags = "train"),
          # % constraints: 1 < num_leaves <= 131072
          ParamInt$new(
            id = "num_leaves",
            default = 31L,
            lower = 1L,
            upper = 131072L,
            tags = "train"),
          ParamFct$new(
            id = "tree_learner",
            default = "serial",
            levels = c("serial", "feature", "data", "voting"),
            tags = "train"),
          ParamInt$new(
            id = "num_threads",
            default = 1L,
            lower = 0L,
            tags = "train"),
          ParamFct$new(
            id = "device_type",
            default = "cpu",
            levels = c("cpu", "gpu"),
            tags = "train"),
          ParamInt$new(
            id = "seed",
            tags = "train"),
          #######################################
          # Learning Control Parameters
          ParamLgl$new(
            id = "force_col_wise",
            default = FALSE,
            tags = "train"),
          ParamLgl$new(
            id = "force_row_wise",
            default = FALSE,
            tags = "train"),
          # % <= 0 means no limit
          ParamInt$new(
            id = "max_depth",
            default = -1L,
            lower = -1L,
            tags = "train"),
          # % constraints: min_data_in_leaf >= 0
          ParamInt$new(
            id = "min_data_in_leaf",
            default = 20L,
            lower = 0L,
            tags = "train"),
          # % constraints: min_sum_hessian_in_leaf >= 0.0
          # Note: to enable bagging, bagging_freq should be set to a non
          # zero value as well
          ParamDbl$new(
            id = "min_sum_hessian_in_leaf",
            default = 1e-3,
            lower = 0,
            tags = "train"),
          # % constraints: 0.0 < bagging_fraction <= 1.0
          ParamDbl$new(
            id = "bagging_fraction",
            default = 1.0,
            lower = 0.0,
            upper = 1.0,
            tags = "train"),
          # % constraints: 0.0 < pos_bagging_fraction <= 1.0
          # Note: to enable this, you need to set bagging_freq and
          # neg_bagging_fraction as well
          # Note: if both pos_bagging_fraction and neg_bagging_fraction
          # are set to 1.0, balanced bagging is disabled
          # Note: if balanced bagging is enabled, bagging_fraction will be ignored
          ParamDbl$new(
            id = "pos_bagging_fraction",
            default = 1.0,
            lower = 0.0,
            upper = 1.0,
            tags = "train"),
          # % constraints: 0.0 < neg_bagging_fraction <= 1.0
          ParamDbl$new(
            id = "neg_bagging_fraction",
            default = 1.0,
            lower = 0,
            upper = 1.0,
            tags = "train"),
          # Note: to enable bagging, bagging_fraction should be set to value
          # smaller than 1.0 as well
          ParamInt$new(
            id = "bagging_freq",
            default = 5L,
            lower = 0L,
            tags = "train"),
          ParamInt$new(
            id = "bagging_seed",
            default = 3L,
            tags = "train"),
          # % constraints: 0.0 < feature_fraction <= 1.0
          ParamDbl$new(
            id = "feature_fraction",
            default = 1.0,
            lower = 0.0,
            upper = 1.0,
            tags = "train"),
          # % constraints: 0.0 < feature_fraction_bynode <= 1.0
          # Note: unlike feature_fraction, this cannot speed up training
          # Note: if both feature_fraction and feature_fraction_bynode are
          # smaller than 1.0, the final fraction of each node is
          # % feature_fraction * feature_fraction_bynode
          ParamDbl$new(
            id = "feature_fraction_bynode",
            default = 1.0,
            lower = 0.0,
            upper = 1.0,
            tags = "train"),
          ParamInt$new(
            id = "feature_fraction_seed",
            default = 2L,
            tags = "train"),
          # <= 0 means disable
          ParamInt$new(
            id = "early_stopping_round",
            default = 0L,
            tags = "train"),
          ParamLgl$new(
            id = "first_metric_only",
            default = FALSE,
            tags = "train"),
          # <= 0 means no constraint
          ParamDbl$new(
            id = "max_delta_step",
            default = 0.0,
            tags = "train"),
          # % constraints: lambda_l1 >= 0.0
          ParamDbl$new(
            id = "lambda_l1",
            default = 0.0,
            lower = 0.0,
            tags = "train"),
          # % constraints: lambda_l2 >= 0.0
          ParamDbl$new(
            id = "lambda_l2",
            default = 0.0,
            lower = 0.0,
            tags = "train"),
          # % constraints: min_gain_to_split >= 0.0
          ParamDbl$new(
            id = "min_gain_to_split",
            default = 0.0,
            lower = 0.0,
            tags = "train"),
          # % constraints: 0.0 <= drop_rate <= 1.0
          ParamDbl$new(
            id = "drop_rate",
            default = 0.1,
            lower = 0.0,
            upper = 1.0,
            tags = "train"),
          # <=0 means no limit
          ParamInt$new(
            id = "max_drop",
            default = 50L,
            tags = "train"),
          # % constraints: 0.0 <= skip_drop <= 1.0
          ParamDbl$new(
            id = "skip_drop",
            default = 0.5,
            lower = 0.0,
            upper = 1.0,
            tags = "train"),
          ParamLgl$new(
            id = "xgboost_dart_mode",
            default = FALSE,
            tags = "train"),
          ParamLgl$new(
            id = "uniform_drop",
            default = FALSE,
            tags = "train"),
          ParamInt$new(
            id = "drop_seed",
            default = 4L,
            tags = "train"),
          # % constraints: 0.0 <= top_rate <= 1.0
          ParamDbl$new(
            id = "top_rate",
            default = 0.2,
            lower = 0.0,
            upper = 1.0,
            tags = "train"),
          # % constraints: 0.0 <= other_rate <= 1.0
          ParamDbl$new(
            id = "other_rate",
            default = 0.1,
            lower = 0.0,
            upper = 1.0,
            tags = "train"),
          # % constraints: min_data_per_group > 0
          ParamInt$new(
            id = "min_data_per_group",
            default = 100L,
            lower = 1L,
            tags = "train"),
          # % constraints: max_cat_threshold > 0
          ParamInt$new(
            id = "max_cat_threshold",
            default = 32L,
            lower = 1L,
            tags = "train"),
          # % constraints: cat_l2 >= 0.0
          ParamDbl$new(
            id = "cat_l2",
            default = 10.0,
            lower = 0.0,
            tags = "train"),
          # % constraints: cat_smooth >= 0.0
          ParamDbl$new(
            id = "cat_smooth",
            default = 10.0,
            lower = 0.0,
            tags = "train"),
          # % constraints: max_cat_to_onehot > 0
          ParamInt$new(
            id = "max_cat_to_onehot",
            default = 4L,
            lower = 1L,
            tags = "train"),
          # % constraints: top_k > 0
          ParamInt$new(
            id = "top_k",
            default = 20L,
            lower = 1L,
            tags = "train"),
          # % constraints: cegb_tradeoff >= 0.0
          ParamDbl$new(
            id = "cegb_tradeoff",
            default = 1.0,
            lower = 0.0,
            tags = "train"),
          # % constraints: cegb_penalty_split >= 0.0
          ParamDbl$new(
            id = "cegb_penalty_split",
            default = 0.0,
            lower = 0.0,
            tags = "train"),
          # IO Parameters
          ParamInt$new(
            id = "verbose",
            default = 1L,
            tags = "train"),
          # % constraints: max_bin > 1
          ParamInt$new(
            id = "max_bin",
            default = 255L,
            lower = 2L,
            tags = "train"),
          # % constraints: min_data_in_bin > 0
          ParamInt$new(
            id = "min_data_in_bin",
            default = 3L,
            lower = 1L,
            tags = "train"),
          # % constraints: bin_construct_sample_cnt > 0
          ParamInt$new(
            id = "bin_construct_sample_cnt",
            default = 200000L,
            lower = 1L,
            tags = "train"),
          # < 0 means no limit
          ParamDbl$new(
            id = "histogram_pool_size",
            default = -1.0,
            tags = "train"),
          ParamInt$new(
            id = "data_random_seed",
            default = 1L,
            tags = "train"),
          ParamInt$new(
            id = "snapshot_freq",
            default = -1L,
            tags = "train"),
          ParamLgl$new(
            id = "pre_partition",
            default = FALSE,
            tags = "train"),
          ParamLgl$new(
            id = "enable_bundle",
            default = TRUE,
            tags = "train"),
          # % constraints: 0.0 <= max_conflict_rate < 1.0
          ParamDbl$new(
            id = "max_conflict_rate",
            default = 0.0,
            lower = 0.0,
            upper = 1.0,
            tags = "train"),
          ParamLgl$new(
            id = "is_enable_sparse",
            default = TRUE,
            tags = "train"),
          # % constraints: 0.0 < sparse_threshold <= 1.0
          ParamDbl$new(
            id = "sparse_threshold",
            default = 0.8,
            lower = 0.0,
            upper = 1.0,
            tags = "train"),
          ParamLgl$new(
            id = "use_missing",
            default = TRUE,
            tags = "train"),
          ParamLgl$new(
            id = "zero_as_missing",
            default = FALSE,
            tags = "train"),
          ParamLgl$new(
            id = "two_round",
            default = FALSE,
            tags = "train"),
          ParamLgl$new(
            id = "save_binary",
            default = FALSE,
            tags = "train"),
          ParamLgl$new(
            id = "header",
            default = FALSE,
            tags = "train"),
          # Objective Parameters
          # % constraints: num_class > 0
          ParamInt$new(
            id = "num_class",
            default = 1L,
            lower = 1L,
            tags = "train"),
          ParamLgl$new(
            id = "is_unbalance",
            default = FALSE,
            tags = "train"),
          # % constraints: scale_pos_weight > 0.0
          ParamDbl$new(
            id = "scale_pos_weight",
            default = 1.0,
            lower = 0.0,
            tags = "train"),
          # % constraints: sigmoid > 0.0
          ParamDbl$new(
            id = "sigmoid",
            default = 1.0,
            lower = 0.0,
            tags = "train"),
          ParamLgl$new(
            id = "boost_from_average",
            default = FALSE,
            tags = "train"),
          ParamLgl$new(
            id = "reg_sqrt",
            default = FALSE,
            tags = "train"),
          # % constraints: alpha > 0.0
          ParamDbl$new(
            id = "alpha",
            default = 0.9,
            lower = 0.0,
            tags = "train"),
          # % constraints: fair_c > 0.0
          ParamDbl$new(
            id = "fair_c",
            default = 1.0,
            lower = 0.0,
            tags = "train"),
          # % constraints: poisson_max_delta_step > 0.0
          ParamDbl$new(
            id = "poisson_max_delta_step",
            default = 0.7,
            lower = 0.0,
            tags = "train"),
          # % constraints: 1.0 <= tweedie_variance_power < 2.0
          ParamDbl$new(
            id = "tweedie_variance_power",
            default = 1.5,
            lower = 1.0,
            upper = 2.0,
            tags = "train"),
          # % constraints: max_position > 0
          ParamInt$new(
            id = "max_position",
            default = 20L,
            lower = 1L,
            tags = "train"),
          ParamLgl$new(
            id = "lambdamart_norm",
            default = TRUE,
            tags = "train"),
          # Metric Parameters
          ParamFct$new(
            id = "metric",
            default = "",
            levels = c(
              "", "None", "l1", "mean_absolute_error",
              "mae", "regression_l1", "l2",
              "mean_squared_error", "mse", "regression_l2",
              "regression", "rmse", "root_mean_squared_error",
              "l2_root", "quantile", "lambdarank",
              "mean_absolute_percentage_error",
              "mean_average_precision",
              "mape", "huber", "fair", "poisson", "gamma",
              "gamma_deviance", "tweedie", "ndcg", "map",
              "cross_entropy", "cross_entropy_lambda",
              "kullback_leibler", "xentropy", "xentlambda",
              "kldiv",
              "multiclass", "softmax", "multiclassova",
              "multiclass_ova", "ova", "ovr", "binary",
              "binary_logloss", "binary_error",
              "multi_logloss", "auc", "multi_error"),
            tags = "train"),
          # % constraints: metric_freq > 0
          ParamInt$new(
            id = "metric_freq",
            default = 20L,
            lower = 1L,
            tags = "train"),
          ParamLgl$new(
            id = "is_provide_training_metric",
            default = FALSE,
            tags = "train"),
          # % constraints: multi_error_top_k > 0
          ParamInt$new(
            id = "multi_error_top_k",
            default = 1L,
            lower = 1L,
            tags = "train")
        )
      )

      # PATRICK: required params without an upstream default should
      # be tagged with "required" and have NO DEFAULT
      # they need to be set by the user (if at all)

      # self$autodetect_categorical = self$lgb_learner$autodetect_categorical

      super$initialize(
        id = "classif.lightgbm",
        packages = "lightgbm",
        feature_types = c(
          "numeric", "factor",
          "character", "integer"
        ),
        predict_types = "prob",
        param_set = ps,
        properties = c(
          "weights",
          "twoclass",
          "multiclass",
          "missings",
          "importance"),
        man = "mlr3learners.lightgbm::mlr_learners_classif_lightgbm"
      )

      # custom defaults
      ps$values = list(
        # FIXME: Add this change to the description of the help page
        # Be silent by default
        verbose = -1
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
        private$imp = self$lgb_learner$importance()
      }
      if (nrow(private$imp) != 0) {
        ret = sapply(private$imp$Feature, function(x) {
          return(private$imp[which(private$imp$Feature == x), ]$Gain)
        }, USE.NAMES = TRUE, simplify = TRUE)
      } else {
        ret = sapply(
          self$lgb_learner$train_data$get_colnames(),
          function(x) {
            return(0)
          },
          USE.NAMES = TRUE, simplify = FALSE)
      }

      return(unlist(ret))
    }
  ),

  private = list(

    # save importance values
    imp = NULL,

    # some pre training checks for this learner
    pre_train_checks = function(task) {

      n = nlevels(factor(task$data()[, get(task$target_names)]))

      if (is.null(self$param_set$values[["objective"]])) {
        # if not provided, set default objective depending on the
        # number of levels
        message("No objective provided...")
        if (n > 2) {
          self$param_set$values = mlr3misc::insert_named(
            self$param_set$values,
            list("objective" = "multiclass")
          )
          message("Setting objective to 'multiclass'")
        } else if (n == 2) {
          self$param_set$values = mlr3misc::insert_named(
            self$param_set$values,
            list("objective" = "binary")
          )
          message("Setting objective to 'binary'")
        } else {
          stop(paste0(
            "Please provide a target with a least ",
            "2 levels for classification tasks"))
        }

      } else {
        stopifnot(
          self$param_set$values[["objective"]] %in%
            c("binary", "multiclass", "multiclassova", "lambdarank")
        )
      }

      # set verbosity to 0L
      # % self$param_set$values = mlr3misc::insert_named(
      # %   self$param_set$values,
      # %   list("verbosity" = -1)
      # % )

      # pass all parameters to the learner
      self$lgb_learner$nrounds = self$nrounds
      self$lgb_learner$early_stopping_rounds = self$early_stopping_rounds
      self$lgb_learner$categorical_feature = self$categorical_feature
      self$lgb_learner$param_set = self$param_set
      self$lgb_learner$autodetect_categorical = self$autodetect_categorical
    },

    .train = function(task) {
      private$pre_train_checks(task)

      mlr3misc::invoke(
        .f = self$lgb_learner$train,
        task = task
      )
    },

    train_cv = function(task, row_ids) {
      if (is.null(self$model)) {

        task = mlr3::assert_task(as_task(task))
        mlr3::assert_learnable(task, self)

        row_ids = mlr3::assert_row_ids(row_ids)

        mlr3::assert_task(task)

        # subset to test set w/o cloning
        row_ids = assert_row_ids(row_ids)
        prev_use = task$row_roles$use
        on.exit({
          task$row_roles$use = prev_use
        }, add = TRUE)
        task$row_roles$use = row_ids

        private$pre_train_checks(task)

        self$lgb_learner$train_cv(task)

        self$cv_model = self$lgb_learner$cv_model

      } else {
        stop("A final model has already been trained!")
      }
    },

    .predict = function(task) {
      p = mlr3misc::invoke(
        .f = self$lgb_learner$predict,
        task = task
      )

      if (self$param_set$values[["objective"]] %in%
        c("multiclass", "multiclassova", "lambdarank")) {

        # process target variable
        c_names = as.character(unique(self$lgb_learner$label_names))

        c_names = plyr::revalue(
          x = c_names,
          replace = self$lgb_learner$trans_tar$value_mapping_dtrain
        )
        colnames(p) = c_names

      } else if (self$param_set$values[["objective"]] == "binary") {

        # reshape binary prob to matrix
        p = cbind(
          "0" = 1 - p,
          "1" = p
        )

        c_names = colnames(p)
        c_names = plyr::revalue(
          x = c_names,
          replace = self$lgb_learner$trans_tar$value_mapping_dtrain
        )
        colnames(p) = c_names
      }

      mlr3::PredictionClassif$new(
        task = task,
        prob = p
      )

    }
  )
)
