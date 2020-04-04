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

  public = list(

    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {

      # initialize ParamSet
      ps = ParamSet$new(
          # https://lightgbm.readthedocs.io/en/latest/Parameters.html#
          params = list(
            #######################################
            # Config Parameters
            ParamUty$new(id = "custom_eval",
                         default = NULL,
                         tags = "config"),
            ParamLgl$new(id = "nrounds_by_cv",
                         default = TRUE,
                         tags = "config"),
            ParamInt$new(id = "nfolds",
                         default = 5L,
                         lower = 3L,
                         tags = "config"),
            #######################################
            #######################################
            # Classification only
            ParamFct$new(id = "objective",
                         default = "binary",
                         levels = c("binary",
                                    "multiclass",
                                    "multiclassova",
                                    "cross_entropy",
                                    "cross_entropy_lambda",
                                    "rank_xendcg",
                                    "lambdarank"),
                         tags = "train"),
            # Objective Parameters
            #% constraints: num_class > 0
            ParamInt$new(id = "num_class",
                         default = 1L,
                         lower = 1L,
                         tags = c("train",
                                  "multi-class")),
            ParamLgl$new(id = "is_unbalance",
                         default = FALSE,
                         tags = c("train",
                                  "binary",
                                  "multiclassova")),
            #% constraints: scale_pos_weight > 0.0
            ParamDbl$new(id = "scale_pos_weight",
                         default = 1.0,
                         lower = 0.0,
                         tags = c("train",
                                  "binary",
                                  "multiclassova")),
            #% constraints: sigmoid > 0.0
            ParamDbl$new(id = "sigmoid",
                         default = 1.0,
                         lower = 0.0,
                         tags = c("train",
                                  "binary",
                                  "multiclassova",
                                  "lambdarank")),
            ParamInt$new(id = "lambdarank_truncation_level",
                         default = 20L,
                         lower = 1L,
                         tags = c("train",
                                  "lambdarank")),
            ParamLgl$new(id = "lambdarank_norm",
                         default = TRUE,
                         tags = c("train",
                                  "lambdarank")),
            # Metric Parameters
            ParamFct$new(id = "metric",
                         default = "",
                         levels = c("", "None",
                                    "ndcg", "lambdarank",
                                    "rank_xendcg", "xendcg",
                                    "xe_ndcg", "xe_ndcg_mart",
                                    "xendcg_mart", "map",
                                    "mean_average_precision",
                                    "cross_entropy",
                                    "cross_entropy_lambda",
                                    "kullback_leibler",
                                    "xentropy", "xentlambda",
                                    "kldiv", "multiclass",
                                    "softmax", "multiclassova",
                                    "multiclass_ova", "ova",
                                    "ovr", "binary",
                                    "binary_logloss",
                                    "binary_error", "auc_mu",
                                    "multi_logloss", "auc",
                                    "multi_error"),
                         tags = "train"),
            #% constraints: multi_error_top_k > 0
            ParamInt$new(id = "multi_error_top_k",
                         default = 1L,
                         lower = 1L,
                         tags = "train"),
            #######################################
            #######################################
            # Core Parameters
            ParamFct$new(id = "boosting",
                         default = "gbdt",
                         levels = c("gbdt",
                                    "rf",
                                    "dart",
                                    "goss"),
                         tags = "train"),
            #% constraints: num_iterations >= 0
            # Note: internally, LightGBM constructs
            # num_class * num_iterations
            # trees for multi-class classification problems
            ParamInt$new(id = "num_iterations",
                         default = 100L,
                         lower = 0L,
                         tags = "train"),
            #% constraints: learning_rate > 0.0
            ParamDbl$new(id = "learning_rate",
                         default = 0.1,
                         lower = 0.0,
                         tags = "train"),
            #% constraints: 1 < num_leaves <= 131072
            ParamInt$new(id = "num_leaves",
                         default = 31L,
                         lower = 1L,
                         upper = 131072L,
                         tags = "train"),
            ParamFct$new(id = "tree_learner",
                         default = "serial",
                         levels = c("serial",
                                    "feature",
                                    "data",
                                    "voting"),
                         tags = "train"),
            ParamInt$new(id = "num_threads",
                         default = 0L,
                         lower = 0L,
                         tags = "train"),
            ParamFct$new(id = "device_type",
                         default = "cpu",
                         levels = c("cpu", "gpu"),
                         tags = "train"),
            ParamUty$new(id = "seed",
                         default = "None",
                         tags = "train"),
            #######################################
            # Learning Control Parameters
            ParamLgl$new(id = "force_col_wise",
                         default = FALSE,
                         tags = "train"),
            ParamLgl$new(id = "force_row_wise",
                         default = FALSE,
                         tags = "train"),
            ParamDbl$new(id = "histogram_pool_size",
                         default = -1.0,
                         tags = "train"),
            #% <= 0 means no limit
            ParamInt$new(id = "max_depth",
                         default = -1L,
                         tags = "train"),
            #% constraints: min_data_in_leaf >= 0
            ParamInt$new(id = "min_data_in_leaf",
                         default = 20L,
                         lower = 0L,
                         tags = "train"),
            #% constraints: min_sum_hessian_in_leaf >= 0.0
            # Note: to enable bagging, bagging_freq
            # should be set to a non
            # zero value as well
            ParamDbl$new(id = "min_sum_hessian_in_leaf",
                         default = 1e-3,
                         lower = 0,
                         tags = "train"),
            #% constraints: 0.0 < bagging_fraction <= 1.0
            ParamDbl$new(id = "bagging_fraction",
                         default = 1.0,
                         lower = 0.0,
                         upper = 1.0,
                         tags = "train"),
            #% constraints: 0.0 < pos_bagging_fraction <= 1.0
            # Note: to enable this, you need to set bagging_freq and
            # neg_bagging_fraction as well
            # Note: if both pos_bagging_fraction and
            # neg_bagging_fraction
            # are set to 1.0, balanced bagging is disabled
            # Note: if balanced bagging is enabled,
            # bagging_fraction will be ignored
            ParamDbl$new(id = "pos_bagging_fraction",
                         default = 1.0,
                         lower = 0.0,
                         upper = 1.0,
                         tags = "train"),
            #% constraints: 0.0 < neg_bagging_fraction <= 1.0
            ParamDbl$new(id = "neg_bagging_fraction",
                         default = 1.0,
                         lower = 0,
                         upper = 1.0,
                         tags = "train"),
            # Note: to enable bagging, bagging_fraction
            # should be set to value
            # smaller than 1.0 as well
            ParamInt$new(id = "bagging_freq",
                         default = 0L,
                         lower = 0L,
                         tags = "train"),
            ParamInt$new(id = "bagging_seed",
                         default = 3L,
                         tags = "train"),
            #% constraints: 0.0 < feature_fraction <= 1.0
            ParamDbl$new(id = "feature_fraction",
                         default = 1.0,
                         lower = 0.0,
                         upper = 1.0,
                         tags = "train"),
            #% constraints: 0.0 < feature_fraction_bynode <= 1.0
            # Note: unlike feature_fraction, this cannot
            # speed up training
            # Note: if both feature_fraction and
            # feature_fraction_bynode are
            # smaller than 1.0, the final fraction of
            # each node is
            #% feature_fraction * feature_fraction_bynode
            ParamDbl$new(id = "feature_fraction_bynode",
                         default = 1.0,
                         lower = 0.0,
                         upper = 1.0,
                         tags = "train"),
            ParamInt$new(id = "feature_fraction_seed",
                         default = 2L,
                         tags = "train"),
            ParamLgl$new(id = "extra_trees",
                         default = FALSE,
                         tags = "train"),
            ParamInt$new(id = "extra_seed",
                         default = 6L,
                         tags = "train"),
            # <= 0 means disable
            ParamInt$new(id = "early_stopping_round",
                         default = 0L,
                         tags = "train"),
            ParamLgl$new(id = "first_metric_only",
                         default = FALSE,
                         tags = "train"),
            # <= 0 means no constraint
            ParamDbl$new(id = "max_delta_step",
                         default = 0.0,
                         tags = "train"),
            #% constraints: lambda_l1 >= 0.0
            ParamDbl$new(id = "lambda_l1",
                         default = 0.0,
                         lower = 0.0,
                         tags = "train"),
            #% constraints: lambda_l2 >= 0.0
            ParamDbl$new(id = "lambda_l2",
                         default = 0.0,
                         lower = 0.0,
                         tags = "train"),
            #% constraints: min_gain_to_split >= 0.0
            ParamDbl$new(id = "min_gain_to_split",
                         default = 0.0,
                         lower = 0.0,
                         tags = "train"),
            #% constraints: 0.0 <= drop_rate <= 1.0
            ParamDbl$new(id = "drop_rate",
                         default = 0.1,
                         lower = 0.0,
                         upper = 1.0,
                         tags = c("train", "dart")),
            # <=0 means no limit
            ParamInt$new(id = "max_drop",
                         default = 50L,
                         tags = c("train", "dart")),
            #% constraints: 0.0 <= skip_drop <= 1.0
            ParamDbl$new(id = "skip_drop",
                         default = 0.5,
                         lower = 0.0,
                         upper = 1.0,
                         tags = c("train", "dart")),
            ParamLgl$new(id = "xgboost_dart_mode",
                         default = FALSE,
                         tags = c("train", "dart")),
            ParamLgl$new(id = "uniform_drop",
                         default = FALSE,
                         tags = c("train", "dart")),
            ParamInt$new(id = "drop_seed",
                         default = 4L,
                         tags = c("train", "dart")),
            #% constraints: 0.0 <= top_rate <= 1.0
            ParamDbl$new(id = "top_rate",
                         default = 0.2,
                         lower = 0.0,
                         upper = 1.0,
                         tags = c("train", "goss")),
            #% constraints: 0.0 <= other_rate <= 1.0
            ParamDbl$new(id = "other_rate",
                         default = 0.1,
                         lower = 0.0,
                         upper = 1.0,
                         tags = c("train", "goss")),
            #% constraints: min_data_per_group > 0
            ParamInt$new(id = "min_data_per_group",
                         default = 100L,
                         lower = 1L,
                         tags = "train"),
            #% constraints: max_cat_threshold > 0
            ParamInt$new(id = "max_cat_threshold",
                         default = 32L,
                         lower = 1L,
                         tags = "train"),
            #% constraints: cat_l2 >= 0.0
            ParamDbl$new(id = "cat_l2",
                         default = 10.0,
                         lower = 0.0,
                         tags = "train"),
            #% constraints: cat_smooth >= 0.0
            ParamDbl$new(id = "cat_smooth",
                         default = 10.0,
                         lower = 0.0,
                         tags = "train"),
            #% constraints: max_cat_to_onehot > 0
            ParamInt$new(id = "max_cat_to_onehot",
                         default = 4L,
                         lower = 1L,
                         tags = "train"),
            #% constraints: top_k > 0
            ParamInt$new(id = "top_k",
                         default = 20L,
                         lower = 1L,
                         tags = "train"),
            #% constraints: cegb_tradeoff >= 0.0
            ParamDbl$new(id = "cegb_tradeoff",
                         default = 1.0,
                         lower = 0.0,
                         tags = "train"),
            #% constraints: cegb_penalty_split >= 0.0
            ParamDbl$new(id = "cegb_penalty_split",
                         default = 0.0,
                         lower = 0.0,
                         tags = "train"),
            #######################################
            # IO Parameters
            ParamInt$new(id = "verbose",
                         default = 1L,
                         tags = "train"),
            ParamUty$new(id = "input_model",
                         default = "",
                         tags = "train"),
            ParamUty$new(id = "output_model",
                         default = "LightGBM_model.txt",
                         tags = "train"),
            ParamInt$new(id = "snapshot_freq",
                         default = -1L,
                         tags = "train"),
            #% constraints: max_bin > 1
            ParamInt$new(id = "max_bin",
                         default = 255L,
                         lower = 2L,
                         tags = "train"),
            #% constraints: min_data_in_bin > 0
            ParamInt$new(id = "min_data_in_bin",
                         default = 3L,
                         lower = 1L,
                         tags = "train"),
            #% constraints: bin_construct_sample_cnt > 0
            ParamInt$new(id = "bin_construct_sample_cnt",
                         default = 200000L,
                         lower = 1L,
                         tags = "train"),
            ParamInt$new(id = "data_random_seed",
                         default = 1L,
                         tags = "train"),
            ParamLgl$new(id = "is_enable_sparse",
                         default = TRUE,
                         tags = "train"),
            ParamLgl$new(id = "enable_bundle",
                         default = TRUE,
                         tags = "train"),
            ParamLgl$new(id = "use_missing",
                         default = TRUE,
                         tags = "train"),
            ParamLgl$new(id = "zero_as_missing",
                         default = FALSE,
                         tags = "train"),
            ParamLgl$new(id = "feature_pre_filter",
                         default = TRUE,
                         tags = "train"),
            ParamLgl$new(id = "pre_partition",
                         default = FALSE,
                         tags = "train"),
            ParamLgl$new(id = "two_round",
                         default = FALSE,
                         tags = "train"),
            ParamLgl$new(id = "header",
                         default = FALSE,
                         tags = "train"),
            ParamUty$new(id = "group_column",
                         default = "",
                         tags = ""),
            ParamUty$new(id = "ignore_column",
                         default = "",
                         tags = "train"),
            ParamUty$new(id = "categorical_feature",
                         default = "",
                         tags = "train"),
            #######################################
            #######################################
            # Predict Parameters TODO are they needed?
            # Convert Parameters TODO are they needed?
            #######################################
            #######################################
            # Objective Parameters
            ParamInt$new(id = "objective_seed",
                         default = 5L,
                         tags = c("train", "rank_xendcg")),
            # moved num_class up to classification part
            # moved is_unbalance up to classification part
            # moved scale_pos_weight up to classification part
            # moved sigmoid up to classification part
            ParamLgl$new(id = "boost_from_average",
                         default = TRUE,
                         tags = c("train", "regression", "binary",
                                  "multiclassova", "cross-entropy")),
            # moved req_sqrt up to regression part
            # moved alpha up to regression part
            # moved fair_c up to regression part
            # moved poisson_max_delta_step up to regression part
            # moved tweedie_variance_power up to regression part
            # moved lambdarank_truncation_level up to classification part
            # moved lambdarank_norm up to classification part
            # moved label_gain up to classification part
            #######################################
            # Metric Parameters
            #% constraints: metric_freq > 0
            ParamInt$new(id = "metric_freq",
                         default = 1L,
                         lower = 1L,
                         tags = "train"),
            ParamLgl$new(id = "is_provide_training_metric",
                         default = FALSE,
                         tags = "train")
          )
        )

      # instantiate the learner
      private$lgb_superclass <- LightGBM$new(ps)

      super$initialize(
        # see the mlr3book for a description:
        # https://mlr3book.mlr-org.com/extending-mlr3.html
        id = "classif.lightgbm",
        packages = "lightgbm",
        feature_types = c(
          "numeric", "factor",
          "character", "integer"
        ),
        predict_types = "prob",
        param_set = ps,
        properties = c("weights",
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
        private$imp <- private$lgb_superclass$importance()
      }
      if (nrow(private$imp) != 0) {
        ret <- sapply(private$imp$Feature, function(x) {
          return(private$imp[which(private$imp$Feature == x), ]$Gain)
        }, USE.NAMES = TRUE, simplify = TRUE)
      } else {
        ret <- sapply(
          private$lgb_superclass$train_data$get_colnames(),
          function(x) {
            return(0)
          }, USE.NAMES = TRUE, simplify = FALSE)
      }

      return(unlist(ret))
    }
  ),

  private = list(

    # save importance values
    imp = NULL,

    # The lightgbm learner instance
    lgb_superclass = NULL,

    # some pre training checks for this learner
    pre_train_checks = function(task) {
      n <- nlevels(factor(task$data()[, get(task$target_names)]))

      if (is.null(self$param_set$values[["objective"]])) {
        # if not provided, set default objective depending on the
        # number of levels
        message("No objective provided...")
        if (n > 2) {
          self$param_set$values <- mlr3misc::insert_named(
            self$param_set$values,
            list("objective" = "multiclass")
          )
          message("Setting objective to 'multiclass'")
        } else if (n == 2) {
          self$param_set$values <- mlr3misc::insert_named(
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
    },

    .train = function(task) {

      private$pre_train_checks(task)

      if (!is.null(self$param_set$values[["custom_eval"]])) {
        self$param_set$values$metric <- "None"
      }

      pars = self$param_set$get_values(tags = "train")

      mlr3misc::invoke(
        .f = private$lgb_superclass$train,
        task = task,
        params = pars,
        feval = self$param_set$values[["custom_eval"]],
        nrounds_by_cv = self$param_set$values[["nrounds_by_cv"]],
        nfolds = self$param_set$values[["nfolds"]]
      ) # use the mlr3misc::invoke function (it's similar to do.call())
    },

    .predict = function(task) {

      p <- mlr3misc::invoke(
        .f = private$lgb_superclass$predict,
        task = task
      )

      if (self$param_set$values[["objective"]] %in%
          c("multiclass", "multiclassova", "lambdarank")) {

        # process target variable
        c_names <- as.character(unique(private$lgb_superclass$label_names))

        c_names <- plyr::revalue(
          x = c_names,
          replace = private$lgb_superclass$trans_tar$value_mapping_dtrain
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
          replace = private$lgb_superclass$trans_tar$value_mapping_dtrain
        )
        colnames(p) <- c_names
      }

      mlr3::PredictionClassif$new(
        task = task,
        prob = p
      )

    }
  )
)
