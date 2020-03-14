TransformTarget <- R6::R6Class(
  "TransformTarget",

  public = list(
    param_set = NULL,

    value_mapping_dtrain = NULL,
    value_mapping_dvalid = NULL,


    initialize = function(param_set) {

      self$param_set <- param_set

    },

    transform_target = function(vector, positive, negative, mapping) {

      stopifnot(
        mapping %in% c("dvalid", "dtrain")
      )

      # transform target to numeric for multiclass classification tasks
      if (self$param_set$values[["objective"]] %in%
          c("binary", "multiclass", "multiclassova", "lambdarank")) {

        error <- FALSE

        if (self$param_set$values[["objective"]] == "binary") {
          stopifnot(
            is.character(positive),
            is.character(negative)
          )

          repl <- c(0, 1)
          names(repl) <- c(eval(negative), eval(positive))

          vector <- as.integer(plyr::revalue(
            x = as.character(vector),
            replace = repl
          ))

          new_levels <- unname(repl)
          old_levels <- names(repl)
          names(old_levels) <- new_levels
          self[[paste0("value_mapping_", mapping)]] <- old_levels

        } else {

          old_levels <- factor(levels(factor(sort(vector))))

          # if target is not numeric
          if (!is.numeric(vector)) {

            # store target as integer (less memory), beginning
            # with "0"
            vector <- (as.integer(factor(vector)) - 1L)

            new_levels <- (as.integer(old_levels) - 1L)

            old_levels <- as.character(old_levels)
            names(old_levels) <- new_levels
            self[[paste0("value_mapping_", mapping)]] <- old_levels

            # if target is numeric
          } else if (is.numeric(vector)) {

            vector <- as.integer(round(vector, 0))
            new_levels <- as.integer(old_levels)

            # check, if minimum != 0 --> if == 0, we have nothing to do
            if (min(vector) != 0) {

              # if min == 1, substract 1 --> lightgbm need the first class
              # to be 0
              if (min(vector) == 1) {
                vector <- as.integer(vector) - 1L
                new_levels <- as.integer(old_levels) - 1L

                # else stop with warning
              } else {
                error <- TRUE
              }
            }
            old_levels <- as.character(old_levels)
            names(old_levels) <- new_levels
            self[[paste0("value_mapping_", mapping)]] <- old_levels
          }

          if (error) {
            stop(
              paste0("Please provide a valid target variable ",
                     "for classification tasks")
            )
          }
        }

        # transform numeric variables
      } else {

        # we have only work here, if target is not numeric, but a
        # character or factor vector holding numerics
        if (!is.numeric(vector)) {

          # try to transform to numeric
          transform_error <- tryCatch(
            expr = {
              vector <- as.numeric(
                as.character(vector)
              )
              ret <- FALSE
              ret
            }, error = function(e) {
              print(e)
              ret <- TRUE
              ret
            }, finally = function(f) {
              return(ret)
            }
          )
          stopifnot(isFALSE(transform_error))
        }
      }
      return(vector)
    }
  )
)
