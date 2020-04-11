#' @title importance_plot
#'
#' @description The function returns a barplot for feature importance vectors.
#'
#' @param importance A named vector of importance values. If supported,
#'   the output of \code{learner$importance}.
#' @param n An integer. The maximum number of importance values included
#'   into the plot (default: NULL, indicating all importance values will
#'   be plotted). Restrictions: 0 < n <= `length(importance)`.
#' @param threshold A numeric. The threshold of importance values to be
#'   included into the plot (default: NULL). Restrictions: 0 < threshold < 1.
#'   Please note, that if `n` and `threshold` are provided, `threshold` is
#'   ignored.
#'
#' @return An importance plot.
#'
#' @export
#'
importance_plot = function(importance, n = NULL, threshold = NULL) {
  stopifnot(
    is.vector(importance)
    , is.numeric(importance)
    , is.character(names(importance))
    , is.null(n) || (is.numeric(n) && n > 0 && n <= length(importance))
    , is.null(threshold) || (
      is.numeric(threshold) && threshold > 0 && threshold < 1
    )
  )

  # priorize n (if n and threshold are provided)
  if (!is.null(n)) {
    importance <- importance[1:n]
  } else if (!is.null(threshold)) {
    importance <- importance[which(importance >= threshold)]
  }

  # create plot
  imp_plot <- ggplot2::ggplot(
    data = NULL,
    ggplot2::aes(x = reorder(names(importance), importance),
                 y = importance,
                 fill = importance)
  ) +
    ggplot2::geom_col() +
    ggplot2::coord_flip() +
    ggplot2::scale_fill_continuous(type = "viridis") +
    ggplot2::ylab("Importance") +
    ggplot2::xlab("Feature") +
    ggplot2::theme(legend.position = "none")

  # return plot
  return(imp_plot)
}
