#' @title lgb_rmsle custom metric
#'
#' @description PR-AUC custom evaluation metric function
#'
#' @inheritParams lgb_rmsle
#'
#' @export
#'
lgb_prauc = function(preds, dtrain) {
  # https://stats.stackexchange.com/questions/10501/calculating-aupr-in-r
  # area under the precision-recall-curve is better for unbalanced targets
  label = dtrain$getinfo("label")
  #% probs <- data.table::data.table(
  #%   cbind(labels = label,
  #%         preds = preds)
  #% )
  #% score = PRROC::pr.curve(
  #%   scores.class0 = probs[labels == "1", preds],
  #%   scores.class1 = probs[labels == "0", preds],
  #%   curve = FALSE
  #% )[["auc.integral"]]
  score = MLmetrics::PRAUC(y_pred = preds, y_true = label)
  return(list(name = "prauc", value = score, higher_better = TRUE))
}
