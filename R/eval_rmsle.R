#' @title lgb_rmsle custom metric
#'
#' @description RMSLE custom evaluation metric function
#'
#' @param preds A vector containing the predicted values
#' @param dtrain The training dataset.
#'
#' @export

lgb_rmsle = function(preds, dtrain) {
  label = lightgbm::getinfo(dtrain, "label")
  score = MLmetrics::RMSLE(y_pred = preds, y_true = label)
  return(list(name = "rmsle", value = score, higher_better = FALSE))
}
