library(mlbench)
data("Glass")
dataset = data.table::as.data.table(Glass)
dataset[, ("Type") := factor(as.numeric(get("Type")) - 1L)]

train.index = caret::createDataPartition(
  y = dataset$Type,
  times = 1,
  p = 0.7
)
valid.index = setdiff(1:nrow(dataset), train.index$Resample1)

task = TaskClassif$new(
  id = "Glass",
  target = "Type",
  backend = dataset
)

ps = lgbparams()

ps$values = list("metric" = "multi_logloss")
ps$values = list("learning_rate" = 0.01)
ps$values = list("bagging_fraction" = 0.6)


lightgbm = reticulate::import("lightgbm")
pars = ps$get_values(tags = "train")

# Get formula, data, classwt, cutoff for the LightGBM
data = task$data() # the data is avail
levs = levels(data[[task$target_names]])
n = length(levs)

# lightgbm needs numeric values
if (is.factor(data[[task$target_names]])) {
  data[, (task$target_names) := as.numeric(
    as.character(get(task$target_names))
  )]
}


# numeric values need to start at 0
stopifnot(
  min(data[[task$target_names]]) == 0,
  n > 1
)

if (n > 2) {
  pars[["objective"]] = "multiclass"
  pars[["num_class"]] = n
  if (is.null(pars[["metric"]])) {
    pars[["metric"]] = c("multi_logloss", "multi_error")
  }
} else {
  pars[["objective"]] = "binary"
  if (is.null(pars[["metric"]])) {
    pars[["metric"]] = c("auc", "binary_error")
  }
}

x_train = as.matrix(data[, task$feature_names, with = FALSE])
for (i in colnames(x_train)) {
  x_train[which(is.na(x_train[, i])), i] = NaN
}
x_label = data[, get(task$target_names)]

mymodel = lightgbm$train(
  train_set = lightgbm$Dataset(
    data = x_train,
    label = x_label
  ),
  params = pars,
  early_stopping_rounds = 1000L,
  verbose_eval = 50L,
  valid_sets = lightgbm$Dataset(
    data = x_train,
    label = x_label
  )
)

newdata = task$data(cols = task$feature_names)

p = mlr3misc::invoke(
  .f = mymodel$predict,
  data = newdata,
  is_reshape = T)
colnames(p) = as.character(unique(x_label))

PredictionClassif$new(task = task, prob = p)


imp = data.table::data.table(
  "Feature" = mymodel$feature_name(),
  "Value" = as.numeric(as.character(mymodel$feature_importance()))
)[order(get("Value"), decreasing = T)]

imp

ggplot2::ggplot(
  data = NULL,
  ggplot2::aes(
    x = reorder(imp$Feature, imp$Value),
    y = imp$Value,
    fill = imp$Value)) +
  ggplot2::geom_col() +
  ggplot2::coord_flip() +
  ggplot2::scale_fill_gradientn(colours = grDevices::rainbow(n = nrow(imp))) +
  ggplot2::labs(title = "LightGBM Feature Importance") +
  ggplot2::ylab("Feature") +
  ggplot2::xlab("Importance") +
  ggplot2::theme(legend.position = "none")

imp
