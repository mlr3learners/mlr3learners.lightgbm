learner <- LearnerClassifLightGBM$new()
expect_learner(learner)
learner$early_stopping_rounds <- 5
learner$nrounds <- 10

# run_autotest function
N = 30L
exclude = NULL
predict_types = learner$predict_types

learner = learner$clone(deep = TRUE)
id = learner$id
tasks = generate_tasks(learner, N = N)
if (!is.null(exclude))
  tasks = tasks[!grepl(exclude, names(tasks))]

task <- tasks$sanity_reordered

# test lgb plain:
dataset <- lightgbm::lgb.prepare(
  task$data()[, task$feature_names, with = F]
)
dataset <- lightgbm::lgb.Dataset(
  data = as.matrix(dataset),
  label = as.numeric(factor(task$data()[, get(task$target_names)])) - 1,
  colnames = task$feature_names,
  free_raw_data = FALSE
)
model <- lightgbm::lgb.train(params = list(objective = "binary"),
                             data = dataset,
                             nrounds = 10)
pred <- model$predict(as.matrix(task$data(cols = task$feature_names)), reshape = TRUE)
pred


predict_type <- "prob"
learner$id = sprintf("%s:%s", id, predict_type)
learner$predict_type = predict_type

# run_experiment function
err = function(info, ...) {
  info = sprintf(info, ...)
  list(
    ok = FALSE,
    task = task, learner = learner, prediction = prediction, score = score,
    error = sprintf("[%s] learner '%s' on task '%s' failed: %s",
                    stage, learner$id, task$id, info)
  )
}

task = mlr3::assert_task(mlr3::as_task(task))

# test lgb plain:
dataset <- lightgbm::lgb.prepare(
  task$data()[, task$feature_names, with = F]
)
dataset <- lightgbm::lgb.Dataset(
  data = as.matrix(dataset),
  label = as.numeric(factor(task$data()[, get(task$target_names)])) - 1,
  colnames = task$feature_names,
  free_raw_data = FALSE
)
model <- lightgbm::lgb.train(params = list(objective = "binary"),
                             data = dataset,
                             nrounds = 10)
pred <- model$predict(as.matrix(task$data(cols = task$feature_names)), reshape = TRUE)
pred

learner = mlr3::assert_learner(mlr3::as_learner(learner, clone = TRUE))
mlr3::assert_learnable(task, learner)
prediction = NULL
score = NULL
learner$encapsulate = c(train = "evaluate", predict = "evaluate")

# test lgb plain:
dataset <- lightgbm::lgb.prepare(
  task$data()[, task$feature_names, with = F]
)
dataset <- lightgbm::lgb.Dataset(
  data = as.matrix(dataset),
  label = as.numeric(factor(task$data()[, get(task$target_names)])) - 1,
  colnames = task$feature_names,
  free_raw_data = FALSE
)
model <- lightgbm::lgb.train(params = list(objective = "binary"),
                             data = dataset,
                             nrounds = 10)
pred <- model$predict(as.matrix(task$data(cols = task$feature_names)), reshape = TRUE)
pred

stage = "train()"
ok = try(learner$train(task), silent = TRUE)
if (inherits(ok, "try-error"))
  return(err(as.character(ok)))
log = learner$log[stage == "train"]
if ("error" %in% log$class)
  return(err("train log has errors: %s", mlr3misc::str_collapse(log[class == "error", msg])))
if (is.null(learner$model))
  return(err("model is NULL"))

stage = "predict()"

if (grepl("reordered", task$id)) {
  task$col_roles$feature = rev(task$col_roles$feature)
}

prediction = try(learner$predict(task), silent = TRUE)
prediction$confusion
if (inherits(ok, "try-error"))
  return(err(as.character(ok)))
log = learner$log[stage == "predict"]
if ("error" %in% log$class)
  return(err("predict log has errors: %s", mlr3misc::str_collapse(log[class == "error", msg])))
msg = checkmate::check_class(prediction, "Prediction")
if (!isTRUE(msg))
  return(err(msg))
if (prediction$task_type != learner$task_type)
  return(err("learner and prediction have different task_type"))

expected = mlr3::mlr_reflections$learner_predict_types[[learner$task_type]][[learner$predict_type]]
msg = checkmate::check_subset(expected, prediction$predict_types, empty.ok = FALSE)
if (!isTRUE(msg))
  return(err(msg))

if(learner$predict_type == "response"){
  msg = checkmate::check_set_equal(learner$predict_type, prediction$predict_types)
  if (!isTRUE(msg))
    return(err(msg))
} else {
  msg = checkmate::check_subset(learner$predict_type, prediction$predict_types, empty.ok = FALSE)
  if (!isTRUE(msg))
    return(err(msg))
}

stage = "score()"
score = try(prediction$score(mlr3::default_measures(learner$task_type)), silent = TRUE)
if (inherits(score, "try-error"))
  return(err(as.character(score)))
msg = checkmate::check_numeric(score, any.missing = FALSE)
if (!isTRUE(msg))
  return(err(msg))

# run sanity check on sanity task
if (grepl("^sanity", task$id) && !sanity_check(prediction)) {
  return(err("sanity check failed"))
}

if (grepl("^feat_all", task$id) && "importance" %in% learner$properties) {
  importance = learner$importance()
  msg = checkmate::check_numeric(rev(importance), any.missing = FALSE, min.len = 1L, sorted = TRUE)
  if (!isTRUE(msg))
    return(err(msg))
  msg = checkmate::check_names(names(importance), subset.of = task$feature_names)
  if (!isTRUE(msg))
    return(err("Names of returned importance scores do not match task names: %s", str_collapse(names(importance))))
  if ("unimportant" %in% head(names(importance), 1L))
    return(err("unimportant feature is important"))
}

if (grepl("^feat_all", task$id) && "selected_features" %in% learner$properties) {
  selected = learner$selected_features()
  msg = checkmate::check_subset(selected, task$feature_names)
  if (!isTRUE(msg))
    return(err(msg))
}

if (grepl("^feat_all", task$id) && "oob_error" %in% learner$properties) {
  err = learner$oob_error()
  msg = checkmate::check_number(err)
  if (!isTRUE(msg))
    return(err(msg))
}
