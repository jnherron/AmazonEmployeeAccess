library(vroom)
library(dplyr)
library(tidymodels)
library(discrim)
library(embed)
library(keras)
library(reticulate)
library(kernlab)
library(themis)

train_data <- vroom("train.csv")
test_data <- vroom("test.csv")

train_data$ACTION <- as.factor(train_data$ACTION)

my_recipe <- recipe(ACTION~., data=train_data) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_upsample(all_outcomes())




forest_mod <- rand_forest(mtry = tune(), min_n=tune(), trees=2000) %>%
  set_engine("ranger") %>%
  set_mode("classification")

forest_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(forest_mod)

tuning_grid <- grid_regular(mtry(range = c(1, 9)), min_n(range = c(1, 10)), levels = 5)
folds <- vfold_cv(train_data, v = 5, repeats=1)

cv_results <- forest_wf %>%
  tune_grid(resamples=folds, grid=tuning_grid, metrics=metric_set(roc_auc))
best_tune <- cv_results %>%
  select_best(metric="roc_auc")

final_wf <- forest_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data=train_data)

forest_preds <- predict(final_wf, new_data=test_data, type="prob")

kag_sub <- data.frame(id = test_data$id,
                      ACTION = forest_preds$.pred_1)
vroom_write(x=kag_sub, file="./Forest_server.csv", delim=",")










