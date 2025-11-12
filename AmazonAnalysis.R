library(vroom)
#library(DataExplorer)
library(dplyr)
library(tidymodels)
library(discrim)
library(embed)
library(keras)
library(reticulate)
library(kernlab)
library(themis)
# py_require_legacy_keras()
# py_require("tensorflow")



# Upload Data ------------------------------------------------------------------
train_data <- vroom("train.csv")
test_data <- vroom("test.csv")

# EDA --------------------------------------------------------------------------

# DataExplorer::plot_intro(train_data)         # Need to change continuous variable to discrete
# DataExplorer::plot_correlation(train_data)
# DataExplorer::plot_missing(train_data)       # None!!!! :)


# Recipe -----------------------------------------------------------------------

train_data$ACTION <- as.factor(train_data$ACTION)

my_recipe <- recipe(ACTION~., data=train_data) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  #step_dummy(all_nominal_predictors()) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  #step_smote(all_outcomes(), neighbors=3) %>%
  step_upsample(all_outcomes())
  #step_downsample(all_outcomes())
  #step_normalize(all_predictors())
  #step_pca(all_predictors(), threshold=.32)



# # apply the recipe to your data1
# prep <- prep(my_recipe)
# baked <- bake(prep, new_data = train_data)

# Logistic Regression ----------------------------------------------------------

# logreg_mod <- logistic_reg() %>%
#   set_engine("glm")
# 
# # Put into a workflow
# logreg_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(logreg_mod) %>%
#   fit(data=train_data)
# 
# # Make Predictions
# logreg_preds <- predict(logreg_wf, new_data=test_data, type="prob")
# 
# # Format for Kaggle
# kag_sub <- data.frame(id = test_data$id,
#                       ACTION = logreg_preds$.pred_1)
# vroom_write(x=kag_sub, file="./LogregPreds_smote.csv", delim=",")


# Pentalized Logistic Regression -----------------------------------------------

# p_logreg_mod <- logistic_reg(mixture=tune(), penalty=tune()) %>%
#   set_engine("glmnet")
# 
# p_logreg_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(p_logreg_mod)
# 
# tuning_grid <- grid_regular(penalty(), mixture(), levels = 3)
# folds <- vfold_cv(train_data, v = 5, repeats=1)
# 
# cv_results <- p_logreg_wf %>%
#   tune_grid(resamples=folds, grid=tuning_grid, metrics=metric_set(roc_auc))
# best_tune <- cv_results %>%
#   select_best(metric="roc_auc")
# final_wf <- p_logreg_wf %>%
#   finalize_workflow(best_tune) %>%
#   fit(data=train_data)
# 
# # Make Predictions
# p_logreg_preds <- predict(final_wf, new_data=test_data, type="prob")
# 
# # Format for Kaggle
# kag_sub <- data.frame(id = test_data$id,
#                       ACTION = p_logreg_preds$.pred_1)
# vroom_write(x=kag_sub, file="./PLogregPreds_pcd2.csv", delim=",")


# Random Forest ----------------------------------------------------------------

# forest_mod <- rand_forest(mtry = tune(), min_n=tune(), trees=500) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")

forest_mod <- rand_forest(mtry = 1, min_n=10, trees=1500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# # Create a workflow with model & recipe
# forest_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(forest_mod)

forest_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(forest_mod)%>%
  fit(data=train_data)

# # Set up grid of tuning values and K-fold
# #tuning_grid <- grid_regular(mtry(range=c(1,9)), min_n(), levels=3)
# tuning_grid <- grid_regular(mtry(range = c(1, 9)), min_n(range = c(1, 10)), levels = 5)

# folds <- vfold_cv(train_data, v = 5, repeats=1)
# 
# # Find best tuning parameters
# cv_results <- forest_wf %>%
#   tune_grid(resamples=folds, grid=tuning_grid, metrics=metric_set(roc_auc))
# best_tune <- cv_results %>%
#   select_best(metric="roc_auc")
# 
# # Finalize workflow and predict
# final_wf <- forest_wf %>%
#   finalize_workflow(best_tune) %>%
#   fit(data=train_data)

# Make Predictions
forest_preds <- predict(forest_wf, new_data=test_data, type="prob")

# Format for Kaggle
kag_sub <- data.frame(id = test_data$id,
                      ACTION = forest_preds$.pred_1)
vroom_write(x=kag_sub, file="./Forest_matt.csv", delim=",")


# KNN Model --------------------------------------------------------------------

# knn_mod <- nearest_neighbor(neighbors=tune()) %>%
#   set_mode("classification") %>%
#   set_engine("kknn")
# 
# knn_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(knn_mod)
# 
# # Set up grid of tuning values and K-fold
# tuning_grid <- grid_regular(neighbors(), levels=3)
# folds <- vfold_cv(train_data, v = 5, repeats=1)
# 
# # Find best tuning parameters
# cv_results <- knn_wf %>%
#   tune_grid(resamples=folds, grid=tuning_grid, metrics=metric_set(roc_auc))
# best_tune <- cv_results %>%
#   select_best(metric="roc_auc")
# 
# # Finalize workflow and predict
# final_wf <- knn_wf %>%
#   finalize_workflow(best_tune) %>%
#   fit(data=train_data)
# 
# # Make Predictions
# knn_preds <- predict(final_wf, new_data=test_data, type="prob")
# 
# # Format for Kaggle
# kag_sub <- data.frame(id = test_data$id,
#                       ACTION = knn_preds$.pred_1)
# vroom_write(x=kag_sub, file="./KNNPreds.csv", delim=",")


# Naive Bayes ------------------------------------------------------------------

# nb_mod <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
#   set_mode("classification") %>%
#   set_engine("naivebayes")
# 
# nb_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(nb_mod)
# 
# # Set up grid of tuning values and K-fold
# tuning_grid <- grid_regular(Laplace(), smoothness(), levels=3)
# folds <- vfold_cv(train_data, v = 5, repeats=1)
# 
# # Find best tuning parameters
# cv_results <- nb_wf %>%
#   tune_grid(resamples=folds, grid=tuning_grid, metrics=metric_set(roc_auc))
# best_tune <- cv_results %>%
#   select_best(metric="roc_auc")
# 
# # Finalize workflow and predict
# final_wf <- nb_wf %>%
#   finalize_workflow(best_tune) %>%
#   fit(data=train_data)
# 
# # Make Predictions
# nb_preds <- predict(final_wf, new_data=test_data, type="prob")
# 
# # Format for Kaggle
# kag_sub <- data.frame(id = test_data$id,
#                       ACTION = nb_preds$.pred_1)
# vroom_write(x=kag_sub, file="./NbPreds.csv", delim=",")


# Neural Network ---------------------------------------------------------------

# # Scale recipe to [0,1]
# nn_recipe <- recipe(ACTION~., data=train_data) %>%
#   step_mutate_at(all_numeric_predictors(), fn=factor) %>%
#   step_other(all_nominal_predictors(), threshold = .001) %>%
#   step_dummy(all_nominal_predictors()) %>%
#   step_range(all_numeric_predictors(), min=0, max=1)
# 
# nn_mod <- mlp(hidden_units = tune(), epochs = 50) %>%
#   set_engine("keras") %>%
#   set_mode("classification")
# 
# nn_wf <- workflow() %>%
#   add_recipe(nn_recipe) %>%
#   add_model(nn_mod)
# 
# tuning_grid <- grid_regular(hidden_units(range=c(1, 17)),levels=3)
# folds <- vfold_cv(train_data, v = 5, repeats=1)
# 
# # Find Best Tuning Parameters
# cv_results <- nn_wf %>%
#   tune_grid(resamples=folds, grid=tuning_grid, metrics=metric_set(roc_auc))
# best_tune <- cv_results %>%
#   select_best(metric="roc_auc")
# 
# # Make Graph
# cv_results %>% collect_metrics() %>%
# filter(.metric=="accuracy") %>%
# ggplot(aes(x=hidden_units, y=mean)) + geom_line()
# 
# # Finalize workflow and predict
# final_wf <- nn_wf %>%
#   finalize_workflow(best_tune) %>%
#   fit(data=train_data)
# 
# # Make Predictions
# nn_preds <- predict(final_wf, new_data=test_data, type="prob")
# 
# # Format for Kaggle
# kag_sub <- data.frame(id = test_data$id,
#                       ACTION = nb_preds$.pred_1)
# vroom_write(x=kag_sub, file="./NnPreds.csv", delim=",")
# 


# SVN Poly ---------------------------------------------------------------------

# svm_poly_mod <- svm_poly(degree=1, cost=.0131) %>%
#   set_mode("classification") %>%
#   set_engine("kernlab")
# 
# svm_poly_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(svm_poly_mod) %>%
#   fit(data=train_data)
# 
# # Make Predictions
# svm_poly_preds <- predict(svm_poly_wf, new_data=test_data, type="prob")
# 
# # Format for Kaggle
# kag_sub <- data.frame(id = test_data$id,
#                       ACTION = svm_poly_preds$.pred_1)
# vroom_write(x=kag_sub, file="./SvmPolyPreds.csv", delim=",")

# SVN Radial -------------------------------------------------------------------

# svm_radial_mod <- svm_rbf(rbf_sigma=.177, cost=.00316) %>%
#   set_mode("classification") %>%
#   set_engine("kernlab")
# 
# svm_radial_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(svm_radial_mod) %>%
#   fit(data=train_data)
# 
# # Make Predictions
# svm_radial_preds <- predict(svm_radial_wf, new_data=test_data, type="prob")
# 
# # Format for Kaggle
# kag_sub <- data.frame(id = test_data$id,
#                       ACTION = svm_radial_preds$.pred_1)
# vroom_write(x=kag_sub, file="./SvmRadialPreds.csv", delim=",")

# SVN Linear -------------------------------------------------------------------

# svm_linear_mod <- svm_linear(cost=.0131) %>%
#   set_mode("classification") %>%
#   set_engine("kernlab")
# 
# svm_linear_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(svm_linear_mod) %>%
#   fit(data=train_data)
# 
# # Make Predictions
# svm_linear_preds <- predict(svm_linear_wf, new_data=test_data, type="prob")
# 
# # Format for Kaggle
# kag_sub <- data.frame(id = test_data$id,
#                       ACTION = svm_linear_preds$.pred_1)
# vroom_write(x=kag_sub, file="./SvmLinearPreds.csv", delim=",")









