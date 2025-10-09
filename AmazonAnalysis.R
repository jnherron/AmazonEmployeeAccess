library(vroom)
library(DataExplorer)
library(dplyr)
library(tidymodels)




# Upload Data ------------------------------------------------------------------
train_data <- vroom("train.csv")
test_data <- vroom("test.csv")

# EDA --------------------------------------------------------------------------
DataExplorer::plot_intro(train_data)         # Need to change continuous variable to discrete
DataExplorer::plot_correlation(train_data)
DataExplorer::plot_missing(train_data)       # None!!!! :)


# Recipe -----------------------------------------------------------------------
train_data$ACTION <- as.factor(train_data$ACTION)

my_recipe <- recipe(ACTION~., data=train_data) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_dummy(all_nominal_predictors())

# apply the recipe to your data1
prep <- prep(my_recipe)
baked <- bake(prep, new_data = train_data)



# Logistic Regression ----------------------------------------------------------
logreg_mod <- logistic_reg() %>%
  set_engine("glm")

# Put into a workflow
logreg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(logreg_mod) %>%
  fit(data=train_data)

# Make Predictions
logreg_preds <- predict(logreg_wf, new_data=test_data, type="prob")

# Format for Kaggle
kag_sub <- data.frame(id = test_data$id,
                      ACTION = logreg_preds$.pred_1)
vroom_write(x=kag_sub, file="./LogregPreds.csv", delim=",")






