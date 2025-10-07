library(vroom)
library(DataExplorer)
library(dplyr)
library(tidymodels)




# Upload Data
train_data <- vroom("train.csv")
test_data <- vroom("test.csv")

# EDA
DataExplorer::plot_intro(train_data)         # Need to change continuous variable to discrete
DataExplorer::plot_correlation(train_data)
DataExplorer::plot_missing(train_data)       # None!!!! :)


# Recipe
my_recipe <- recipe(ACTION~., data=train_data) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_dummy(all_nominal_predictors())

# apply the recipe to your data1
prep <- prep(my_recipe)
baked <- bake(prep, new_data = train_data)









