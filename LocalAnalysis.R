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
  step_other(all_factor_predictors(), threshold = .0001) %>%
  step_upsample() %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_range(all_numeric_predictors(), min=0, max=1)

#prep <- prep(my_recipe)
#baked <- bake(prep, new_data = train_data)


forest_mod <- rand_forest(mtry = 1, min_n=10, trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

forest_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(forest_mod)%>%
  fit(data=train_data)

forest_preds <- predict(forest_wf, new_data=test_data, type="prob")

kag_sub <- data.frame(id = test_data$id,
                      ACTION = forest_preds$.pred_1)
vroom_write(x=kag_sub, file="./Forest_local.csv", delim=",")







