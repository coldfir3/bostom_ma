rm(list = ls())
library(xgboost)
library(tidyverse)

load('dataset.data')
load('scale.data')
x_train <- scale(x_train, center = mean, scale = std)
x_test <- scale(x_test, center = mean, scale = std)

model_xgb1 <- xgboost(data.matrix(x_train), label = y_train, nrounds = 100, verbose = FALSE)
model_xgb1 %>% xgb.save('model_xgb1.xgb')

y_pred <- model_xgb1 %>% predict(data.matrix(x_test))
mae(y_pred, y_test)


