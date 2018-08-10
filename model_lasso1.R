rm(list = ls())
library(glmnet)
library(tidyverse)
library(Metrics)

load('dataset.data')
load('scale.data')
x_train <- scale(x_train, center = mean, scale = std)
x_test <- scale(x_test, center = mean, scale = std)

model_las1 = cv.glmnet(x_train, as.matrix(y_train), nfolds = 3)
save(model_las1, file = 'model_las1.las')

y_pred <- model_las1 %>% predict(x_test) %>% as.vector()
mae(y_pred, y_test)
