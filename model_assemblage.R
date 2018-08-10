rm(list = ls())
library(keras)
library(xgboost)
library(glmnet)
library(tidyverse)
library(Metrics)

load('dataset.data')
load('scale.data')
x_train <- scale(x_train, center = mean, scale = std)
x_test <- scale(x_test, center = mean, scale = std)

model_ann1 <- load_model_hdf5('model_ann1.h5')
model_xgb1 <- xgb.load('model_xgb1.xgb')
load('model_las1.las')

y_ann1 <- model_ann1 %>% predict(x_test) %>% as.vector
y_xgb1 <- model_xgb1 %>% predict(x_test)
y_las1 <- model_las1 %>% predict(x_test) %>% as.vector

fun <- function(w){
  w <- w/sum(w)
  y_pred <- w[1]*y_ann1 + w[2]*y_xgb1 + w[3]*y_las1
  mae(y_test, y_pred)
}
w <- rgenoud::genoud(fun, 3, Domains = cbind(rep(0,3), rep(1,3)), print.level = 1, gradient.check = FALSE,  boundary.enforcement = 3)$par
w <- w/sum(w)

w_ann1 <- w[1]
w_xgb1 <- w[2]
w_las1 <- w[3]

predict_asbl <- function(x, model_ann, model_xgb, model_las, w_ann, w_xgb, w_las){
  y_ann <- model_ann %>% predict(x_test) %>% as.vector()
  y_xgb <- model_xgb %>% predict(x_test)
  y_las <- model_las %>% predict(x_test) %>% as.vector()
  y_pred <- y_ann * w_ann + y_xgb * w_xgb + y_las * w_las
  y_pred
}

c(
  assem = mae(y_test, predict_asbl(x_test, model_ann1, model_xgb1, model_las1, w_ann1, w_xgb1, w_las1)),
  ann = mae(y_test, predict_asbl(x_test, model_ann1, model_xgb1, model_las1, 1, 0, 0)),
  xgb = mae(y_test, predict_asbl(x_test, model_ann1, model_xgb1, model_las1, 0, 1, 0)),
  lass = mae(y_test, predict_asbl(x_test, model_ann1, model_xgb1, model_las1, 0, 0, 1))
)



