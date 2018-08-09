rm(list = ls())
library(keras)
library(xgboost)
library(tidyverse)

dataset <- dataset_boston_housing()
c(c(x_train, y_train), c(x_test, y_test)) %<-% dataset
rm(dataset)

mean <- apply(x_train, 2, mean)
std <- apply(x_train, 2, sd)

x_train <- scale(x_train, center = mean, scale = std)
x_test <- scale(x_test, center = mean, scale = std)

model_ann1 <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = dim(x_train)[2]) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1) %>% 
  compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = c("mae")
  )



