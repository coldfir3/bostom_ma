rm(list = ls())
library(keras)
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

ann1_weights <- get_weights(model_ann1)

k <- 4
indices <- sample(1:nrow(x_train))
folds <- cut(indices, breaks = k, labels = FALSE)
num_epochs <- 100
scores <- NULL
for (i in 1:k) {
  cat("processing fold #", i, "\n")
  val_indices <- which(folds == i, arr.ind = TRUE)
  val_data <- x_train[val_indices,]
  val_targets <- y_train[val_indices]
  partial_x_train <- x_train[-val_indices,]
  partial_y_train <- y_train[-val_indices]
  model_ann1 %>% set_weights(ann1_weights)
  model_ann1 %>% fit(partial_x_train, partial_y_train,
                epochs = num_epochs, batch_size = 10, verbose = 0)
  results <- model_ann1 %>% evaluate(val_data, val_targets, verbose = 0)
  scores <- c(scores, results$mean_absolute_error)
}
mean(scores)/diff(range(y_train))*100

callbacks_list <- list(
  callback_early_stopping(
    monitor = "mae",
    patience = 1
  ),
  callback_model_checkpoint(
    filepath = "model_ann1.h5",
    monitor = "val_loss",
    save_best_only = TRUE
  )
)

model_ann1 %>% set_weights(ann1_weights)
model_ann1 %>% fit(x_train, y_train,
                   epochs = num_epochs, batch_size = 10,
                   validation_data = list(x_test, y_test),
                   callbacks = callbacks_list)
