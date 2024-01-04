# Learning Algorithms
  
# create indices for splitting (80% train, 20% test)
train_indices <- createDataPartition(lc_data$int_rate, p = 0.8, list = FALSE)

# create training and testing datasets
train_data <- lc_data[train_indices, ]
test_data <- lc_data[-train_indices, ]

#### Linear Regression ####

lm.fit <- lm(int_rate ~ ., data = train_data)

# make predictions on the training and testing data
lm.train_predictions <- predict(lm.fit, newdata = train_data)
lm.test_predictions <- predict(lm.fit, newdata = test_data)

# calculate Mean Squared Error (MSE) for training and testing
lm.train_mse <- mean((lm.train_predictions - train_data$int_rate)^2)
lm.test_mse <- mean((lm.test_predictions - test_data$int_rate)^2)

# calculate Root Mean Squared Error (RMSE) for training and testing
lm.train_rmse <- sqrt(lm.train_mse)
lm.test_rmse <- sqrt(lm.test_mse)

# calculate Mean Absolute Error (MAE) for training and testing
lm.train_mae <- mean(abs(lm.train_predictions - train_data$int_rate))
lm.test_mae <- mean(abs(lm.test_predictions - test_data$int_rate))

# calculate R-squared (R²) for training and testing
lm.train_r2 <- 1 - (sum((train_data$int_rate - lm.train_predictions)^2) / sum((train_data$int_rate - mean(train_data$int_rate))^2))
lm.test_r2 <- 1 - (sum((test_data$int_rate - lm.test_predictions)^2) / sum((test_data$int_rate - mean(test_data$int_rate))^2))

# display the metrics
cat("Training MSE:", lm.train_mse, "\n")
cat("Testing MSE:", lm.test_mse, "\n")
cat("Training RMSE:", lm.train_rmse, "\n")
cat("Testing RMSE:", lm.test_rmse, "\n")
cat("Training MAE:", lm.train_mae, "\n")
cat("Testing MAE:", lm.test_mae, "\n")
cat("Training R-squared (R²):", lm.train_r2, "\n")
cat("Testing R-squared (R²):", lm.test_r2, "\n")

# Lasso

lasso.predictors_train <- model.matrix(int_rate ~ ., train_data)[,-1]
lasso.target_train <- train_data$int_rate
lasso.predictors_test <- model.matrix(int_rate ~ ., test_data)[,-1]
lasso.target_test <- test_data$int_rate

lasso.fit <- glmnet(lasso.predictors_train, lasso.target_train, alpha = 1)

plot(lasso.fit, label=TRUE)

# make predictions on the training and testing data
lasso.train_predictions <- predict(lasso.fit, newdata = train_data, newx = lasso.predictors_train)
lasso.test_predictions <- predict(lasso.fit, newdata = test_data, newx = lasso.predictors_train)

# calculate Mean Squared Error (MSE) for training and testing
lasso.train_mse <- mean((lasso.train_predictions - train_data$int_rate)^2)
lasso.test_mse <- mean((lasso.test_predictions - test_data$int_rate)^2)

# calculate Root Mean Squared Error (RMSE) for training and testing
lasso.train_rmse <- sqrt(lasso.train_mse)
lasso.test_rmse <- sqrt(lasso.test_mse)

# calculate Mean Absolute Error (MAE) for training and testing
lasso.train_mae <- mean(abs(lasso.train_predictions - train_data$int_rate))
lasso.test_mae <- mean(abs(lasso.test_predictions - test_data$int_rate))

# calculate R-squared (R²) for training and testing
lasso.train_r2 <- 1 - (sum((train_data$int_rate - lasso.train_predictions)^2) / sum((train_data$int_rate - mean(train_data$int_rate))^2))
lasso.test_r2 <- 1 - (sum((test_data$int_rate - lasso.test_predictions)^2) / sum((test_data$int_rate - mean(test_data$int_rate))^2))

# display the metrics
cat("Training MSE:", lasso.train_mse, "\n")
cat("Testing MSE:", lasso.test_mse, "\n")
cat("Training RMSE:", lasso.train_rmse, "\n")
cat("Testing RMSE:", lasso.test_rmse, "\n")
cat("Training MAE:", lasso.train_mae, "\n")
cat("Testing MAE:", lasso.test_mae, "\n")
cat("Training R-squared (R²):", lasso.train_r2, "\n")
cat("Testing R-squared (R²):", lasso.test_r2, "\n")


# Ridge

ridge.predictors_train <- model.matrix(int_rate ~ ., train_data)[,-1]
ridge.target_train <- train_data$int_rate
ridge.predictors_test <- model.matrix(int_rate ~ ., test_data)[,-1]
ridge.target_test <- test_data$int_rate

ridge.fit <- glmnet(ridge.predictors_train, ridge.target_train, alpha = 0)

plot(ridge.fit, label=TRUE, xlab = "L2 Norm")

# make predictions on the training and testing data
ridge.train_predictions <- predict(ridge.fit, newdata = train_data, newx = ridge.predictors_train)
ridge.test_predictions <- predict(ridge.fit, newdata = test_data, newx = ridge.predictors_train)

# calculate Mean Squared Error (MSE) for training and testing
ridge.train_mse <- mean((ridge.train_predictions - train_data$int_rate)^2)
ridge.test_mse <- mean((ridge.test_predictions - test_data$int_rate)^2)

# calculate Root Mean Squared Error (RMSE) for training and testing
ridge.train_rmse <- sqrt(ridge.train_mse)
ridge.test_rmse <- sqrt(ridge.test_mse)

# calculate Mean Absolute Error (MAE) for training and testing
ridge.train_mae <- mean(abs(ridge.train_predictions - train_data$int_rate))
ridge.test_mae <- mean(abs(ridge.test_predictions - test_data$int_rate))

# calculate R-squared (R²) for training and testing
ridge.train_r2 <- 1 - (sum((train_data$int_rate - ridge.train_predictions)^2) / sum((train_data$int_rate - mean(train_data$int_rate))^2))
ridge.test_r2 <- 1 - (sum((test_data$int_rate - ridge.test_predictions)^2) / sum((test_data$int_rate - mean(test_data$int_rate))^2))

# display the metrics
cat("Training MSE:", ridge.train_mse, "\n")
cat("Testing MSE:", ridge.test_mse, "\n")
cat("Training RMSE:", ridge.train_rmse, "\n")
cat("Testing RMSE:", ridge.test_rmse, "\n")
cat("Training MAE:", ridge.train_mae, "\n")
cat("Testing MAE:", ridge.test_mae, "\n")
cat("Training R-squared (R²):", ridge.train_r2, "\n")
cat("Testing R-squared (R²):", ridge.test_r2, "\n")


# K fold using K=5:
  
# define the number of folds for cross-validation
num_folds <- 5
folds <- createFolds(train_data$int_rate, k = num_folds, list = TRUE)


# K fold using K=5 and linear regression:
  
#### Linear Regresion applying Cross Validation with k=5  ####

# initialize lists to store models and their results
lm.k5.models <- list()
lm.k5.results <- data.frame()

# perform k-fold cross-validation
for(i in seq_along(folds)) {
  # split the data into training and testing for the current fold
  train_indices <- folds[[i]]
  test_indices <- setdiff(seq_len(nrow(train_data)), train_indices)
  
  train_data_fold <- train_data[train_indices, ]
  test_data_fold <- train_data[test_indices, ]
  
  # fit the model on the training fold
  lm.k5 <- lm(int_rate ~ ., data = train_data_fold)
  lm.k5.models[[i]] <- lm.k5  # Store the model
  
  # make predictions on the training and testing fold
  lm.k5.train_predictions <- predict(lm.k5, newdata = train_data_fold)
  lm.k5.test_predictions <- predict(lm.k5, newdata = test_data_fold)
  
  # calculate metrics for training fold
  lm.k5.train_mse <- mean((lm.k5.train_predictions - train_data_fold$int_rate)^2)
  lm.k5.train_rmse <- sqrt(lm.k5.train_mse)
  lm.k5.train_mae <- mean(abs(lm.k5.train_predictions - train_data_fold$int_rate))
  lm.k5.train_r2 <- summary(lm.k5)$r.squared
  
  # calculate metrics for testing fold
  lm.k5.test_mse <- mean((lm.k5.test_predictions - test_data_fold$int_rate)^2)
  lm.k5.test_rmse <- sqrt(lm.k5.test_mse)
  lm.k5.test_mae <- mean(abs(lm.k5.test_predictions - test_data_fold$int_rate))
  lm.k5.test_r2 <- 1 - (sum((test_data_fold$int_rate - lm.k5.test_predictions)^2) / sum((test_data_fold$int_rate - mean(test_data_fold$int_rate))^2))
  
  # store metrics in the results dataframe
  lm.k5.results <- rbind(lm.k5.results, data.frame(
    Fold = i,
    Train_MSE = lm.k5.train_mse, Test_MSE = lm.k5.test_mse,
    Train_RMSE = lm.k5.train_rmse, Test_RMSE = lm.k5.test_rmse,
    Train_MAE = lm.k5.train_mae, Test_MAE = lm.k5.test_mae,
    Train_R2 = lm.k5.train_r2, Test_R2 = lm.k5.test_r2
  ))
}

# display the models and their metrics
print(lm.k5.models)
print(lm.k5.results)


plot_metric <- function(results_long, metric) {
  # adjust the variable names based on the metric
  variables <- if (metric == "OOB") {
    "OOB_Error"
  } else {
    c(paste0('Train_', metric), paste0('Test_', metric))
  }
  title <- if (metric == "OOB") {
    paste0(metric, ' per Fold')
  } else {
    paste0('Train vs Test ', metric, ' per Fold')
  }
  
  ggplot(results_long[results_long$variable %in% variables, ],
         aes(x = Fold, y = value, color = variable)) +
    geom_line() +
    geom_point() +
    theme_minimal() +
    labs(title = title,
         x = 'Fold',
         y = metric)
}


# reshape data for plotting
lm.k5.results_long <- melt(lm.k5.results, id.vars = 'Fold')

# plot for each metric
p1 <- plot_metric(lm.k5.results_long, 'MSE')
p2 <- plot_metric(lm.k5.results_long, 'RMSE')
p3 <- plot_metric(lm.k5.results_long, 'MAE')
p4 <- plot_metric(lm.k5.results_long, 'R2')

# arrange the plots in a 2x2 grid
grid.arrange(p1, p2, p3, p4, ncol = 2, nrow = 2)

plot(p1)
plot(p2)
plot(p3)
plot(p4)

# K fold using K=5 and Random Forest:

#### Random Forest applying Cross Validation with k=5  ####

# initialize lists to store models and their results
rf.k5.models <- list()
rf.k5.results <- data.frame()

# perform k-fold cross-validation
for(i in seq_along(folds)) {
  # split the data into training and testing for the current fold
  train_indices <- folds[[i]]
  test_indices <- setdiff(seq_len(nrow(train_data)), train_indices)
  
  train_data_fold <- train_data[train_indices, ]
  test_data_fold <- train_data[test_indices, ]
  
  # fit the model on the training fold
  rf.k5 <- ranger(formula = int_rate ~ ., data = train_data_fold, num.trees = 500, verbose=TRUE, importance = "impurity", oob.error = TRUE)
  rf.k5.models[[i]] <- rf.k5  # Store the model
  
  # make predictions on the training and testing fold
  rf.k5.train_predictions <- predict(rf.k5, data = train_data_fold)$predictions
  rf.k5.test_predictions <- predict(rf.k5, data = test_data_fold)$predictions
  
  # calculate metrics for training fold
  rf.k5.train_mse <- mean((rf.k5.train_predictions - train_data_fold$int_rate)^2)
  rf.k5.train_rmse <- sqrt(rf.k5.train_mse)
  rf.k5.train_mae <- mean(abs(rf.k5.train_predictions - train_data_fold$int_rate))
  rf.k5.oob_error <- rf.k5$prediction.error
  
  # calculate metrics for testing fold
  rf.k5.test_mse <- mean((rf.k5.test_predictions - test_data_fold$int_rate)^2)
  rf.k5.test_rmse <- sqrt(rf.k5.test_mse)
  rf.k5.test_mae <- mean(abs(rf.k5.test_predictions - test_data_fold$int_rate))
  rf.k5.test_r2 <- 1 - (sum((test_data_fold$int_rate - rf.k5.test_predictions)^2) / sum((test_data_fold$int_rate - mean(test_data_fold$int_rate))^2))
  
  # store metrics in the results dataframe
  rf.k5.results <- rbind(rf.k5.results, data.frame(
    Fold = i,
    Train_MSE = rf.k5.train_mse, Test_MSE = rf.k5.test_mse,
    Train_RMSE = rf.k5.train_rmse, Test_RMSE = rf.k5.test_rmse,
    Train_MAE = rf.k5.train_mae, Test_MAE = rf.k5.test_mae,
    OOB_Error = rf.k5.oob_error
  ))
}

# display the models and their metrics
print(rf.k5.models)
print(rf.k5.results)

# reshape data for plotting
rf.k5.results_long <- melt(rf.k5.results, id.vars = 'Fold')

# plot for each metric
p1 <- plot_metric(rf.k5.results_long, 'MSE')
p2 <- plot_metric(rf.k5.results_long, 'RMSE')
p3 <- plot_metric(rf.k5.results_long, 'MAE')
p4 <- plot_metric(rf.k5.results_long, 'OOB')

# arrange the plots in a 2x2 grid
grid.arrange(p1, p2, p3, p4, ncol = 2, nrow = 2)

plot(p1)
plot(p2)
plot(p3)
plot(p4)

# K fold using K=5 and Boosting:

#### Boosting applying Cross Validation with k=5  ####

# initialize lists to store models and their results
xgb.k5.models <- list()
xgb.k5.results <- data.frame()

# perform k-fold cross-validation
for(i in seq_along(folds)) {
  # split the data into training and testing for the current fold
  train_indices <- folds[[i]]
  test_indices <- setdiff(seq_len(nrow(train_data)), train_indices)
  
  train_data_fold <- train_data[train_indices, ]
  test_data_fold <- train_data[test_indices, ]
  
  # prepare data for xgboost
  xgb.y_train_fold <- train_data_fold$int_rate
  xgb.X_train_fold <- as.matrix(train_data_fold[, -which(names(train_data_fold) == 'int_rate')])
  
  xgb.y_test_fold <- test_data_fold$int_rate
  xgb.X_test_fold <- as.matrix(test_data_fold[, -which(names(test_data_fold) == 'int_rate')])
  
  # fit the xgboost model on the training fold
  xgb.k5 <- xgboost(
    data = xgb.X_train_fold,
    label = xgb.y_train_fold,
    nrounds = 100,
    verbose = 0
  )
  xgb.k5.models[[i]] <- xgb.k5  # store the model
  
  # make predictions on the training fold
  xgb.k5.train_predictions <- predict(xgb.k5, newdata = xgb.X_train_fold)
  # make predictions on the testing fold
  xgb.k5.test_predictions <- predict(xgb.k5, newdata = xgb.X_test_fold)
  
  # calculate metrics for training fold
  xgb.k5.train_mse <- mean((xgb.k5.train_predictions - train_data_fold$int_rate)^2)
  xgb.k5.train_rmse <- sqrt(xgb.k5.train_mse)
  xgb.k5.train_mae <- mean(abs(xgb.k5.train_predictions - train_data_fold$int_rate))
  xgb.k5.train_r2 <- 1 - (sum((xgb.y_train_fold - xgb.k5.train_predictions)^2) / sum((xgb.y_train_fold - mean(xgb.y_train_fold))^2))
  
  # calculate metrics for testing fold
  xgb.k5.test_mse <- mean((xgb.k5.test_predictions - xgb.y_test_fold)^2)
  xgb.k5.test_rmse <- sqrt(xgb.k5.test_mse)
  xgb.k5.test_mae <- mean(abs(xgb.k5.test_predictions - xgb.y_test_fold))
  xgb.k5.test_r2 <- 1 - (sum((xgb.y_test_fold - xgb.k5.test_predictions)^2) / sum((xgb.y_test_fold - mean(xgb.y_test_fold))^2))  
  
  # store metrics in the results dataframe
  xgb.k5.results <- rbind(xgb.k5.results, data.frame(
    Fold = i,
    Train_MSE = xgb.k5.train_mse, Test_MSE = xgb.k5.test_mse,
    Train_RMSE = xgb.k5.train_rmse, Test_RMSE = xgb.k5.test_rmse,
    Train_MAE = xgb.k5.train_mae, Test_MAE = xgb.k5.test_mae,
    Train_R2 = xgb.k5.train_r2, Test_R2 = xgb.k5.test_r2
  ))
}

# display the models and their metrics
print(xgb.k5.models)
print(xgb.k5.results)

# reshape data for plotting
xgb.k5.results_long <- melt(xgb.k5.results, id.vars = 'Fold')

# plot for each metric
p1 <- plot_metric(xgb.k5.results_long, 'MSE')
p2 <- plot_metric(xgb.k5.results_long, 'RMSE')
p3 <- plot_metric(xgb.k5.results_long, 'MAE')
p4 <- plot_metric(xgb.k5.results_long, 'R2')

# arrange the plots in a 2x2 grid
grid.arrange(p1, p2, p3, p4, ncol = 2, nrow = 2)

plot(p1)
plot(p2)
plot(p3)
plot(p4)


# K fold using K=10:

# define the number of folds for cross-validation
num_folds <- 10
folds <- createFolds(train_data$int_rate, k = num_folds, list = TRUE)



# K fold using K=10 and linear regression:
  
#### Linear Regresion applying Cross Validation with k=10  ####

# initialize lists to store models and their results
lm.k10.models <- list()
lm.k10.results <- data.frame()

# perform k-fold cross-validation
for(i in seq_along(folds)) {
  # split the data into training and testing for the current fold
  train_indices <- folds[[i]]
  test_indices <- setdiff(seq_len(nrow(train_data)), train_indices)
  
  train_data_fold <- train_data[train_indices, ]
  test_data_fold <- train_data[test_indices, ]
  
  # fit the model on the training fold
  lm.k10 <- lm(int_rate ~ ., data = train_data_fold)
  lm.k10.models[[i]] <- lm.k10  # Store the model
  
  # make predictions on the training and testing fold
  lm.k10.train_predictions <- predict(lm.k10, newdata = train_data_fold)
  lm.k10.test_predictions <- predict(lm.k10, newdata = test_data_fold)
  
  # calculate metrics for training fold
  lm.k10.train_mse <- mean((lm.k10.train_predictions - train_data_fold$int_rate)^2)
  lm.k10.train_rmse <- sqrt(lm.k10.train_mse)
  lm.k10.train_mae <- mean(abs(lm.k10.train_predictions - train_data_fold$int_rate))
  lm.k10.train_r2 <- summary(lm.k10)$r.squared
  
  # calculate metrics for testing fold
  lm.k10.test_mse <- mean((lm.k10.test_predictions - test_data_fold$int_rate)^2)
  lm.k10.test_rmse <- sqrt(lm.k10.test_mse)
  lm.k10.test_mae <- mean(abs(lm.k10.test_predictions - test_data_fold$int_rate))
  lm.k10.test_r2 <- 1 - (sum((test_data_fold$int_rate - lm.k10.test_predictions)^2) / sum((test_data_fold$int_rate - mean(test_data_fold$int_rate))^2))
  
  # store metrics in the results dataframe
  lm.k10.results <- rbind(lm.k10.results, data.frame(
    Fold = i,
    Train_MSE = lm.k10.train_mse, Test_MSE = lm.k10.test_mse,
    Train_RMSE = lm.k10.train_rmse, Test_RMSE = lm.k10.test_rmse,
    Train_MAE = lm.k10.train_mae, Test_MAE = lm.k10.test_mae,
    Train_R2 = lm.k10.train_r2, Test_R2 = lm.k10.test_r2
  ))
}

# display the models and their metrics
print(lm.k10.models)
print(lm.k10.results)


plot_metric <- function(results_long, metric) {
  # adjust the variable names based on the metric
  variables <- if (metric == "OOB") {
    "OOB_Error"
  } else {
    c(paste0('Train_', metric), paste0('Test_', metric))
  }
  title <- if (metric == "OOB") {
    paste0(metric, ' per Fold')
  } else {
    paste0('Train vs Test ', metric, ' per Fold')
  }
  
  ggplot(results_long[results_long$variable %in% variables, ],
         aes(x = Fold, y = value, color = variable)) +
    geom_line() +
    geom_point() +
    theme_minimal() +
    labs(title = title,
         x = 'Fold',
         y = metric)
}

# reshape data for plotting
lm.k10.results_long <- melt(lm.k10.results, id.vars = 'Fold')

# plot for each metric
p1 <- plot_metric(lm.k10.results_long, 'MSE')
p2 <- plot_metric(lm.k10.results_long, 'RMSE')
p3 <- plot_metric(lm.k10.results_long, 'MAE')
p4 <- plot_metric(lm.k10.results_long, 'R2')

# arrange the plots in a 2x2 grid
grid.arrange(p1, p2, p3, p4, ncol = 2, nrow = 2)

plot(p1)
plot(p2)
plot(p3)
plot(p4)

# K fold using K=10 and Random Forest:
  
# #### Random Forest applying Cross Validation with k=10  ####
# 
# # initialize lists to store models and their results
# rf.k10.models <- list()
# rf.k10.results <- data.frame()
# 
# # perform k-fold cross-validation
# for(i in seq_along(folds)) {
#   # split the data into training and testing for the current fold
#   train_indices <- folds[[i]]
#   test_indices <- setdiff(seq_len(nrow(train_data)), train_indices)
#   
#   train_data_fold <- train_data[train_indices, ]
#   test_data_fold <- train_data[test_indices, ]
#   
#   # fit the model on the training fold
#   rf.k10 <- ranger(formula = int_rate ~ ., data = train_data, num.trees = 500, verbose=TRUE, importance = "impurity", oob.error = TRUE)
#   rf.k10.models[[i]] <- rf.k10  # Store the model
#   
#   # make predictions on the training and testing fold
#   rf.k10.train_predictions <- predict(rf.k10, data = train_data_fold)$predictions
#   rf.k10.test_predictions <- predict(rf.k10, data = test_data_fold)$predictions
#   
#   # calculate metrics for training fold
#   rf.k10.train_mse <- mean((rf.k10.train_predictions - train_data_fold$int_rate)^2)
#   rf.k10.train_rmse <- sqrt(rf.k10.train_mse)
#   rf.k10.train_mae <- mean(abs(rf.k10.train_predictions - train_data_fold$int_rate))
#   rf.k10.oob_error <- rf.k10$prediction.error
#   
#   # calculate metrics for testing fold
#   rf.k10.test_mse <- mean((rf.k10.test_predictions - test_data_fold$int_rate)^2)
#   rf.k10.test_rmse <- sqrt(rf.k10.test_mse)
#   rf.k10.test_mae <- mean(abs(rf.k10.test_predictions - test_data_fold$int_rate))
#   rf.k10.test_r2 <- 1 - (sum((test_data_fold$int_rate - rf.k10.test_predictions)^2) / sum((test_data_fold$int_rate - mean(test_data_fold$int_rate))^2))
#   
#   # store metrics in the results dataframe
#   rf.k10.results <- rbind(rf.k10.results, data.frame(
#     Fold = i,
#     Train_MSE = rf.k10.train_mse, Test_MSE = rf.k10.test_mse,
#     Train_RMSE = rf.k10.train_rmse, Test_RMSE = rf.k10.test_rmse,
#     Train_MAE = rf.k10.train_mae, Test_MAE = rf.k10.test_mae,
#     OOB_Error = rf.k10.oob_error
#   ))
# }
# 
# # display the models and their metrics
# print(rf.k10.models)
# print(rf.k10.results)

# reshape data for plotting
# rf.k10.results_long <- melt(rf.k10.results, id.vars = 'Fold')
# 
# # plot for each metric
# p1 <- plot_metric(rf.k10.results_long, 'MSE')
# p2 <- plot_metric(rf.k10.results_long, 'RMSE')
# p3 <- plot_metric(rf.k10.results_long, 'MAE')
# p4 <- plot_metric(rf.k10.results_long, 'OOB')
# 
# # arrange the plots in a 2x2 grid
# grid.arrange(p1, p2, p3, p4, ncol = 2, nrow = 2)
# 
# plot(p1)
# plot(p2)
# plot(p3)
# plot(p4)


# K fold using K=10 and Boosting:

#### Boosting applying Cross Validation with k=10  ####

# initialize lists to store models and their results
xgb.k10.models <- list()
xgb.k10.results <- data.frame()

# perform k-fold cross-validation
for(i in seq_along(folds)) {
  # split the data into training and testing for the current fold
  train_indices <- folds[[i]]
  test_indices <- setdiff(seq_len(nrow(train_data)), train_indices)
  
  train_data_fold <- train_data[train_indices, ]
  test_data_fold <- train_data[test_indices, ]
  
  # prepare data for xgboost
  xgb.y_train_fold <- train_data_fold$int_rate
  xgb.X_train_fold <- as.matrix(train_data_fold[, -which(names(train_data_fold) == 'int_rate')])
  
  xgb.y_test_fold <- test_data_fold$int_rate
  xgb.X_test_fold <- as.matrix(test_data_fold[, -which(names(test_data_fold) == 'int_rate')])
  
  # fit the xgboost model on the training fold
  xgb.k10 <- xgboost(
    data = xgb.X_train_fold,
    label = xgb.y_train_fold,
    nrounds = 100,
    verbose = 0
  )
  xgb.k10.models[[i]] <- xgb.k10  # store the model
  
  # make predictions on the training fold
  xgb.k10.train_predictions <- predict(xgb.k10, newdata = xgb.X_train_fold)
  # make predictions on the testing fold
  xgb.k10.test_predictions <- predict(xgb.k10, newdata = xgb.X_test_fold)
  
  # calculate metrics for training fold
  xgb.k10.train_mse <- mean((xgb.k10.train_predictions - train_data_fold$int_rate)^2)
  xgb.k10.train_rmse <- sqrt(xgb.k10.train_mse)
  xgb.k10.train_mae <- mean(abs(xgb.k10.train_predictions - train_data_fold$int_rate))
  xgb.k10.train_r2 <- 1 - (sum((xgb.y_train_fold - xgb.k10.train_predictions)^2) / sum((xgb.y_train_fold - mean(xgb.y_train_fold))^2))
  
  # calculate metrics for testing fold
  xgb.k10.test_mse <- mean((xgb.k10.test_predictions - xgb.y_test_fold)^2)
  xgb.k10.test_rmse <- sqrt(xgb.k10.test_mse)
  xgb.k10.test_mae <- mean(abs(xgb.k10.test_predictions - xgb.y_test_fold))
  xgb.k10.test_r2 <- 1 - (sum((xgb.y_test_fold - xgb.k10.test_predictions)^2) / sum((xgb.y_test_fold - mean(xgb.y_test_fold))^2))  
  
  # store metrics in the results dataframe
  xgb.k10.results <- rbind(xgb.k10.results, data.frame(
    Fold = i,
    Train_MSE = xgb.k10.train_mse, Test_MSE = xgb.k10.test_mse,
    Train_RMSE = xgb.k10.train_rmse, Test_RMSE = xgb.k10.test_rmse,
    Train_MAE = xgb.k10.train_mae, Test_MAE = xgb.k10.test_mae,
    Train_R2 = xgb.k10.train_r2, Test_R2 = xgb.k10.test_r2
  ))
}

# display the models and their metrics
print(xgb.k10.models)
print(xgb.k10.results)

# reshape data for plotting
xgb.k10.results_long <- melt(xgb.k10.results, id.vars = 'Fold')

# plot for each metric
p1 <- plot_metric(xgb.k10.results_long, 'MSE')
p2 <- plot_metric(xgb.k10.results_long, 'RMSE')
p3 <- plot_metric(xgb.k10.results_long, 'MAE')
p4 <- plot_metric(xgb.k10.results_long, 'R2')

# arrange the plots in a 2x2 grid
grid.arrange(p1, p2, p3, p4, ncol = 2, nrow = 2)

plot(p1)
plot(p2)
plot(p3)
plot(p4)

# Decision Trees

#### Decision Trees ####

# error in tree: "factor predictors must have at most 32 levels" is thrown

# basically, it becomes computationally expensive to create so many splits in your data, since you are selecting the best split out of all 2^32 (approx) possible splits

# The error above was solved with the factor and then numeric variable transformation

# fit a decision tree model on the training data
tm <- tree(int_rate ~ ., data = train_data)

# make predictions on the training and testing data
tm.train_predictions <- predict(tm, newdata = train_data)
tm.test_predictions <- predict(tm, newdata = test_data)

# calculate Mean Squared Error (MSE) for training and testing
tm.train_mse <- mean((tm.train_predictions - train_data$int_rate)^2)
tm.test_mse <- mean((tm.test_predictions - test_data$int_rate)^2)

# calculate Root Mean Squared Error (RMSE) for training and testing
tm.train_rmse <- sqrt(tm.train_mse)
tm.test_rmse <- sqrt(tm.test_mse)

# calculate Mean Absolute Error (MAE) for training and testing
tm.train_mae <- mean(abs(tm.train_predictions - train_data$int_rate))
tm.test_mae <- mean(abs(tm.test_predictions - test_data$int_rate))

# calculate R-squared (R²) for training and testing
tm.train_r2 <- 1 - (sum((train_data$int_rate - tm.train_predictions)^2) / sum((train_data$int_rate - mean(train_data$int_rate))^2))
tm.test_r2 <- 1 - (sum((test_data$int_rate - tm.test_predictions)^2) / sum((test_data$int_rate - mean(test_data$int_rate))^2))

# display the metrics
cat("Training MSE:", tm.train_mse, "\n")
cat("Testing MSE:", tm.test_mse, "\n")
cat("Training RMSE:", tm.train_rmse, "\n")
cat("Testing RMSE:", tm.test_rmse, "\n")
cat("Training MAE:", tm.train_mae, "\n")
cat("Testing MAE:", tm.test_mae, "\n")
cat("Training R-squared (R²):", tm.train_r2, "\n")
cat("Testing R-squared (R²):", tm.test_r2, "\n")


#### Random Forest ####

# train a Random Forest model
rf <- ranger(formula = int_rate ~ ., data = train_data, num.trees = 500, verbose=TRUE, importance = "impurity", oob.error = TRUE)

# print the model summary
print("Random Forest Model Summary:")
print(rf)

# make predictions on the training and testing data
rf.train_predictions <- predict(rf, data = train_data)
rf.test_predictions <- predict(rf, data = test_data)

# calculate Mean Squared Error (MSE) for training and testing
rf.train_mse <- mean((rf.train_predictions$predictions - train_data$int_rate)^2)
rf.test_mse <- mean((rf.test_predictions$predictions - test_data$int_rate)^2)

# calculate Root Mean Squared Error (RMSE) for training and testing
rf.train_rmse <- sqrt(rf.train_mse)
rf.test_rmse <- sqrt(rf.test_mse)

# calculate Mean Absolute Error (MAE) for training and testing
rf.train_mae <- mean(abs(rf.train_predictions$predictions - train_data$int_rate))
rf.test_mae <- mean(abs(rf.test_predictions$predictions - test_data$int_rate))

# calculate R-squared (R²) for training and testing
rf.train_r2 <- 1 - (sum((train_data$int_rate - rf.train_predictions$predictions)^2) / sum((train_data$int_rate - mean(train_data$int_rate))^2))
rf.test_r2 <- 1 - (sum((test_data$int_rate - rf.test_predictions$predictions)^2) / sum((test_data$int_rate - mean(test_data$int_rate))^2))

# display the metrics
cat("Training MSE:", rf.train_mse, "\n")
cat("Testing MSE:", rf.test_mse, "\n")
cat("Training RMSE:", rf.train_rmse, "\n")
cat("Testing RMSE:", rf.test_rmse, "\n")
cat("Training MAE:", rf.train_mae, "\n")
cat("Testing MAE:", rf.test_mae, "\n")
cat("Training R-squared (R²):", rf.train_r2, "\n")
cat("Testing R-squared (R²):", rf.test_r2, "\n")


#### Boosting ####

# define the target variable for training and testing
xgb.y_train <- train_data$int_rate
xgb.y_test <- test_data$int_rate

# define the feature matrix for training and testing (exclude the target variable)
xgb.X_train <- train_data[, -which(names(train_data) == 'int_rate')]
xgb.X_test <- test_data[, -which(names(test_data) == 'int_rate')]

# fit a gradient boosting regression model using xgboost
xgb <- xgboost(
  data = as.matrix(xgb.X_train),
  label = xgb.y_train,
  nrounds = 100,
  verbose = 0
)

# make predictions on the training and testing data
xgb.train_predictions <- predict(xgb, newdata = as.matrix(xgb.X_train))
xgb.test_predictions <- predict(xgb, newdata = as.matrix(xgb.X_test))

# calculate Mean Squared Error (MSE) for training and testing
xgb.train_mse <- mean((xgb.train_predictions - xgb.y_train)^2)
xgb.test_mse <- mean((xgb.test_predictions - xgb.y_test)^2)

# calculate Root Mean Squared Error (RMSE) for training and testing
xgb.train_rmse <- sqrt(xgb.train_mse)
xgb.test_rmse <- sqrt(xgb.test_mse)

# calculate Mean Absolute Error (MAE) for training and testing
xgb.train_mae <- mean(abs(xgb.train_predictions - xgb.y_train))
xgb.test_mae <- mean(abs(xgb.test_predictions - xgb.y_test))

# calculate R-squared (R²) for training and testing
xgb.train_r2 <- 1 - (sum((xgb.y_train - xgb.train_predictions)^2) / sum((xgb.y_train - mean(xgb.y_train))^2))
xgb.test_r2 <- 1 - (sum((xgb.y_test - xgb.test_predictions)^2) / sum((xgb.y_test - mean(xgb.y_test))^2))

# display the metrics
cat("Training MSE:", xgb.train_mse, "\n")
cat("Testing MSE:", xgb.test_mse, "\n")
cat("Training RMSE:", xgb.train_rmse, "\n")
cat("Testing RMSE:", xgb.test_rmse, "\n")
cat("Training MAE:", xgb.train_mae, "\n")
cat("Testing MAE:", xgb.test_mae, "\n")
cat("Training R-squared (R²):", xgb.train_r2, "\n")
cat("Testing R-squared (R²):", xgb.test_r2, "\n")

# Following, a scatter plot of actual vs predicted training values for each model is plot.
# This plot helps us visualize how well each model's predictions align with the actual data points.

# create a scatter plot function
create_scatter_plot <- function(actual_values, predicted_values, model_name) {
  model_comparison_data <- data.frame(
    Actual = actual_values,
    Predicted = predicted_values
  )
  
  scatter_plot <- ggplot(model_comparison_data, aes(x = Actual, y = Predicted)) +
    geom_point() +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +  # add a diagonal reference line
    labs(x = "Actual Training Values", y = "Predicted Training Values", title = model_name) +
    theme_minimal() +
    ylim(-50, 50)
  
  return(scatter_plot)
}

# create scatter plots for each model
lm_scatter_plot <- create_scatter_plot(
  actual_values = train_data$int_rate,
  predicted_values = lm.train_predictions,
  model_name = "Linear Regression"
)

rf_scatter_plot <- create_scatter_plot(
  actual_values = train_data$int_rate,
  predicted_values = rf.train_predictions$predictions,
  model_name = "Random Forest"
)

xgb_scatter_plot <- create_scatter_plot(
  actual_values = xgb.y_train,
  predicted_values = xgb.train_predictions,
  model_name = "XGBoost"
)

# display the scatter plots separately
print(lm_scatter_plot)
print(rf_scatter_plot)
print(xgb_scatter_plot)

# Following, a scatter plot of actual vs predicted testing values for each model is plot.
# This plot helps us visualize how well each model's predictions align with the actual data points.

# create a scatter plot function
create_scatter_plot <- function(actual_values, predicted_values, model_name) {
  model_comparison_data <- data.frame(
    Actual = actual_values,
    Predicted = predicted_values
  )
  
  scatter_plot <- ggplot(model_comparison_data, aes(x = Actual, y = Predicted)) +
    geom_point() +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +  # add a diagonal reference line
    labs(x = "Actual Testing Values", y = "Predicted Testing Values", title = model_name) +
    theme_minimal() +
    ylim(-50, 50) +
    xlim(0, 40)
  
  return(scatter_plot)
}

# create scatter plots for each model
lm_scatter_plot <- create_scatter_plot(
  actual_values = test_data$int_rate,
  predicted_values = lm.test_predictions,
  model_name = "Linear Regression"
)

rf_scatter_plot <- create_scatter_plot(
  actual_values = test_data$int_rate,
  predicted_values = rf.test_predictions$predictions,
  model_name = "Random Forest"
)

xgb_scatter_plot <- create_scatter_plot(
  actual_values = xgb.y_test,
  predicted_values = xgb.test_predictions,
  model_name = "XGBoost"
)

# display the scatter plots separately
print(lm_scatter_plot)
print(rf_scatter_plot)
print(xgb_scatter_plot)

# Residual plots can help identify patterns in prediction errors and assess whether the assumptions of linear regression (if applicable) are met.

# create a residual plot function
create_residual_plot <- function(actual_values, predicted_values, model_name) {
  residuals <- actual_values - predicted_values
  residual_data <- data.frame(
    Predicted = predicted_values,
    Residuals = residuals
  )
  
  residual_plot <- ggplot(residual_data, aes(x = Predicted, y = Residuals)) +
    geom_point() +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +  # Red horizontal reference line
    labs(x = "Predicted Values", y = "Residuals", title = paste("Residual Plot -", model_name)) +
    theme_minimal() +
    ylim(-30, 30) +
    xlim(0, 40)
  
  return(residual_plot)
}

# create residual plots for each model
lm_residual_plot <- create_residual_plot(
  actual_values = train_data$int_rate,
  predicted_values = lm.train_predictions,
  model_name = "Linear Regression"
)

rf_residual_plot <- create_residual_plot(
  actual_values = train_data$int_rate,
  predicted_values = rf.train_predictions$predictions,
  model_name = "Random Forest"
)

xgb_residual_plot <- create_residual_plot(
  actual_values = xgb.y_train,
  predicted_values = xgb.train_predictions,
  model_name = "XGBoost"
)

# display the residual plots separately
print(lm_residual_plot)
print(rf_residual_plot)
print(xgb_residual_plot)


# create a density plot function for residuals
create_residual_density_plot <- function(actual_values, predicted_values, model_name) {
  residuals <- actual_values - predicted_values
  residual_data <- data.frame(Residuals = residuals)
  
  density_plot <- ggplot(residual_data, aes(x = Residuals)) +
    geom_density(fill = "skyblue", color = "black", alpha = 0.7) +
    labs(x = "Residuals", y = "Density", title = paste("Residual Density Plot -", model_name)) +
    theme_minimal() +
    xlim(-30,30) + 
    ylim(0, 0.35)
  
  
  return(density_plot)
}

# create density plots for residuals for each model
lm_residual_density_plot <- create_residual_density_plot(
  actual_values = train_data$int_rate,
  predicted_values = lm.train_predictions,
  model_name = "Linear Regression"
)

rf_residual_density_plot <- create_residual_density_plot(
  actual_values = train_data$int_rate,
  predicted_values = rf.train_predictions$predictions,
  model_name = "Random Forest"
)

xgb_residual_density_plot <- create_residual_density_plot(
  actual_values = xgb.y_train,
  predicted_values = xgb.train_predictions,
  model_name = "XGBoost"
)

# display the density plots separately
print(lm_residual_density_plot)
print(rf_residual_density_plot)
print(xgb_residual_density_plot)


# This visualization can help you compare the distribution of prediction errors across models through histograms.

# create a histogram plot function for residuals with a red density curve
create_residual_histogram_plot <- function(actual_values, predicted_values, model_name) {
  residuals <- actual_values - predicted_values
  residual_data <- data.frame(Residuals = residuals)
  
  histogram_plot <- ggplot(residual_data, aes(x = Residuals)) +
    geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "skyblue", color = "black", alpha = 0.7) +  # use density on the y-axis for the histogram
    geom_density(color = "red", linewidth = 1.5) +  # add the density plot in red
    labs(x = "Residuals", y = "Density", title = paste("Residual Histogram Plot with Density Curve -", model_name)) +
    theme_minimal() +
    xlim(-20,20) + 
    ylim(0, 0.3)
  
  return(histogram_plot)
}

# create histogram plots for residuals for each model
lm_residual_histogram_plot <- create_residual_histogram_plot(
  actual_values = train_data$int_rate,
  predicted_values = lm.train_predictions,
  model_name = "Linear Regression"
)

rf_residual_histogram_plot <- create_residual_histogram_plot(
  actual_values = train_data$int_rate,
  predicted_values = rf.train_predictions$predictions,
  model_name = "Random Forest"
)

xgb_residual_histogram_plot <- create_residual_histogram_plot(
  actual_values = xgb.y_train,
  predicted_values = xgb.train_predictions,
  model_name = "XGBoost"
)

# display the histogram plots separately
print(lm_residual_histogram_plot)
print(rf_residual_histogram_plot)
print(xgb_residual_histogram_plot)

# For each model a bar chart that displays the R-squared (coefficient of determination) values is created.
# R-squared measures the proportion of variance in the target variable explained by the model. 
# Higher R-squared values indicate better model fit.

# create a data frame with R-squared values for each model
model_names <- c("Linear Regression", "Random Forest", "XGBoost")
r_squared_values_train <- c(
  lm.train_r2,
  rf.train_r2,
  xgb.train_r2
)
r_squared_values_test <- c(
  lm.test_r2,
  rf.test_r2,
  xgb.test_r2
)

r_squared_data_train <- data.frame(Model = factor(model_names),
                                   R_squared = r_squared_values_train)
r_squared_data_test <- data.frame(Model = factor(model_names),
                                  R_squared = r_squared_values_test)

# create the R-squared comparison bar chart
r_squared_bar_chart_train <- ggplot(r_squared_data_train, aes(x = Model, y = R_squared, fill = Model)) +
  geom_bar(stat = "identity") +
  labs(x = "Model", y = "R-squared (R²)", title = "R-squared Comparison Training") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  ylim(0,1)
r_squared_bar_chart_test <- ggplot(r_squared_data_test, aes(x = Model, y = R_squared, fill = Model)) +
  geom_bar(stat = "identity") +
  labs(x = "Model", y = "R-squared (R²)", title = "R-squared Comparison Testing") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  ylim(0,1)

# display the R-squared comparison bar chart
print(r_squared_bar_chart_train)
print(r_squared_bar_chart_test)

# A bar chart that compares the MAE or RMSE values, is generated for each model.
# These metrics quantify the average prediction errors of each model, and lower values are preferred.

# create a data frame with MAE and RMSE values for each model
model_names <- c("Linear Regression", "Random Forest", "XGBoost","Linear Regression", "Random Forest", "XGBoost")
error_values_train <- c(
  lm.train_mae,
  rf.train_mae,
  xgb.train_mae,
  lm.train_rmse,
  rf.train_rmse,
  xgb.train_rmse
)
error_values_test <- c(
  lm.test_mae,
  rf.test_mae,
  xgb.test_mae,
  lm.test_rmse,
  rf.test_rmse,
  xgb.test_rmse
)
error_type <- c(
  "MAE", "MAE", "MAE","RMSE","RMSE","RMSE"
)
model_errors_train <- data.frame(Model = factor(model_names, levels = c("Linear Regression", "Random Forest", "XGBoost")),
                                 Error = error_values_train, Type = error_type)
model_errors_test <- data.frame(Model = factor(model_names, levels = c("Linear Regression", "Random Forest", "XGBoost")),
                                Error = error_values_test, Type = error_type)
# create the MAE or RMSE comparison bar chart
error_bar_chart_train <- ggplot(model_errors_train, aes(x = Model, y = Error, fill = Type)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Model", y = "Error Value", title = "Training MAE and RMSE Comparison") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  ylim(0, 4)

error_bar_chart_test <- ggplot(model_errors_test, aes(x = Model, y = Error, fill = Type)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Model", y = "Error Value", title = "Testing MAE and RMSE Comparison") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  ylim(0, 4)

# display the MAE and RMSE comparison bar chart
print(error_bar_chart_train)
print(error_bar_chart_test)


#### Random Forest Feature Importance Plot ####
v1 <- vip(rf, title = "Ranger", num_features = 20) 
plot(v1)

# Feature Selection from the variable importance's analysis:

imp.variables <- lc_data[, v1$data$Variable]
imp.variables$int_rate <- lc_data$int_rate
imp.train_indices <- createDataPartition(imp.variables$int_rate, p = 0.8, list = FALSE)

# create training and testing datasets
imp.train_data <- imp.variables[imp.train_indices, ]
imp.test_data <- imp.variables[-imp.train_indices, ]


#### Linear Regression with only importance variables ####

imp.lm.fit <- lm(int_rate ~ ., data = imp.train_data)

# make predictions on the training and testing data
imp.lm.train_predictions <- predict(imp.lm.fit, newdata = imp.train_data)
imp.lm.test_predictions <- predict(imp.lm.fit, newdata = imp.test_data)

# calculate Mean Squared Error (MSE) for training and testing
imp.lm.train_mse <- mean((imp.lm.train_predictions - imp.train_data$int_rate)^2)
imp.lm.test_mse <- mean((imp.lm.test_predictions - imp.test_data$int_rate)^2)

# calculate Root Mean Squared Error (RMSE) for training and testing
imp.lm.train_rmse <- sqrt(imp.lm.train_mse)
imp.lm.test_rmse <- sqrt(imp.lm.test_mse)

# calculate Mean Absolute Error (MAE) for training and testing
imp.lm.train_mae <- mean(abs(imp.lm.train_predictions - imp.train_data$int_rate))
imp.lm.test_mae <- mean(abs(imp.lm.test_predictions - imp.test_data$int_rate))

# calculate R-squared (R²) for training and testing
imp.lm.train_r2 <- 1 - (sum((imp.train_data$int_rate - imp.lm.train_predictions)^2) / sum((imp.train_data$int_rate - mean(imp.train_data$int_rate))^2))
imp.lm.test_r2 <- 1 - (sum((imp.test_data$int_rate - imp.lm.test_predictions)^2) / sum((imp.test_data$int_rate - mean(imp.test_data$int_rate))^2))

# display the metrics
cat("Training MSE:", imp.lm.train_mse, "\n")
cat("Testing MSE:", imp.lm.test_mse, "\n")
cat("Training RMSE:", imp.lm.train_rmse, "\n")
cat("Testing RMSE:", imp.lm.test_rmse, "\n")
cat("Training MAE:", imp.lm.train_mae, "\n")
cat("Testing MAE:", imp.lm.test_mae, "\n")
cat("Training R-squared (R²):", imp.lm.train_r2, "\n")
cat("Testing R-squared (R²):", imp.lm.test_r2, "\n")

#### Random Forest with only importance variables ####

# train a Random Forest model
imp.rf <- ranger(formula = int_rate ~ ., data = imp.train_data, num.trees = 500, verbose=TRUE, importance = "impurity", oob.error = TRUE)

# print the model summary
print("Random Forest Model Summary:")
print(imp.rf)

# make predictions on the training and testing data
imp.rf.train_predictions <- predict(imp.rf, data = imp.train_data)
imp.rf.test_predictions <- predict(imp.rf, data = imp.test_data)

# calculate Mean Squared Error (MSE) for training and testing
imp.rf.train_mse <- mean((imp.rf.train_predictions$predictions - imp.train_data$int_rate)^2)
imp.rf.test_mse <- mean((imp.rf.test_predictions$predictions - imp.test_data$int_rate)^2)

# calculate Root Mean Squared Error (RMSE) for training and testing
imp.rf.train_rmse <- sqrt(imp.rf.train_mse)
imp.rf.test_rmse <- sqrt(imp.rf.test_mse)

# calculate Mean Absolute Error (MAE) for training and testing
imp.rf.train_mae <- mean(abs(imp.rf.train_predictions$predictions - imp.train_data$int_rate))
imp.rf.test_mae <- mean(abs(imp.rf.test_predictions$predictions - imp.test_data$int_rate))

# calculate R-squared (R²) for training and testing
imp.rf.train_r2 <- 1 - (sum((imp.train_data$int_rate - imp.rf.train_predictions$predictions)^2) / sum((imp.train_data$int_rate - mean(imp.train_data$int_rate))^2))
imp.rf.test_r2 <- 1 - (sum((test_data$int_rate - rf.test_predictions$predictions)^2) / sum((imp.test_data$int_rate - mean(imp.test_data$int_rate))^2))

# display the metrics
cat("Training MSE:", imp.rf.train_mse, "\n")
cat("Testing MSE:", imp.rf.test_mse, "\n")
cat("Training RMSE:", imp.rf.train_rmse, "\n")
cat("Testing RMSE:", imp.rf.test_rmse, "\n")
cat("Training MAE:", imp.rf.train_mae, "\n")
cat("Testing MAE:", imp.rf.test_mae, "\n")
cat("Training R-squared (R²):", imp.rf.train_r2, "\n")
cat("Testing R-squared (R²):", imp.rf.test_r2, "\n")

#### Boosting with only importance variables ####

# define the target variable for training and testing
imp.xgb.y_train <- imp.train_data$int_rate
imp.xgb.y_test <- imp.test_data$int_rate

# define the feature matrix for training and testing (exclude the target variable)
imp.xgb.X_train <- imp.train_data[, -which(names(imp.train_data) == 'int_rate')]
imp.xgb.X_test <- imp.test_data[, -which(names(imp.test_data) == 'int_rate')]

# fit a gradient boosting regression model using xgboost
imp.xgb <- xgboost(
  data = as.matrix(imp.xgb.X_train),
  label = imp.xgb.y_train,
  nrounds = 100,
  verbose = 0
)

# make predictions on the training and testing data
imp.xgb.train_predictions <- predict(imp.xgb, newdata = as.matrix(imp.xgb.X_train))
imp.xgb.test_predictions <- predict(imp.xgb, newdata = as.matrix(imp.xgb.X_test))

# calculate Mean Squared Error (MSE) for training and testing
imp.xgb.train_mse <- mean((imp.xgb.train_predictions - imp.xgb.y_train)^2)
imp.xgb.test_mse <- mean((imp.xgb.test_predictions - imp.xgb.y_test)^2)

# calculate Root Mean Squared Error (RMSE) for training and testing
imp.xgb.train_rmse <- sqrt(imp.xgb.train_mse)
imp.xgb.test_rmse <- sqrt(imp.xgb.test_mse)

# calculate Mean Absolute Error (MAE) for training and testing
imp.xgb.train_mae <- mean(abs(imp.xgb.train_predictions - imp.xgb.y_train))
imp.xgb.test_mae <- mean(abs(imp.xgb.test_predictions - imp.xgb.y_test))

# calculate R-squared (R²) for training and testing
imp.xgb.train_r2 <- 1 - (sum((imp.xgb.y_train - imp.xgb.train_predictions)^2) / sum((imp.xgb.y_train - mean(imp.xgb.y_train))^2))
imp.xgb.test_r2 <- 1 - (sum((imp.xgb.y_test - imp.xgb.test_predictions)^2) / sum((imp.xgb.y_test - mean(imp.xgb.y_test))^2))

# display the metrics
cat("Training MSE:", imp.xgb.train_mse, "\n")
cat("Testing MSE:", imp.xgb.test_mse, "\n")
cat("Training RMSE:", imp.xgb.train_rmse, "\n")
cat("Testing RMSE:", imp.xgb.test_rmse, "\n")
cat("Training MAE:", imp.xgb.train_mae, "\n")
cat("Testing MAE:", imp.xgb.test_mae, "\n")
cat("Training R-squared (R²):", imp.xgb.train_r2, "\n")
cat("Testing R-squared (R²):", imp.xgb.test_r2, "\n")

# The dataset was filtered by the 20 variables with the most importance (from the rf results). 
# As we can see above, the errors of each model are more or less the errors with the double variables we had before, 
# so filtering by these 20 "important variables" does not seem making sense...