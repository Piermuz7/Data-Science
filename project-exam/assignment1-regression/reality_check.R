# Loading the new "secret" data
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
lc_data <- read.csv("secret_data.csv",sep = ";")

# Transforming and preprocessing it
source("preprocessing_final_version.R")

# Applying the final model to it, and making a prediction for it
best_model_xgb <- readRDS("best_model_xgb.rds")

# Splitting in input and target data
best_model_xgb.X <- lc_data[, -which(names(lc_data) == 'int_rate')]
best_model_xgb.y <- lc_data$int_rate

# Evaluate the prediction on the new data
best_model_xgb.X.predictions <- predict(best_model_xgb, newdata = as.matrix(best_model_xgb.X))

# calculate Mean Squared Error (MSE) for new data
best_model_xgb.mse <- mean((best_model_xgb.X.predictions - best_model_xgb.y)^2)

# calculate Root Mean Squared Error (RMSE) for new data
best_model_xgb.rmse <- sqrt(best_model_xgb.mse)

# calculate Mean Absolute Error (MAE) for new data
best_model_xgb.mae <- mean(abs(best_model_xgb.X.predictions - best_model_xgb.y))

# calculate R-squared (R²) for new data
best_model_xgb.r2 <- 1 - (sum((best_model_xgb.y - best_model_xgb.X.predictions)^2) / sum((best_model_xgb.y - mean(best_model_xgb.X.predictions))^2))

# Display the metrics
cat("Best XGB MSE:", best_model_xgb.mse, "\n")
cat("Best XGB RMSE:", best_model_xgb.rmse, "\n")
cat("Best XGB MAE:", best_model_xgb.mae, "\n")
cat("Best XGB R-squared (R²):", best_model_xgb.r2, "\n")