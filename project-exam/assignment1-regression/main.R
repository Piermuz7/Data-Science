# Load the dataset
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
original_lc_data <- read.csv("LCdata.csv",sep = ";")
lc_data <- original_lc_data

# Data Preprocessing
source("preprocessing_final_version.R")

# Learning Algorithms
source("learning_algorithms.R")

# CSV Dataset after Data Preprocessing
write.csv(lc_data,"cleaned_dataset.csv")

# CSV Training and Testing Datasets
write.csv(train_data,"train_dataset.csv")
write.csv(test_data,"test_dataset.csv")

# Best Model Saving
saveRDS(xgb, file = "best_model_xgb.rds")