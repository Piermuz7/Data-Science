# Packages
#install.packages("fastDummies")
#install.packages("readr")
#install.packages("dplyr")
#install.packages("caret")
library(fastDummies)
library(readr)
library(dplyr)
library(caret)

#### Functions ####
# function to replace NAs by column mean
replace_na_by_mean <- function(col) {
  tmp_mean <- as.integer(mean(lc_data[[col]], na.rm = TRUE))
  lc_data[[col]] <- ifelse(is.na(lc_data[[col]]), tmp_mean, lc_data[[col]])
}

# function to replace NAs by column mode
replace_na_by_mode <- function(col) {
  tmp_mode <- names(sort(table(lc_data[[col]]), decreasing = TRUE))[1]
  lc_data[[col]] <- ifelse(is.na(lc_data[[col]]), tmp_mode, lc_data[[col]])
}

# function to replace dates with unix time
to_unix_time <- function(date) {
  tmp <- paste("01", date, sep="-")
  return (as.numeric(as.POSIXct(tmp, format="%d-%b-%Y", tz="UTC")))
}

#### Data Preprocessing ####

set.seed(42)  # For reproducibility

# set the working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# read the csv
original_lc_data <- read_delim("LCdata.csv")
lc_data <- original_lc_data

# remove unnecessary columns
lc_data <- subset(lc_data, select = -c(collection_recovery_fee, installment, issue_d,
                                       last_pymnt_amnt, last_pymnt_d, loan_status,
                                       next_pymnt_d, out_prncp, out_prncp_inv,
                                       pymnt_plan, recoveries, term, total_pymnt,
                                       total_pymnt_inv, title, desc, url, member_id, policy_code, emp_title, pymnt_plan, id, zip_code))

### Data Cleaning ###

# count all NAs in the lc dataset
all_NAs <- colSums(is.na(lc_data))
summary(lc_data)
# drop all columns that have NAs >= 40% of the lc dataset rows
# now the "almost clean" dataset has 31 columns
threshold <- round(nrow(lc_data) * 0.4)
# remove columns with NA values greater than or equal to the threshold
lc_data <- lc_data[, all_NAs < threshold]

# test omit rows contain NA
lc_data <- na.omit(lc_data)

# map dates to unix time
lc_data$earliest_cr_line <- apply(lc_data, 1, function(row) to_unix_time(row["earliest_cr_line"]))
lc_data$last_credit_pull_d <- apply(lc_data, 1, function(row) to_unix_time(row["last_credit_pull_d"]))

## NA handling step ##
na_mean <- c("last_credit_pull_d","earliest_cr_line","annual_inc","delinq_2yrs","inq_last_6mths", "revol_bal", "revol_util", "tot_coll_amt","tot_cur_bal","total_rev_hi_lim")
na_mode <- c("open_acc", "pub_rec", "total_acc", "collections_12_mths_ex_med","acc_now_delinq")

# apply function to replace each column value by mean
lc_data[na_mean] <- lapply(na_mean, replace_na_by_mean)
# apply function to replace each column value by mode
lc_data[na_mode] <- lapply(na_mode, replace_na_by_mode)

# Categorical variables have to be mapped into numerical
cat_columns <- c("emp_length", "home_ownership", "verification_status", "purpose", "addr_state", "initial_list_status", "application_type")
# Continuous numerical variables have to be standardized
num_columns <- c("earliest_cr_line","last_credit_pull_d","loan_amnt", "annual_inc", "dti", "revol_bal", "revol_util", "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim")
# extract response variable
response_int_rate <- lc_data$int_rate
lc_data$int_rate <- NULL

# subset data for preprocessing (include numerical variables only)
lc_data_num <- lc_data[num_columns]
# standardize data
lc_data_num <- predict(preProcess(lc_data_num, method = c("center", "scale")), newdata = lc_data_num)

# subset data for preprocessing (include categorical variables only)
lc_data_cat <- lc_data[cat_columns]
# one-hot encode all categorical columns
lc_data_cat <- dummy_cols(lc_data_cat, remove_selected_columns = TRUE)

lc_data <- cbind(lc_data_cat, lc_data_num)

# reattach response variable
lc_data$int_rate <- response_int_rate

# Create indices for splitting (80% train, 20% test)
train_indices <- createDataPartition(lc_data$int_rate, p = 0.8, list = FALSE)

# Create training and testing datasets
train_data <- lc_data[train_indices, ]
test_data <- lc_data[-train_indices, ]

#lc_data <- subset(lc_data, select = c(int_rate, names(lc_data_cat)))

sampled_data <- lc_data %>% sample_frac(0.1)
lm.fit <- lm(int_rate ~ ., data = train_data)

#vif(lm.fit) # there is multicollinearity
#cor(lc_data) 

# Make predictions on training and testing data
train_predictions <- predict(lm.fit, newdata = train_data)
test_predictions <- predict(lm.fit, newdata = test_data)

# Evaluate model performance on training data
train_rmse <- sqrt(mean((train_predictions - train_data$int_rate)^2))
train_r_squared <- summary(lm.fit)$r.squared

# Evaluate model performance on testing data
test_rmse <- sqrt(mean((test_predictions - test_data$int_rate)^2))
test_r_squared <- summary(lm.fit, test_data)$r.squared

# Print evaluation metrics
cat("Training RMSE:", train_rmse, "\n")
cat("Training R-squared:", train_r_squared, "\n")
cat("\nTesting RMSE:", test_rmse, "\n")
cat("Testing R-squared:", test_r_squared, "\n")

# linear regression on ALL pre-processed data attributes:
lm.fit <- lm(int_rate ~ ., data = lc_data)

# calculate RMSE
print(sqrt(mean(lm.fit$residuals^2)))
