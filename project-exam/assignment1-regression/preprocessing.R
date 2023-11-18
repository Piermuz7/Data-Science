# Packages
#install.packages("fastDummies")
#install.packages("readr")
#install.packages("dplyr")
#install.packages("caret")
library(fastDummies)
library(readr)
library(dplyr)
library(caret)

#### Data Preprocessing ####

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
                                       total_pymnt_inv, title, desc, url, member_id, policy_code, emp_title, pymnt_plan, id))

### Data Cleaning ###

# count all NAs in the lc dataset
all_NAs <- colSums(is.na(lc_data))
summary(lc_data)
# drop all columns that have NAs >= 40% of the lc dataset rows
# now the "almost clean" dataset has 31 columns
threshold <- nrow(lc_data) * 0.4
# remove columns with NA values greater than or equal to the threshold
lc_data <- lc_data[, all_NAs < threshold]
# annual_inc: 4 NAs replacement by the mean
tmp_mean <- as.integer(mean(lc_data$annual_inc, na.rm = TRUE))
lc_data$annual_inc <- 
  ifelse(is.na(lc_data$annual_inc),
         tmp_mean, lc_data$annual_inc)
# delinq_2yrs: 25 NAs replacement by the mean
tmp_mean <- as.integer(mean(lc_data$delinq_2yrs, na.rm = TRUE))
lc_data$delinq_2yrs <- 
  ifelse(is.na(lc_data$delinq_2yrs),
         tmp_mean, lc_data$delinq_2yrs)
# earliest_cr_line: 25 NAs replacement by the mode
tmp_mode <- names(sort(table(lc_data$earliest_cr_line), decreasing = TRUE))[1]
lc_data$earliest_cr_line[is.na(lc_data$earliest_cr_line)] <- tmp_mode
# inq_last_6mths: 25 NAs replacement by the mean
tmp_mean <- as.integer(mean(lc_data$inq_last_6mths, na.rm = TRUE))
lc_data$inq_last_6mths <- 
  ifelse(is.na(lc_data$inq_last_6mths),
         tmp_mean, lc_data$inq_last_6mths)
# open_acc: 25 NAs replacement by the mode
tmp_mode <- names(sort(table(lc_data$open_acc), decreasing = TRUE))[1]
lc_data$open_acc[is.na(lc_data$open_acc)] <- tmp_mode
# pub_rec: 25 NAs replacement by the mode
tmp_mode <- names(sort(table(lc_data$pub_rec), decreasing = TRUE))[1]
lc_data$pub_rec[is.na(lc_data$pub_rec)] <- tmp_mode
# revol_bal: 2 NAs replacement by the mean
tmp_mean <- as.integer(mean(lc_data$revol_bal, na.rm = TRUE))
lc_data$revol_bal <- 
  ifelse(is.na(lc_data$revol_bal),
         tmp_mean, lc_data$revol_bal)
# revol_util: 454 NAs replacement by the mean
tmp_mean <- as.integer(mean(lc_data$revol_util, na.rm = TRUE))
lc_data$revol_util <- 
  ifelse(is.na(lc_data$revol_util),
         tmp_mean, lc_data$revol_util)
# total_acc: 25 NAs replacement by the mode
tmp_mode <- names(sort(table(lc_data$total_acc), decreasing = TRUE))[1]
lc_data$total_acc[is.na(lc_data$total_acc)] <- tmp_mode

#########################
# TODO: remaining columns
#########################

# remove all NAs values
lc_data <- na.omit(lc_data)

# remove zero variance columns
lc_data <- subset(lc_data, select = -c(total_rec_late_fee, collections_12_mths_ex_med, acc_now_delinq))

#identify categorical columns
cat_col_names <- names(lc_data)[sapply(lc_data, is.character)]

# extract response variable
repsponse_int_rate <- lc_data$int_rate
lc_data$int_rate <- NULL

# identify categorical columns
lc_data_cat <- lc_data[, names(lc_data)[sapply(lc_data, is.character)]]

# subset data for preprocessing (exclude categorical variables)
lc_data_num <- lc_data[, setdiff(names(lc_data), cat_col_names), drop = FALSE]
# standardize data
lc_data_num <- predict(preProcess(lc_data_num, method = c("center", "scale")), newdata = lc_data_num)

# one-hot encode all categorical columns
lc_data_cat <- dummy_cols(lc_data_cat, remove_selected_columns = TRUE)

lc_data <- cbind(lc_data_cat, lc_data_num)

# reattach response variable
lc_data$int_rate <- repsponse_int_rate

lc_data <- subset(lc_data, select = c(int_rate, names(lc_data_cat)))

# linear regression on ALL pre-processed data attributes:
lm.fit <- lm(int_rate ~ ., data = lc_data)

# calculate RMSE
print(sqrt(mean(lm.fit$residuals^2)))
