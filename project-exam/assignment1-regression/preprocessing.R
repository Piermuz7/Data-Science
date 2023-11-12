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
lc_data <- read_delim("LCdata.csv")
# remove unnecessary columns
lc_data <- subset(lc_data, select = -c(collection_recovery_fee, installment, issue_d,
                                       last_pymnt_amnt, last_pymnt_d, loan_status,
                                       next_pymnt_d, out_prncp, out_prncp_inv,
                                       pymnt_plan, recoveries, term, total_pymnt,
                                       total_pymnt_inv, title, desc, url, member_id, policy_code, emp_title, pymnt_plan, id))
# remove all NAs values
lc_data <- lc_data[complete.cases(lc_data),]

# identify categorical columns
cat_col_names <- names(lc_data)[sapply(lc_data, is.character)]

# subset data for preprocessing (exclude categorical variables)
lc_data_num <- lc_data[, setdiff(names(lc_data), cat_col_names), drop = FALSE]

# first standardize and then normalize data
preprocess <- preProcess(lc_data_num, method = c("center", "scale"))

# Print summary statistics of the preprocessed data
print(summary(predict(preprocess, newdata = lc_data_num)))


# one-hot encode all categorical columns
#lc_data <- dummy_cols(lc_data, select_columns = cat_col_names, remove_selected_columns = TRUE)


# linear regression on ALL pre-processed data attributes:
#lm.fit <- lm(int_rate~., data=lc_data)

# calculate RMSE
#sqrt(mean(lm.fit$residuals^2))