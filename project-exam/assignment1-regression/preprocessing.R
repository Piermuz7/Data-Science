# Packages
#install.packages("fastDummies")
#install.packages("readr")
library(fastDummies)
library(readr)

#### Data Preprocessing ####

# set the working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# read the csv
lc_data <- read.csv("LCdata.csv", sep = ";", fileEncoding="UTF-8")
lc_data <- subset(lc_data, select = -c(collection_recovery_fee, installment, issue_d,
                                 last_pymnt_amnt, last_pymnt_d, loan_status,
                                 next_pymnt_d, out_prncp, out_prncp_inv,
                                 pymnt_plan, recoveries, term, total_pymnt,
                                 total_pymnt_inv, title, desc, url, member_id, policy_code, emp_title, pymnt_plan,id))

# TODO: remove NAs values

# Identify categorical columns
cat_col_names <- names(lc_data)[sapply(lc_data, is.character)]

# One-Hot encode all categorical columns
lc_data <- dummy_cols(lc_data, select_columns = cat_col_names, remove_selected_columns = TRUE)

# linear regression on ALL pre-processed data attributes:
lm.fit <- lm(int_rate~., data=lc_data)

# calculate RMSE
sqrt(mean(lm.fit$residuals^2))