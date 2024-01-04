# Data Pre-processing
  
# Load needed libraries
library(readr)
library(ggplot2)
library(dplyr)
library(caret)
library(glmnet)
library(boot)
library(tree)
library(ranger)
library(xgboost)
library(gbm)
library(vip)
library(ISLR)
library(tidyr)
library(gridExtra)
library(reshape2)

# Set the seed for reproducibility
set.seed(1)

# Remove attributes not available for prediction
lc_data <- subset(lc_data, select = -c(collection_recovery_fee, installment, issue_d,
                                       last_pymnt_amnt, last_pymnt_d, loan_status,
                                       next_pymnt_d, out_prncp, out_prncp_inv,
                                       pymnt_plan, recoveries, total_pymnt,
                                       total_pymnt_inv,total_rec_int, total_rec_late_fee, 
                                       total_rec_prncp))

summary(lc_data)

# First we delete the columns which aren't useful for our prediction
lc_data$id <- NULL
lc_data$member_id <- NULL
lc_data$zip_code <- NULL
lc_data$url <- NULL

# Looks like policy_code contains just value equal to 1, it can be removed
lc_data$policy_code <- NULL

# Remove additional columns which are related to the historical data
lc_data$last_credit_pull_d <- NULL

# Then we delete the columns which can't be converted to categorical and require NLP
lc_data$title <- NULL
lc_data$desc <- NULL
lc_data$emp_title <- NULL

# Let's examine the loan_amnt column
sum(is.na(lc_data$loan_amnt))
cor(lc_data$loan_amnt, lc_data$int_rate)
hist(lc_data$loan_amnt, breaks = 20, main = "loan_amnt distribution", xlab = "loan_amnt", col = "lightblue", border = "black")
ggplot(data = lc_data, mapping = aes(x=int_rate,y=loan_amnt)) + geom_boxplot()

# Standardize loan_amnt
#lc_data$loan_amnt <- scale(lc_data$loan_amnt)

# Let's examine the funded_amnt column
sum(is.na(lc_data$funded_amnt))
cor(lc_data$funded_amnt, lc_data$int_rate)
hist(lc_data$funded_amnt, breaks = 20, main = "funded_amnt distribution", xlab = "funded_amnt", col = "lightblue", border = "black")

# As we can see, funded_amnt is almost the same as the loan_amnt column, consequently, we remove it.
lc_data$funded_amnt <- NULL

# Let's examine the funded_amnt_inv column
sum(is.na(lc_data$funded_amnt_inv))
cor(lc_data$funded_amnt_inv, lc_data$int_rate)
hist(lc_data$funded_amnt_inv, breaks = 20, main = "funded_amnt_inv distribution", xlab = "funded_amnt_inv", col = "lightblue", border = "black")

# Remove funded_amnt_inv for the same reason as above
lc_data$funded_amnt_inv <- NULL

# Let's see the int_rate distribution.
hist(lc_data$int_rate, breaks = 20, main = "int_rate distribution", xlab = "int_rate", col = "lightblue", border = "black")

# Standardize int rate:
#lc_data$int_rate <- scale(lc_data$int_rate)

# As we can observe, there are 40363 NAs. We can assume 40363 do not work.
barplot(table(lc_data$emp_length),
        xlab = "emp_length years", 
        ylab = "Frequency", 
        col = "skyblue", 
        border = "black",
        cex.names = 0.6)  # The size of the main title


# Since emp_length seems to be categorical, we transform it to as a factor and then as numeric.
# The conversion to numeric is needed for supporting the Decision Trees and XGBoost
lc_data$emp_length <- as.factor(lc_data$emp_length)
ggplot(data = lc_data, mapping = aes(x=int_rate,y=emp_length)) + geom_boxplot()
lc_data$emp_length <- as.numeric(lc_data$emp_length)

# As we can see, term plays a crucial role in predicting the interest rate.
lc_data$term <- as.factor(lc_data$term)
ggplot(data = lc_data, mapping = aes(x=int_rate,y=term)) + geom_boxplot()
lc_data$term <- as.numeric(lc_data$term)

# Cleaning of home_ownership:
# During the data cleaning phase, our analysis revealed that the variable "home_ownership" does not show
# a distinct correlation with interest rates. Specifically, among the categories, "ANY" and "OTHER" 
# contain 2 and 154 cases, respectively, while the "NONE" category comprises 39 cases.
# Although the "NONE" category appears to demonstrate a higher interest rate compared to others,
# the limited sample size of 39 cases raises doubts about the reliability of this observation.
# Notably, the "NONE" category might pertain to individuals experiencing homelessness,
# prompting ethical concerns about loan provision to this demographic.
table(lc_data$home_ownership)
ggplot(data = lc_data, mapping = aes(x=int_rate,y=home_ownership)) + geom_boxplot()

# Then, we retain mortgage, own and rent:
lc_data <- lc_data %>% filter(home_ownership %in% c("MORTGAGE","OWN","RENT"))
lc_data$home_ownership <- as.numeric(as.factor(lc_data$home_ownership))

# Most of the loan applications are Individual, this means that most of the values of the columns dti_joint,
# annual_inc_joint and verification_status_joint are Null.
# We would like to keep the information about Joint loans, this means that we can replace the Null values with 0.
nav <- c('', ' ')
lc_data <- transform(lc_data, verification_status_joint=replace(verification_status_joint, verification_status_joint %in% nav, NA))
lc_data <-
  lc_data %>%
  mutate(dti_joint = ifelse(is.na(dti_joint) == TRUE, 0, dti_joint)) %>%
  mutate(annual_inc_joint = ifelse(is.na(annual_inc_joint) == TRUE, 0, annual_inc_joint)) %>%
  mutate(verification_status_joint = ifelse(is.na(verification_status_joint) == TRUE, 'NA', verification_status_joint))

# The empty string or null value in verification_status_joint is replaced successfully.
table(lc_data$verification_status)
table(lc_data$verification_status_joint)

# Then verification_status_joint and verification_status columns are converted in categorical and then numerical value.
# The column application_type is obsolete, since the information about whether the loan is individual or joint is
# already contained in the previous variables.
lc_data$verification_status <- as.numeric(as.factor(lc_data$verification_status))
lc_data$verification_status_joint <- as.numeric(as.factor(lc_data$verification_status_joint))
lc_data <- lc_data %>% select(-application_type)

# Let's check if other is NA or a real value for purpose. It's a real one, so we don't have to handle it.
lc_data$purpose <- as.factor(lc_data$purpose)
ggplot(data = lc_data, mapping = aes(x=int_rate,y=purpose)) + geom_boxplot()
lc_data$purpose <- as.numeric(lc_data$purpose)

# Let's have a glance to the state address:
table(lc_data$addr_state)
lc_data$addr_state <- as.factor(lc_data$addr_state)
ggplot(data = lc_data, mapping = aes(x=int_rate,y=addr_state)) + geom_boxplot()
lc_data$addr_state <- as.numeric(lc_data$addr_state)

# Regarding delinquency in the last 2 years, there are few NAs then remove them:
lc_data <- lc_data %>% 
  filter(!(is.na(delinq_2yrs)))

# The columns mths_since_delinq_cat, mths_since_last_record, mths_since_rcnt_il and mths_since_last_major_derog
# contain numerical values which refer to the number of the months. 
# Since this columns contain a lot of null values which can't be replaced with 0's, 
# one of the most appropriate operations that can be made is applying discretization. 
# We do this by creating a set of contiguous bins based on years, while for the null values we create a separate bin.
lc_data <- lc_data %>%
  mutate(mths_since_delinq_cat = ifelse(
    is.na(mths_since_last_delinq) == TRUE,
    "NONE",
    ifelse(
      mths_since_last_delinq <= 12,
      "Less_1_Y",
      ifelse(
        mths_since_last_delinq <= 24,
        "Less_2_Y",
        ifelse(
          mths_since_last_delinq <= 36,
          "Less_3_Y",
          ifelse(mths_since_last_delinq <= 48, "Less_4_Y", "More_4_Y")
        )
      )
    )
  )) %>% select(-mths_since_last_delinq)

lc_data$mths_since_delinq_cat <- as.factor(lc_data$mths_since_delinq_cat)
ggplot(data = lc_data, mapping = aes(x=int_rate,y=mths_since_delinq_cat))+geom_boxplot()
lc_data$mths_since_delinq_cat <- as.numeric(lc_data$mths_since_delinq_cat)


lc_data <- lc_data %>%
  mutate(mths_since_last_record_cat = ifelse(
    is.na(mths_since_last_record) == TRUE,
    "NONE",
    ifelse(
      mths_since_last_record <= 12,
      "Less_1_Y",
      ifelse(
        mths_since_last_record <= 24,
        "Less_2_Y",
        ifelse(
          mths_since_last_record <= 36,
          "Less_3_Y",
          ifelse(mths_since_last_record <= 48, "Less_4_Y", "More_4_Y")
        )
      )
    )
  )) %>% select(-mths_since_last_record)

lc_data$mths_since_last_record_cat <- as.factor(lc_data$mths_since_last_record_cat)
ggplot(data = lc_data, mapping = aes(x=int_rate,y=mths_since_last_record_cat))+geom_boxplot()
lc_data$mths_since_last_record_cat <- as.numeric(lc_data$mths_since_last_record_cat)


lc_data <-lc_data %>% 
  mutate(mths_since_rcnt_il_cat =  ifelse(
    is.na(mths_since_rcnt_il) == TRUE,
    "NONE",
    ifelse(
      mths_since_rcnt_il <= 12,
      "Less_1_Y",
      ifelse(
        mths_since_rcnt_il <= 24,
        "Less_2_Y",
        ifelse(
          mths_since_rcnt_il <= 36,
          "Less_3_Y",
          ifelse(mths_since_rcnt_il <= 48, "Less_4_Y", "More_4_Y")
        )
      )
    )
  )) %>% select(-mths_since_rcnt_il)

lc_data$mths_since_rcnt_il_cat <- as.factor(lc_data$mths_since_rcnt_il_cat)
ggplot(data = lc_data, mapping = aes(x=int_rate,y=mths_since_rcnt_il_cat))+geom_boxplot()
lc_data$mths_since_rcnt_il_cat <- as.numeric(lc_data$mths_since_rcnt_il_cat)


lc_data <-lc_data %>% 
  mutate(mths_since_last_major_derog_cat =  ifelse(
    is.na(mths_since_last_major_derog) == TRUE,
    "NONE",
    ifelse(
      mths_since_last_major_derog <= 12,
      "Less_1_Y",
      ifelse(
        mths_since_last_major_derog <= 24,
        "Less_2_Y",
        ifelse(
          mths_since_last_major_derog <= 36,
          "Less_3_Y",
          ifelse(mths_since_last_major_derog <= 48, "Less_4_Y", "More_4_Y")
        )
      )
    )
  )) %>% select(-mths_since_last_major_derog)

lc_data$mths_since_last_major_derog_cat <- as.factor(lc_data$mths_since_last_major_derog_cat)
ggplot(data = lc_data, mapping = aes(x=int_rate,y=mths_since_last_major_derog_cat))+geom_boxplot()
lc_data$mths_since_last_major_derog_cat <- as.numeric(lc_data$mths_since_last_major_derog_cat)


# The variable initial_list_status identifies whether a loan was initially listed in the whole (W) or fractional (F) market.
# This variable could be useful so we can keep it and transform it to a factor and then to a numeric value, 
# for the same purpose of compatibility with the XGBoost function.

lc_data$initial_list_status <- as.factor(lc_data$initial_list_status)
ggplot(data = lc_data, mapping = aes(x=int_rate,y=initial_list_status))+geom_boxplot()
lc_data$initial_list_status <- as.numeric(lc_data$initial_list_status)

# Let's check which columns still have null values
colSums(is.na(lc_data))

#The columns revol_bal and revol_util contain only few NA values, those values can't be replaced with 0, then we filter the values which are not NAs.
lc_data <- lc_data %>% 
  filter(!(is.na(revol_bal))) %>% 
  filter(!(is.na(revol_util)))

# Let's check which columns still have null values:
names(which(colSums(is.na(lc_data)) > 0))

# Replace null values with 0 where is possible
lc_data <-
  lc_data %>%
  mutate(open_acc_6m = ifelse(is.na(open_acc_6m) == TRUE, 0, open_acc_6m)) %>%
  mutate(tot_cur_bal = ifelse(is.na(tot_cur_bal) == TRUE, 0, tot_cur_bal)) %>%
  mutate(open_il_6m = ifelse(is.na(open_il_6m) == TRUE, 0, open_il_6m)) %>%
  mutate(open_il_12m = ifelse(is.na(open_il_12m) == TRUE, 0, open_il_12m)) %>%
  mutate(open_il_24m = ifelse(is.na(open_il_24m) == TRUE, 0, open_il_24m)) %>%
  mutate(total_bal_il = ifelse(is.na(total_bal_il) == TRUE, 0, total_bal_il)) %>%
  mutate(il_util = ifelse(is.na(il_util) == TRUE, 0, il_util)) %>%
  mutate(open_rv_12m = ifelse(is.na(open_rv_12m) == TRUE, 0, open_rv_12m)) %>%
  mutate(total_rev_hi_lim = ifelse(is.na(total_rev_hi_lim) == TRUE, 0, total_rev_hi_lim)) %>%
  mutate(max_bal_bc = ifelse(is.na(max_bal_bc) == TRUE, 0, max_bal_bc)) %>%
  mutate(all_util = ifelse(is.na(all_util) == TRUE, 0, all_util)) %>%
  mutate(inq_fi = ifelse(is.na(inq_fi) == TRUE, 0, inq_fi)) %>%
  mutate(total_cu_tl = ifelse(is.na(total_cu_tl) == TRUE, 0, total_cu_tl)) %>%
  mutate(inq_last_12m = ifelse(is.na(inq_last_12m) == TRUE, 0, inq_last_12m)) %>%
  mutate(open_rv_24m = ifelse(is.na(open_rv_24m) == TRUE, 0, open_rv_24m)) %>%
  mutate(tot_coll_amt = ifelse(is.na(tot_coll_amt)== TRUE,0, tot_coll_amt)) %>%
  mutate(collections_12_mths_ex_med = ifelse(is.na(collections_12_mths_ex_med)== TRUE,0, collections_12_mths_ex_med))

# earliest_cr_line contains the month the borrower's earliest reported credit line was opened.
# Even if this date consists only on month and year, still there are too many unique values.
# We could transform the dates in to a numerical value, by converting them from date into Unix Time.
# This unit measures time by the number of seconds that have elapsed since 00:00:00 UTC on 1 January 1970.
# Since this column doesn't contain the day number, we take as a reference the first day of the month.
lc_data <- lc_data %>% 
    filter(!(is.na(earliest_cr_line)))

# function to replace dates with unix time
to_unix_time <- function(date) {
  tmp <- paste("01", date, sep="-")
  return (as.numeric(as.POSIXct(tmp, format="%d-%b-%Y", tz="UTC")))
}

# map dates to unix time
lc_data$earliest_cr_line <- apply(lc_data, 1, function(row) to_unix_time(row["earliest_cr_line"]))

# standardize them
#lc_data$earliest_cr_line <- scale(lc_data$earliest_cr_line)

# Outliers Removal:
boxplot(lc_data$int_rate)
# Identify outliers using boxplot
outliers <- boxplot(lc_data$int_rate, plot = FALSE)$out
# Remove outliers from the dataset
lc_data <- lc_data[!lc_data$int_rate %in% outliers, ]

summary(lc_data)