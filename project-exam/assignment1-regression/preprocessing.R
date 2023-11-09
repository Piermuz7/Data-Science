# set the working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# read the csv
data <- read.csv("LCdata.csv", sep = ";", fileEncoding="UTF-8")
print(names(data))
data <- subset(data, select = -c(collection_recovery_fee, installment, issue_d,
                                 last_pymnt_amnt, last_pymnt_d, loan_status,
                                 next_pymnt_d, out_prncp, out_prncp_inv,
                                 pymnt_plan, recoveries, term, total_pymnt,
                                 total_pymnt_inv, title, desc, url, member_id, policy_code, emp_title, pymnt_plan,id))
