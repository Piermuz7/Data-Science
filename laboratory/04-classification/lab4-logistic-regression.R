#### Lab 4: Classification: Logistic Regression ####
#
# TASK:
# Fit a logistic regression model to the Data Set Default of the ISLR library to predict the class variable default.
#

install.packages("ISLR")
install.packages("ggplot2")
install.packages("gridExtra")

library(ISLR)
library(ggplot2)
library(gridExtra)

#### Data exploration of the "Default" data set  ####


# Split the dataset in training and test set

# Since standardization of the input variables is not needed for logistic regression,
# we don't reuse the training/tets split we did for kNN, but instead apply the split to the original dataset:

set.seed(1)
(indices <- sort(sample(1:dim(Default)[1], 2000))) # create 2000 randomly sampled indices -  we use a 80% - 20% split
(test.data <- Default[indices,]) # select the corresponding observations for the test set
(training.data <- Default[-indices,]) # select the remaining observations for the training set

# Logistic Regression on "balance"

# Fit a logistic regression model with "balance" as the only predictor
set.seed(1)
glm.fit <- glm(default~balance,family="binomial", data=training.data)
summary(glm.fit)
# Both parameters are significant
# The coefficient associated with balance is positive. This indicates that customers with higher balance tend to have higher default probabilities
# The intercept and slope "live" in the backtransfomed logit-space
# If we increase the credit card balance by 1 unit, the logit-value (backtransformed value) of the probability to default increases by 0.55%
# We cannot directly translate this in the probability space, because the logit function is non-linear
# But instead, we can calculate & plot the probability values that correspond to a number of balances in order to get a feeling for the whole thing
# Manually, we would do that by calculating y= ax+b for every given balance value x, and then use the "logistic function" to transform y in the probability space
# Fortunately, we can use the predict() function to that for us
# As a start, we use the training data and let the function predict it's probabilities

# Training error rate

# We calculate the fitted probabilities for the training data
(pred.train.probs = predict(glm.fit, training.data, type = "response"))
# With type = "response", we get the probabilities. If we leave it out, we get the values on the regression line before the transformation

# We plot the fitted training probabilities
# Since the output of predict() is a vector, we need to build a dataframe before we can use ggplot2:
pred.train.probs.df <- data.frame(balance = training.data$balance, pred.train.probs = pred.train.probs)
ggplot() + geom_point(data = pred.train.probs.df, aes(x=balance, y=pred.train.probs, col=training.data$default), size=1) + geom_hline(yintercept = 0) + geom_hline(yintercept = 1) + geom_hline(yintercept = 0.5)
# We can see that the regression line parameters a and b where indeed fit so that most observed defaults (green) are close to 1 (100% default probability),
# and most observed non-defaults (red) are close to 0 (0% default probability)

# We calculate the fitted training class values
# To check the training error rate, we first need to convert the probabilities in a decision vector
# To do that, we need to set a threshold on the probabilities that implements the decision
# The canonical threshold is 50%, since this fits with the maximum likelihood method
# To build the decision vector, we initialize it as a vector of "No"s:
(pred.training.classes = rep("No",nrow(training.data)))
# Now all values of the vector to "Yes" whose probability is greater than 0.5:
pred.training.classes[pred.train.probs > 0.5] = "Yes"

# We plot the fitted training class values
pred.training.classes.df <- data.frame(balance=training.data$balance,pred.training.classes=pred.training.classes) # make it a data frame for plotting
ggplot() + geom_point(data = pred.training.classes.df, aes(x=balance, y=pred.training.classes, col=training.data$default), size=1)
# We can see that most observed defaults are mapped to "Yes" and most observed non-defaults are mapped to "No"

# We calculate the training error rate
mean(pred.training.classes != training.data$default)
# The training error rate is 2.6%
# This is smaller than the training error rate of 3.32% of the trivial classifier that always predicts "No" for the training data:
table(training.data$default) # 263/8000 = 0.03288
# Yet, this is only evaluating the training error rate - it may well overfit!

# Test error rate

# We predict the test data probabilities
pred.test.probs = predict(glm.fit, test.data, type = "response")

# We plot them
pred.test.probs.df <- data.frame(balance=test.data$balance,pred.test.probs=pred.test.probs) # make it a data frame fro plotting
ggplot() + geom_point(data = pred.test.probs.df, aes(x=balance, y=pred.test.probs, col=test.data$default), size=1) + geom_hline(yintercept = 0) + geom_hline(yintercept = 1) + geom_hline(yintercept = 0.5, linetype="dashed") + ylim(0,1)

# We calculate the predicted test class values
pred.test.classes = rep("No",nrow(test.data)) # initialize the decision vector
pred.test.classes[pred.test.probs > 0.5] = "Yes"  # populate the decision vector
pred.test.classes.df <- data.frame(balance=test.data$balance,pred.test.classes=pred.test.classes) # make a data frame for plotting
ggplot() + geom_point(data = pred.test.classes.df, aes(x=balance, y=pred.test.classes, col=test.data$default), size=1)

# We calculate the test error rate
mean(pred.test.classes != test.data$default)
# The test error rate is 3.15%. Not surprisingly, that is higher than the training error rate of 2.6%
# It performs barely better than the trivial "No" classifier with 3.5%:
table(test.data$default) # 70/2000 = 0.035

# Logistic Regression on a qualitative predictor

# "student" is a categorical predictor. Let's see how one-hot-encoding is implemented in logistic regression:
glm.fit.stud <- glm(default~student,family="binomial", data=training.data)
summary(glm.fit.stud)
# The variables are significant
# The dummy variable studentYes is positive
# This indicates that students tend to have higher default probabilities than non-students

# Inspect the test probabilities
pred.test.probs.stud = predict(glm.fit.stud, test.data, type = "response")
# Notice that only two values can be predicted! (One for each dummy variable)
# We can extract them using unique():
unique(pred.test.probs.stud)
# Non-students are predicted to have a probability of 2.68% to default
# Students are predicted to have a probability of 4.74% to default
# Notice that none of the probabilities is higher than 50%!
# This none of the test set customers will be predicted to default by this model that only uses student as an input

# Calculate the test error rate
# Since all test data points are predicted "No" by glm.fit.stud, the test error rate is easy to calculate manually:
table(test.data$default)
# 70 "Yes" out of 2000, thus the test error rate is 70/2000 = 3.5%

# Logistic Regression on all predictors

# Fitting the model to the training data
glm.fit.all <- glm(default~balance + student + income, family = "binomial", data = training.data)
summary(glm.fit.all)
# All input variables except income are significant. Lets kick it out
glm.fit.2 <- glm(default~balance + student , family = "binomial", data = training.data) 
summary(glm.fit.2)
# All input variables are significant
# The dummy variable studentYes is now negative
# This indicates that students tend to have a lower default probability than non-students
# This seems to contradict the results of the single-variable-model glm.fit.student
# The reason for this is confounding:
# Students on average have a higher default probability than non-students
# But that's only because they usually have a higher credit card balance than non-student customers, which is known to increase the default risk:
boxplot(balance~default, data=Default)
# This example shows that leaving out a variable ("balance" in this case) can introduce a model bias
# Note: This holds not only for classification tasks, but also for regression tasks

# Predict the test data probabilities
pred.test.probs.2 = predict(glm.fit.2, test.data, type = "response")

# We calculate the predicted test class values
pred.test.classes.2 = rep("No",nrow(test.data)) # initialize the decision vector
pred.test.classes.2[pred.test.probs.2 > 0.5] = "Yes"  # populate the decision vector

# We calculate the test error rate 
mean(pred.test.classes.2 != test.data$default)
# The test error rate is 2.85% - this is slightly better than our first model's test error rate of 3.15%
# and identical to the test error rate of our 20NN classifier from Lab 1
# To performs a bit better than the trivial "No" classifier, which has a test error rate of 3.5% on test data
# Still, this is not a really good result
# The success rate in catching the bad guys is still relatively low with 32.8%:
table(pred.test.classes.2, test.data$default) # 23/(23+47) true positive rate for "default"

# Dealing with unbalanced data

# The problem in the data set lies in the fact that the target variable is highly unbalanced:
table(Default$default)
# Only 3.33% of all customers default on their credit card
# It is very hard to predict something that happens so rarely!

# In practice, we are often more interested in predicting one specific class
# If the data set is unbalanced and this class is the minority class, we have a problem:
# A model that is able to accurately predict a minority class in an unbalanced data set has to be really good 
# to be able to beat the trivial classifier
# Do notice that this is only a problem when trying to predict the minority class
# OVERALL, an error rate of 3.5% is a very low!
# If we would want to predict non-default, it's only wrong in 3.5% of the cases - that is really good!
# (However, in this case, we also could employ the trivial classifier instead - this is less effort and performs equally good)

# How can we solve the problem of unbalanced data?
# If we are more interested in predicting one specific class, we can adapt the decision threshold so that it favors this class
# Lets try to set the threshold to 30% instead of 50%
pred.test.classes.2.30 = rep("No",nrow(test.data)) # initialize the decision vector
pred.test.classes.2.30[pred.test.probs > 0.3] = "Yes"  # populate the decision vector

# We calculate the test error rate
mean(pred.test.classes.2.30 != test.data$default)
# The OVERALL test error rate is worse than before.
# This is logical, since more of the "good guys" are now put in the "Yes" category
# The model predicts that they will default, while they actually will not (They are "false positives")
# But, as a compensation, the success rate in catching bad guys improved massively to 51% !
table(pred.test.classes.2.30, test.data$default) # 36/(36+34) = 0.5143 "true positive rate"

# Notice that there is always a trade-off between false positive rate and true positive rate
# It always depends on the domain context, if you wanna lower a threshold or not
# E.g., in the case of suspected credit card fraud, it may be cheaper for the credit card company to reject a credit card
# application of a "good guy" (a false positive) than accepting an application of a "bad guy" (false negative)