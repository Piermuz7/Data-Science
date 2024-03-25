#####################################################
# Data Science
# Subset Selection
# Dr. Gwendolin Wilke
# gwendolin.wilke@fhnw.ch
#####################################################
# Contents:
# 1. Preliminaries
# 2. Data Exploration 
# 3. Manual Feature Selection
# 4. Best Subset Selection
#    4.1 Using regsubsets() to find the best model for each model size k
#    4.2 Using AdjR2, AIC, Cp and BIC to find the best model size k 
#    4.3 Using Cross Validation Error to find the best model size k
# 5. Stepwise Selection
#####################################################




##############   1. Preliminaries   ##########################

# load libraries
library(ISLR)      # contains Credit data set
library(tidyverse) # for nice plots, include automatically ggplot2 and other utilities
library(corrplot)

Cr1 <- Credit # make a copy of the original data set, so that we dont mess with it

##############   2. Data Exploration   ##########################

## Read the data dictionary 
?Credit 
  # For accessing the help page, we need to use the original name of the data set.
  # In this exercise, we will try to predict the credit card balance from the other variables (regression task).

## Examine Data Set
View(Cr1)
str(Cr1)
summary(Cr1)
  # We see here that we have 12 attributes. 
    # We want to predict balance, which leaves us with 11 predictors.
    # 4 of the predictors are categorical, i.e., they need to be converted in dummy variables.
  # We also see that Student is highly unbalanced.

## Visual exploration
library(Hmisc)
hist(Cr1)
  # We check the distributions of the variables.
  # Many of them are skewed. This does not impact linear regression directly, but we can keep it in mind.
  # We see also here that Student is highly unbalanced. 
  # Be careful to notice that the x-axes not always start with zero!  
pairs(Cr1) 
  # Use the zoom in the plot window to see it better.
  # We see that limit, rating and income are highly collinear. 
    # That's not very surprising, since an individual's credit limit is directly related to their credit rating. Both are specified by the bank.
    # The bank maybe derives the rating mainly from an individual's income.
    # For linear regression, collinear predictors are bad, because they make coefficient estimates highly unstable! 
    # We therefore want to exclude some of these predictors from our fit later on.
    # We double-check the collinearities using corrplot(): 
corrplot(cor(select(Cr1, -c(Gender, Student, Married, Ethnicity))))
    # We indeed see an obvious cluster in these variables.
    # We also see that Balance is highly correlated with Limit and Rating - so these may be good predictors in linear regression.
    # Income may be an ok predictor, too.
    # Let's additionally look at the numbers (pearson's correlation coefficients):
cor(Cr1$Limit, Cr1$Rating) 
cor(Cr1$Limit, Cr1$Income) 
cor(Cr1$Rating, Cr1$Income)
cor(Cr1$Limit, Cr1$Balance)
cor(Cr1$Rating, Cr1$Balance)
cor(Cr1$Income, Cr1$Balance)
  # To see the impact of the 4 categorical variables on balance, corrplot() does not help, and pairs() only shows scatter diagrams. 
  # To see it more clearly, we can use boxplots. 
  # plot() provides the boxplotsby default: 
plot(Balance ~ ., data = Cr1)
  # Only Student shows some obvious impact on balance. 
  # Yet, since Students=Yes is under-represented in the dataset, we need to be a bit careful here.

# Check for missing values to make sure our algorithms work
sum(is.na(Cr1)) # none

##############   3. Manual Feature Selection   ##########################

# Now let's kick out one of the highly collinear variables:
Cr1 <- select(Credit, -c(Limit))

# Additionally we can manually kick out ID 
Cr1 <- select(Cr1, -c(ID))

# Let's check if we are good:
library(car)
lm.fit <- lm(Balance~., data=Cr1)
vif(lm.fit)
   # No value >= 5, all good.

##############   4. BEST SUBSET SELECTION  ############

# The Best Subset Selection Algorithm consists of 3 Steps:
  # 1. Initializing the null model.
  # 2. For each fixed model size k, choose the best model using RSS.
  # 3. Choose the best k 
  #    -> using AIC, BIC, Cp or Adjusted R2, 
  #    -> or using the cross validation error.

# We use the function regsubsets() from the leaps package to perform Best Subset Selection.

# Note! 
    # Best Subsets Selection is a so-called "Wrapper Method" for Feature Selection. 
    # This means that it can be used for feature selection for virtually every regression or classification model.
    # In the lecture, we introduced it using the example of *linear regression* models.
    # The function regsubsets() is also only applicable for *linear regression* models.

# regsubsets() performs only steps 1+2 of the Best Subset Selection Algorithm:
  # For each fixed model size k, it chooses the best model based on RSS.
  # It does NOT select the best model size k for us, though! 
  # We need to do it manually after regsubsets().

##############   4.1. Using regsubsets() to find the best model for each model size k ############

# install.packages("leaps")
library("leaps")

# Apply regsubsets() to the Credit dataset.
# The syntax is the same as for lm().
sets <- regsubsets(Balance ~ ., Cr1, nvmax = 10) 


# Apply summary() to the output of regsubsets() to inspect the results:
summary(sets)
  # The row numbers indicate the model size (number of included predictors).
  # Remember that regsubsets() returns for each model size the best model based on RSS.
  # An asterisk in the summary indicates that a given variable is included in the model.
  # E.g., we see that the best model with 4 predictors includes the variables Income, Rating, Age and Student.

##############   4.2 Using AdjR2, AIC, Cp and BIC to find the best model size k ############

# We now want to select the best model size k for our data set. 
  # To do this, we can use one or all of the metrics AdjR2, AIC, Cp or BIC.
  # Conveniently, regsubsets() stors AdjR2, Cp and BIC in its summary. 
  # Note!
      # In this example, we use the error metrics that are stored in the summary of regsubsets to determin the best model.
      # Yet, these are training error metrics! Using training error metrics is not reliable, since models can overfit.
      # If we would want to have a more trustworthy comparison, we should *not* use the values stored 
      # the summary of regsubsets, but instead apply the validation set approach and calculate
      # AdjR2, Cp and BIC on the test set for each model.
      # The reason why we are *not* doing this here is (1.) to keep the example short and simple, and
      # (2.) because we will show how to use the cross-validation error as an alternative to AdjR2, Cp and BIC later on.
      # As you know, the CV error is even more reliable than the test error. 

# We first  inspect the output of regsubsets() a bit closer.
  # summary(sets) actually includes more information than it shows us right away.
  # We can use names() to get all the variables stored in it: 
sets_summary <- summary(sets)
names(sets_summary)  
  # We see that it indeed stores the values of the metrics R^2, RSS, AdjR2, Cp and BIC
  # for the best performing model per model size.
  # Let's view them all together
data.frame("rsq"=sets_summary$rsq, "rss"=sets_summary$rss, "adjr2"=sets_summary$adjr2, "cp"=sets_summary$cp, "bic"=sets_summary$bic)
  # Since regsubsets() target the models ordered by size, the row numbers indicate the model size.
  # We observe that 
  #     - R2 steadily increases with model size - as expected.
  #     - RSS steadily decreases with model size - as expected. 
              # Remember that both, R2 and RSS, are useless for determining the best model size.
              # Instead, we must use one of the other metrics.
  #     - AdjR2 (the higher the better) does not change much after model size 3.
  #     - CP (the lower the better) does not change much after model size 3.
  #     - BIC (the lower the better) does not change much after model size 3.

# We look up exactly which model size wins per metric:
(adjR2.max <- which.max(sets_summary$adjr2)) # the model with 8 predictors wins according to AdjR2
(cp.min <- which.min(sets_summary$cp)) # the model with 4 predictors wins according to CP
(bic.min <- which.min(sets_summary$bic)) # the model with 3 predictors wins according to BIC
    # which.max() and which.min() are functions in the base package that look up the highest and lowest value in a vector and return the corresponding row number.
    # Since our models are ordered by size, the row number is the model size.

# Unfortunately, the three metrics give us three different winners! The models with 8, 4 and 3 predictors. 
# Which model should we choose? Did we do something wrong?
# Maybe not: remember that there was not much change after model size 3. 

# We can double-check our models visually using plot(): 
plot(sets, scale="adjr2")
plot(sets, scale="Cp")
plot(sets, scale="bic")
  # Each table shows one row per model, and the models are sorted in descending order of performance. 
  # I.e., the best model is in the top row, the worst model is in the bottom row. 
  # We can easily see which predictors each model includes. Particularly, we see that:
      # The best Model according to BIC has 3 predictors: Income, Rating, Student
      # The best Model according to CP has 4 predictors: Income, Rating, Age, Student
      # The best Model according to AdjR2 has 8 predictors: Income, Rating, Cards, Age, Gender, Student, Married, Ethnicity
  # The plots also confirm visually that none of the error values do not change much from k=3 on for all metrics: All bigger models are are black in all the graphs.

# For interpretability reasons, less predictors would be better. So, we are inclined to choose the model with k=3.
# Before we decide, we fit them and look at their p-values:
lm.fit.3 <- lm(Balance ~ Income + Rating + Student, data=Cr1)
lm.fit.4 <- lm(Balance ~ Income + Rating + Age + Student, data=Cr1)
lm.fit.8 <- lm(Balance ~ Income + Rating + Cards + Age + Gender + Student + Married + Ethnicity, data=Cr1)
summary(lm.fit.3) 
summary(lm.fit.4) 
summary(lm.fit.8) 
# For the models with k=3 and k=4 all predictors are significant. Therefore we prefer to choose one of them.
# For interpretability reasons, I would choose the smallest model with k=3.

# Remark:
# Seeing that BIC (model size 3) and CP (model size 4) provide better results than AdjR2 is 
# not very surprising: AdjR2 only measures goodness of fit, which, on training data, bears the risk of overfitting.
# Indeed, AdjR2 chooses a highly flexible model. (More predictors bring more flexibility.)
# In contrast, BIC and CP try to balance both, the risk overfitting and the risk of underfitting. As a result, 
# both choose a less flexible model. 

# As an alternative to the plots above, we could plot a line graph for each of the 3 metrics (AdjR2, Cp, BIC) 
# as a function of the model size (number of predictors included), and in each graph we mark the winning model.

steps <- 1:10 
    # First we need to define the x-axes (that stands for the number of predictors). 
    # For this we construct the vector 1,...,10

p1 <- ggplot() + # set up ggplot
  geom_point(aes(x = steps, y = sets_summary$adjr2), color = "black") + # plot the points
  geom_line(aes(x = steps, y = sets_summary$adjr2), color = "black") + # connect them with lines
  geom_point(aes(x = adjR2.max, y = sets_summary$adjr2[adjR2.max]), color = "red", shape = 4, size=5) + # sets the cross for the winner
  xlab("Number of predictors") + ylab("Adjuster R squared") # names the axes
    # First we plot AdjR2.
    # Line 4 inserts a cross for the "winner":
    #   - Remember that adjR2.max holds the model size of the winning model - we defined it above.
    #   - sets_summary$adjr2[adjR2.max] gives us the corresponding R^2 value
    #   - shape=4 plots a cross instead of a point

p2 <- ggplot() +
  geom_point(aes(x = steps, y = sets_summary$cp), color = "blue") + 
  geom_line(aes(x = steps, y = sets_summary$cp), color = "blue") +
  geom_point(aes(x = cp.min, y = sets_summary$cp[cp.min]),color = "red", shape = 4, size = 5) +
  geom_point(aes(x = steps, y = sets_summary$bic), color = "red") +
  geom_line(aes(x = steps, y = sets_summary$bic), color = "red") +
  geom_point(aes(x = bic.min, y = sets_summary$bic[bic.min]), color = "black", shape = 4, size = 5) +
  xlab("Number of predictors") + ylab("Cp (blue) , BIC (red) ")
    # In the second plot we put Cp and BIC.

library(gridExtra) 
grid.arrange(p1, p2, nrow = 2) 
    # gridarrange() from the library gridExtra is convenient for arranging several plots in one window. 
    # nrow specifies the number of rows in the final plot window.

# Here we see also that not much is changing after k=3. Thus, if we would take only the error metrics into account, 
# it would not make a big difference which of the 3 best models we chose. 



############ 4.3 Using Cross Validation Error to find the Best Model Size k ############

# In the previous section, we used the metrics Adjusted R^2, Cp and BIC on the training set to find the best model size k.
# Now we use the cross validation error instead. 
# The cross validation error is a more trustworthy error measure, because it is a robust test error estimate.

# We proceed as follows:
  # We fit all of the models that regsubset() has given us using the glm() function,
  # We use glm() instead of lm(), because glm() gives us the function cv.glm() to apply cross validation easily.
glm1 <- glm(Balance ~ Rating, data=Cr1) 
glm2 <- glm(Balance ~ Income + Rating, data=Cr1) 
glm3 <- glm(Balance ~ Income + Rating + Student, data=Cr1) 
glm4 <- glm(Balance ~ Income + Rating + Age + Student, data=Cr1) 
glm5 <- glm(Balance ~ Income + Rating + Age + Student + Married, data = Cr1)
glm6 <- glm(Balance ~ Income + Rating + Age + Student + Married + Ethnicity, data = Cr1)
glm7 <- glm(Balance ~ Income + Rating + Cards + Age + Student + Married + Ethnicity, data=Cr1) 
glm8 <- glm(Balance ~ Income + Rating + Cards + Age + Gender + Student + Married + Ethnicity, data=Cr1)
glm10 <- glm(Balance ~ Income + Rating + Cards + Age + Education + Gender + Student + Married + Ethnicity, data=Cr1)
  # Notice that jump over the model with 9 predictors, because it only adds the second dummy variable for Ethnicity.
  # Ethnicity is split up in dummy variables by glm(), and we don't need to distinguish them in the model fit.
  # To see that, look again at summary(sets) and compare.

# Now we use cv.glm() to do 10-fold cross validation
library(boot) # load library boot to use cv.glm
set.seed(1)
cv.err.glm1 <- cv.glm(Cr1, glm1, K = 10) 
cv.err.glm2 <- cv.glm(Cr1, glm2, K = 10) 
cv.err.glm3 <- cv.glm(Cr1, glm3, K = 10) 
cv.err.glm4 <- cv.glm(Cr1, glm4, K = 10) 
cv.err.glm5 <- cv.glm(Cr1, glm5, K = 10)
cv.err.glm6 <- cv.glm(Cr1, glm6, K = 10) 
cv.err.glm7 <- cv.glm(Cr1, glm7, K = 10) 
cv.err.glm8 <- cv.glm(Cr1, glm8, K = 10)
cv.err.glm10 <- cv.glm(Cr1, glm10, K = 10)
# We put the cv-errors of all models in a data frame to look at them.
# Remember that the cross validation error is stored in the attribute delta[2]. 
(cv.error <- data.frame("cv-error" = c(cv.err.glm1$delta[2], cv.err.glm2$delta[2],
                                cv.err.glm3$delta[2], cv.err.glm4$delta[2], 
                                cv.err.glm5$delta[2],cv.err.glm6$delta[2],
                                cv.err.glm7$delta[2], cv.err.glm8$delta[2], 
                                cv.err.glm10$delta[2])))
# Find the minimum numerically:
(cv.min <- which.min(cv.error$cv.error)) # The winner is model 4.

# We can plot the CV error as a function of the number of predictors:
steps.cv <- 1:9 # define the x-axes (number of predictors - this time only 9!)
ggplot() + 
  geom_point(aes(x = steps.cv, y = cv.error$cv.error), color = "black") + 
  geom_line(aes(x = steps.cv, y = cv.error$cv.error), color = "black") + 
  geom_point(aes(x = cv.min, y = cv.error$cv.error[cv.min]), color = "red", shape = 4, size=5) + 
  xlab("Number of predictors") + ylab("cv error") 

# We inspect the model once again to remember:
summary(glm4)

# Yeah - we have a final winner!


############ 5. STEPWISE SELECTION ############

# Best Subset Selection gives us the security to find the best model.
# Yet, since it tries out all possible combination of predictors, it easily
# runs much too long.
# As an alternative, we can choose one of the following heuristics: 
#   - Stepwise forward selection
#   - Stepwise backward selection

# Stepwise forward selection 
sets_FWS <- regsubsets(Balance ~ ., Cr1, nvmax = 10, method = "forward") # set the parameter method="foreward" 
summary(sets_FWS)

# Stepwise backward selection
sets_BWS <- regsubsets(Balance ~ ., Cr1, nvmax = 10, method = "backward") # set the parameter method="foreward" 
summary(sets_BWS)

# In this example, both methods arrive at the same best model as Best Subset Selection. Yet, this is not always the case.




