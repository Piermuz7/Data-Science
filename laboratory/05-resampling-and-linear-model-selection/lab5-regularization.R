#####################################################
# Data Science
# Regularization
# Dr. Gwendolin Wilke
# gwendolin.wilke@fhnw.ch
#####################################################
# Contents:
# 1. Preliminaries
# 2. Data Exploration 
# 3. Manual Feature Selection
# 4. Data Preparation
# 5. Ridge Regression 
#     5.1 Applying Ridge Regression
#     5.2 Exploring the output
#     5.3 Using cross-validation to select the best lambda
#     5.4 Doing predictions
#     5.5 Comparing the Test MSE to OLS 
# 6. Lasso Regression
#####################################################
# We use the function glmnet() from library glmnet.
  # It fits a generalized linear model via penalized maximum likelihood. 
  # It can do both, ridge regression and LASSO regression:
  #   - For Ridge regression, set the parameter alpha=0.
  #   - For LASSO, set the parameter alpha=1. 
  #   - If we dont provide a parameter alpha, glmnet falls back to ordinary least squares.
# Remember that regularization needs standardized predictors. Conveniently, gmlnet standardizes the varibales for us.
# See help(glmnet) for more detail.
#####################################################




##############   1. Preliminaries   ##########################

library(ISLR)      # contains Credit data set
library("glmnet")  # install first if needed: install.packages("glmnet")

Cr1 <- Credit # make a copy of the original data set, so that we dont mess with it

##############   2. Data Exploration   ##########################

# See Model Selection Lab.

##############   3. Manual Feature Selection   ##########################

# Now let's kick out one of the highly collinear variables:
Cr1 <- select(Credit, -c(Limit))

# Additionally we can manually kick out ID 
Cr1 <- select(Cr1, -c(ID))

############ 4. Data Preparation ############

# glmnet() uses a specific input format - the predictors and the target must be provided separately:
#   - the predictors as a model matrix with the intercept removed, 
#   - the target variable as a vector.
# We need to transform our data set so that it fits this format.

predictors <- model.matrix(Balance ~ ., Cr1)[,-1]  # first column is the intercept, we remove it
target <- Cr1$Balance 

############ 5. Ridge Regression ############

##### 5.1 Applying Ridge Regression

m_ridge <- glmnet(predictors, target, alpha = 0) # calling ridge regression by using alpha=0

##### 5.2 Exploring the output

m_ridge
# Lambda ...value of lambda. 
# Df     ... degrees of freedom (number of nonzero coefficients)
# %Dev   ... percentage of (null) deviance explained 
#            %Dev is a measure that indicates how close the fitted model is to a perfect model. 
#            The perfect model hass %Dev = 1. (The perfect model is the model that correctly predicts all response values.)
#            The worst model has %Dev = 0. (The worst model is the "Null Model" that has no predictors, but uses only the intercept to predict.)
#
# How to interpret the values of Lambda and %Dev:
#   We observe that 100 different values of Lambda are used by default. 
#   For each of these Lambdas, a model is fit and %Dev is evaluated.
#   A bigger Lambda puts more penalty on large coefficients, thereby constraining the coefficients more (adding more bias).
#   As a result, the model gets further away from the perfect model. 
#   We see this by observing that bigger Lambdas result in smaller %Dev (further away from the perfect model).
#   The model with the highest Lambda has %Dev=0. This corresponds to the Null Model (worst model).
#   This is intended: The idea is to introduce bias to the model, with the hope to reduce variance.
#   According to the bias-variance tradeoff, the optimal value of Lambda will likely be somewhere in the middle. 
#   In order to find it, we need to do cross validation.
# 
# How to interpret the values of Df:
#   We observe that the value of Df does not change, but is constant at the value 10. 
#   I.e., each of the models has 10 non-zero coefficients, no matter how Lambda is chosen. 
#   Checking the number of predictors in our model matrix, we see that we originally had 10 predictors:
dim(predictors)[2]
#   This confirms that Ridge Regression never kicks out a predictor. 

# plotting the result:
plot(m_ridge, label=TRUE) 
# x-axes: "L1 Norm" is a metric for the aggregated size of the coefficients (L1-norm = sum of absolute values).
# y-axes: the coefficient values.
#     This is like reading our lambda-graphs from right to left: 
#     Large aggregated coefficients are now on the right of the picture instead of left.
# label=TRUE labels the coefficient curves with variable sequence numbers.
#     We observe that variable 7 (Student) dominates in all models.
#     We also observe that all coefficients shrink towards zero as the aggregated coefficients decrease.


##### 5.3 Using cross-validation to select the best lambda 

# Now let's do cross validation to find the optimal Lambda. 
# glmnet() comes with a method for CV, namely cv.glmnet(). Default is 10 folds. 
# Conveniently, it not only does CV for us, but directly returns the best Lambda.
(cv_ridge <- cv.glmnet(predictors, target, alpha = 0) )
# Lambda min: the value of lambda that gives minimum cv error
# Lambda 1se: largest value of lambda (most regularized model) such that the cv error is within 1 standard error of the minimum cv error.
# We only need Lambda min for the moment: It's value is 39.66.
# Checking again the output of glmnet(), we see that this is the smallest Lambda, with %Dev = 91.58%:
m_ridge
# This is not too surprising, since we have one dominating predictor (Student). 
# Constraining the aggregated coefficients comes down to constraining Student. 
# If we constrain our only predictor too much though, we quickly loose predictive power. 
# Also, we have enough data points to not introduce extra variance: n=400 and p=10, so n>>p.

# Let's look at the cv errors as a function of Lambda:
plot(cv_ridge)
# x-axes: the natural logarithm of Lambda. (As Lambda is between 40 and 396600, ln(Lambda) is between 3.6 and 13.)
# y-axes: cross-validated MSE
# error bars: upper and lower standard deviations
# 2 vertical dashed lines: "Lambda min" and "Lambda 1se"
# 
# We observe that increasing Lambda (constraining aggregated betas) immediately increases the cv error.
# Thus, regularization in this case does not help too much to reduce the variance, but only increases the bias.

# coefficients of the best model:
coef(cv_ridge, s = "lambda.min")
# Notice that cross validation does not give us a specific model, it only tells us which the best lambda is. 
# Therefore coef() retrieves the coefficient estimates from m_ridge, where all the 
# fitted models (for the different lambdas) are stored. 

##### 5.4 Doing predictions 

# We applied cross validation on the whole data set to find the best lambda. 
# Now we want to find a specific model for prediction, and want to be able to test it on new data. 
# Since we didnt hold out a test data set in the beginning, we need to do the fit again, 
# this time only to the training data. We use the best lambda from above.

# We split our original data set in training and test data
set.seed(1)
(train=sample(nrow(Cr1),nrow(Cr1)*0.8)) # indices of a training data (80%)
Cr1.Train <- Cr1[train,] # training data 
Cr1.Test <- Cr1[-train,] # test data

# Train ridge regression on training data
predictors.Train <- model.matrix(Balance ~ ., Cr1.Train)[,-1] # prepare format for glmnet
target.Train <- Cr1.Train$Balance  # prepare format for glmnet
m_ridge.Train <- glmnet(predictors.Train, target.Train, alpha = 0) # do the fit

# We can use the predict() function to make predictions with ridge regression
predictors.Test <- model.matrix(Balance ~ ., Cr1.Test)[,-1] # prepare format for glmnet
(ridge.pred <- predict(m_ridge.Train, newx = predictors.Test, s = cv_ridge$lambda.min)) # s specifies the lambda to use

##### 5.5 Comparing the Test MSE to OLS  

# Compute ridge MSE:
(MSE_ridge <- mean((Cr1.Test$Balance-ridge.pred)^2)) 

# Compare with MSE of OLS:
m_ols <- glm(Balance ~ ., data = Cr1.Train)
ols.pred <- predict(m_ols, Cr1.Test)
(MSE_ols <- mean((ols.pred - Cr1.Train$Balance)^2) ) 

# We observe that ridge regression performs better than OLS.
# This indicates that putting a penalty on large coefficients does at least have some effect, even in this case.


############ 6. Lasso Regression ############

m_LASSO <- glmnet(predictors, target, alpha = 1)

dim(coef(m_LASSO))
# We have only 69 rows, because glmnet has a stop criterion, see help.

m_LASSO  
# We observe that some of the coefficients are set to zero (Df=0, %Dev=0)

plot(m_LASSO, label=TRUE)
# Also here we can see that some of the coefficients are set to zero.


(cv_LASSO <- cv.glmnet(predictors, target, alpha = 1) )
plot(cv_LASSO)

(best_lambda_LASSO <- cv_LASSO$lambda.min) 
# best lamda is very small, but lambda.1se is considerably bigger
coef(m_LASSO, s=best_lambda_LASSO) 
# for small lambda almost all coefficients are included
coef(m_LASSO, s=cv_LASSO$lambda.1se) 
# We can try again with lambda.1se
coef(m_ridge, s=cv_ridge$lambda.min) 
# compare with ridge

# We can do predictions in the same way as we do with ridge.
# We prefer LASSO, because it not only decreases variance, but also does subset selection for us.

