#### Lab 3: Regression on Auto dataset ####
#
# TASK:
# Apply linear regression on the dataset "Auto" that comes with the ISLR library.
# Train different models that will predict mpg.
# Interpret and compare different models based on p-values, RSE and R^2.
# Select the best of these models.
#

install.packages("ISLR")
install.packages("tidyverse")
install.packages("Hmisc")
install.packages("corrplot")
install.packages("car")
install.packages("lmtest")

library(ISLR)
library(tidyverse)
library(Hmisc)
library(corrplot)
library(car)
library(lmtest)

#### Data Exploration ####

# View dataset
Auto
# View first few rows of the dataset
head(Auto)
# Or view it as a table
View(Auto)

### Understand the variables
# Since Auto is embedded in ISLR, a help page exists in R.
# It contains a data dictionary for interpretation of variable names.
help(Auto)
# structure gives us the data types of the attributes
str(Auto)
# We notice that "name" is categorical.
# Also, that "origin" is numerical, but should be categorical (see data dictionary).
# Convert "origin" to factor (i.e., to categorical variable), and store the result in a new variable.
Auto1 <- Auto
Auto1$origin <- as.factor(Auto1$origin)
str(Auto1) # it worked

### Explore a bit
# View statistical summary
summary(Auto1)
# Majority (62%) of cars is American (see "origin") - sthg to keep in mind for later, maybe we need it.
# How about car names? Is there one car that dominates the dataset?
max(table(Auto1$name)) # no (table() lists the frequencies of occurrence)
# Missing data checking
sum(is.na(Auto1)) # no missing values
# Outliers checking
hist.data.frame(Auto1) # no obvious outliers
# Anything else to see in the histograms?
# cylinders: dominated by 4,6,8 - 3 and 5 rarely occur
# mpg, displacement, horsepower, weight are right skewed, but that is not a problem for linear regression
# year is almost uniformly distributed


#### Investigate Collinearities ####

# Create a scatterplot matrix
plot(Auto1) 
# mpg seems to have a strong relationship with displacement, horsepower and weight.
# Thus, displacement, horsepower and weight are good candidates to be strong predictors for mpg. The relationships don't look linear, though.
# Yet, since we don't have another option a.t.m, we try linear regression anyways. Linear regression can also serve as a first benchmark for other approaches we may try later.
# We also notice that the input variables displacement, horsepower, weight and cylinders are strongly linearly correlated among each other.
# This is a problem, since collinear input variables mess up the p-values in a model, so that we cannot trust the evaluation.
# We therefore will have to remove all but one of the collinear input variables before training a model.
# mpg also seems to have some relationship with year and origin. origin does not count, though, since the order of its values is arbitrary.
# Only acceleration seems a bit random.
# We double check our first visual impressions by plotting a linear correlaton matrix:
corrplot(cor(Auto1[,-c(8,9)]), order = "hclust", tl.col = "black", tl.srt = 45, method="number")
# Notice also that cor() uses Pearson's correlation coefficient as a default. This means that it checks specifically for _linear_ correlation between variables.
# The correlation plot supports most of our impressions: displacement, horsepower, weight and cylinders correlate strongly with mpg. Only year and acceleration do not. (Notice that this also makes sense from a domain perspective.)
# We also confirm that the input variables displacement, horsepower, weight and cylinders are strongly linearly correlated among each other.
# We will take care of the collinearity problem later. Instead we train a model on ALL variable first, just to see what happens.
# Also, once we have a model matrix, we can apply the function vif() to check for multicollinearity (hidden collinearity).

### A tiny bit more of data exploration based on plot(Auto1) before we go on...
# Before we train the first model, we check if we notice any other patterns in the scatterplot matrix:
# origin: Japanese cars are best and American cars are worst in terms of efficiency. Also, Japanese cars are the lightest, have least horsepower and least displacement. Fits together...
# year: Newer cars tend to be lighter, and having less horsepower, displacement and cylinders.
# name: seems mostly random, as expected.


#### Fit a model and check for multicollinearity (hidden collinearity) ####

### 1st iteration: fit a regression model on ALL variables
attach(Auto1)
# Start by using all the predictors in the dataset - backward selection
lm.fit1 <- lm(mpg~., data=Auto1)
summary(lm.fit1)
# We see that the categorical variables "names" and "origin" are automatically on-hot-encoded by the lm() function.
# Since "name" has a lot of different values, it blows up our model massively.
# Based on the scatterplots, we assume that the car-names are not be related to mpg, so we will kick the variable "name" out in the next iteration.
# We also see that the variables that are marked highly significant (***) look kind of arbitrary.

### Check for multicollinearity (hidden collinearity that we don't see in a correlation plot)
# Now that we have a model, we can feed it in the function vif() from package car.
# vif() outputs a metric for each variable that indicates how much the variable is responsible for multicollinearity. vif=1 means no collinearity. vif>=5 is too much - we should kick this variable out.
vif(lm.fit1) # throws the error "there are aliased coefficients in the model". A variable is an "alias" of another variable if they are almost perfectly linearly correlated.
# we therefore check the correlation matrix using cor() that gives us Pearson's correlation index as a number.
# Notice that cor() only can handle numerical variables, so we have to remove the categorical variables beforehand.
cor(Auto1[,-c(8,9)]) 
# We see that "displacement" and "cylinders" are almost perfectly correlated, as well as "displacement" and "weight"
# We kick out displacement and hope that vif() will work then.


#### Variable selection ####

### 2nd iteration: fit a regression model on ALMOST ALL variables
# We kick out "name" and "displacement". To do this with the "select" function, we activate the dplyr library first.
# Fit a model without variables "name" and "displacement"
lm.fit2 <- lm(mpg~., data=select(Auto1,-c(displacement, name)))
summary(lm.fit2)
# Before we dive deeper into the model summars, we first check if we can trust the p-values of this model now:
vif(lm.fit2) # vif() works now. cylinders, horsepower and weight have vif values >5 - kick out two of them, before doing anything else!

## 3rd iteration: remove collinearities of input variables
# We decide to kick out cylinders and horsepower and keep weight, because weight has the strongest linear correlation with mpg among the three (cf. cor(Auto1[,-c(8,9)]))
# The decision is a bit arbitrary, though.
lm.fit3 <- lm(mpg~., data=select(Auto1,-c(cylinders, horsepower, displacement, name)))
# check vif now
vif(lm.fit3) # all good now
cor(Auto1[,-c(2,3,4,8,9)]) # good as well.
# now that we can trust the model output, we can take a closer look at it:
summary(lm.fit3)
# All variables but acceleration are significant, the f-statistics has a low p-value as well.
# Only acceleration is not significant - we kick it out in the next iteration.
# The RSE is around 3.3 miles per gallon - that is about 10% of the range of the mpg variable (which is 37 mpg, cf. summery(Auto1) ) - seems not too bad.
# The adjR2 is 82%, i.e. the variability of the input variables explain 83% of the variability of the output.
# weight has a slope of -5.8, meaning that a car that is 10 lbs heavier (about 4.5kg) covers about 6 miles (ca. 10km) less with one gallon of gas on average that the lighter car.
# year has a positive impact: the newer the car the more efficient. On average, one year accounts for 7 (ca. 10km) miles per gallon increase.
# To interpret origin, we need to look up how R encoded its values:
contrasts(origin) # intercept -> American, origin2 -> European, origin3 -> Japanese
# Not surprisingly, the model confirms that Japanese cars win in terms of efficiency on average.

# 4th iteration:
lm.fit4 <- lm(mpg~., data=select(Auto1,-c(cylinders, horsepower, acceleration, displacement, name)))
# double check vif to be sure
vif(lm.fit4) 
summary(lm.fit4)
# All p-values are good now.
# RSE and adjR2 are are almost the same than in model 3.
# We prefer the last model, since it includes less variables and all of them are significant.

### Calculate average model accuracy on the training data
# Let's compare the RMSEs of models 3 and model 4:
sqrt(mean(lm.fit3$residuals^2))
sqrt(mean(lm.fit4$residuals^2))
# Model 3 has a slightly better(smaller) RMSE than model 4:
# The average error made when we would predict the training data using model 3 is 3.312 miles per gallon.
# The average error made when we would predict the training data using model 4 is 3.316 miles per gallon.
# The difference of 0.004 mpg seems insignificant, though. E.g., it amounts to 12% of the RMSE and 0.02% of the average mpg in the dataset.

#### REMARK 1: Training Error vs. Test Error
# Above, we calculated the "Training RMSE", i.e., the RMSE on the same sample that we used to train (fit) our model.
# The problem with this approach is that a model can "overfit" the training data: A model that overfits will have a very low Training RMSE, but does not generalize well to new data.
# Consequently, Training RMSE is not sufficient to compare model accuracy.
# Instead, we must calculate the "Test-RMSE": the Test-RMSE is the RMSE of a "Test Set" of observations that was held back from training. (Validation-Set Approach)

#### REMARK 2: RMSE vs. Prediction Interval
# The RMSE captures the _average_ error of observations.
# In contrast, the prediction interval captures the error of a _single_ observation.

####Checking the OLS assumptions ####

## Check Residual plots
# The residual plots are exploratory tools. They allow us to visually check whether the preconditions for OLS of 1. "linearity, 2. "strict exogeneity" 3. "homoscedasticity", 4. "normality of error terms" are met.
# If linearity and/or strict exogeneity is violated, we might get a systematic error (bias), and we might be better off trying a non-linear model. If "homoscedasticity" and "normality of error terms" are violated, our p-values may be not trustworthy.
# The assumptions 3 and 4 are used to estimate the Standard Error, and thus the p-values, which are calculated from it. If the assumptions are violated, the p-value estimates may be wrong.
# They residual plots can also help us to identify leverage points (observations with unusually high input variable values) and potential outliers (observations with unusually high output variable values).
plot(lm.fit4)
# 1. Fitted vs. residuals:
# Used to check linearity, strict exogeneity (mean of error terms is zero), homoscedasticity (constant variance of error terms), and to identify potential outliers.
# If there was no random and no systematic error, all the observation would lie exactly on the regression line (hyperplane). In this case all of the dots on this plot would lie on the gray horizontal dashed line through zero.
# The red line is a smoothed curve that passes through the residuals as a visual aid to identify deviations from the grey dashed line.
# For strict exogeneity, the mean of the error terms need to be zero. So the red line should ideally fall directly on the grey dashed line.
# For homoscedasticity, the variation in the error terms should not change with the response. So the spread around the red line shouldn't vary with the fitted values.
# In our model, we see a strong pattern of variation with the fitted values, and the red line looks quadratic, meaning that a linear hyperplane does not fit the data well.
# Maybe a nonlinear model would fit the data better - we must postpone that to later, though.
# The pattern also indicates that the size of error terms change with the y-values, i.e., their variability is not constant (heteroscedasticity).
# The plot also marks the three observations with the highest error terms: 323, 326, 327. Since there are always 3 points numbered, this doesn’t necessarily mean that these observations are outliers, but we may want to take a closer look at them later.
# Notice that the numbered points will be the same in all of the following plots
select(Auto1,-c(cylinders, horsepower, acceleration, displacement, name))[c(321,324, 325),]
# 2. Normal Q-Q-plot:
# Used to check normality and potential outliers.
# The QQ-plot plots the quantiles of a normal distribution ("theoretical quantiles") against the quantiles of the standardized residuals. If it's a perfect 45-degree line, your residuals are normal.
# Some deviation is to be expected, particularly near the extremes, but the deviations should be small.
# We observe that the residuals deviate a lot from the line in the upper right, hinting to non-normality. Yet, this is only a first visual indication, and we must confirm using a statistical test, e.g., the Shapiro-Wilks-test:
shapiro.test(residuals(lm.fit4)) # Confirmed: The W-statistics has a low p-value. That this is bad, since the null hypothesis of the test is "The data is normally distributed". The test in this case rejects the Null-hypothesis, meaning we don't have normal residuals.
# The QQ-plot can also give us an indication of outliers. An outlier has an unusually high error term, like, e.g. 323 in our plot.
# 3. Scale-location plot:
# Used to double-check homoscedasticity and to identify potential outliers.
# The scale-location plot is very similar to the residuals vs fitted plot, but often simplifies analysis of the homoscedasticity assumption.
# In contrast to residual vs. fitted, the y-axis does not show the residuals directly, but shows the square root of the absolute value of the standardized residuals. The standardized residuals are residuals rescaled so that they have a mean of zero and a variance of one. Standardizing helps us assess if an error is "too high". The absolute value makes all values positive so that large residuals (both positive and negative!) are plotting at the top and small residuals plotting at the bottom. The square root is only there to dampen high values so that everything is better visible in one picture. E.g., a y-value of 2 in the scale location plot means that the observation's error is 4 standard deviations from the mean - which is a lot.
# Again, for homoscedasticity, the red line should be horizontal, since it indicates the average magnitude of error terms. Also, the spread around the red line shouldn't vary with the fitted values.
# For our model, we see that the red line is not strictly horizontal. Notice that the non-linear shape of the red line is not so prominent as it is in the 1st plot (residual vs. fitted). This is due to the square root, which dampens high values. (Also, notice that the shape is different, because we look at absolute values here.)
# The variability of errors increases a bit with the response, which again shows as a slight funnel shape. This fits with our observations from the residual vs. fitted plot, and leads us to suspected heteroscedasticity, but it is not super clearly discernible here.
# We test our suspicion about heteroscedasticity with a statistical test, e.g., the Breusch-Pagan-test:
bptest(lm.fit4) # Confirmed. The Null Hypothesis of the bptest is "The residuals are homoscedastic." Our p-value is smaller than 1%, which rejects the Null Hypothesis, confirming our conjecture of heteroscedasticity.
# Again, we see that the numbered points have exceptionally high residuals.
# 4. Residuals vs. Leverage plot:
# Used to identify outliers and leverage points.
# The x-axis shows a metric for the "leverage" of each observation, and the y-axis shows the standardized residual of that observation. Leverage is a measure of how much each data point influences the regression line (hyperplane).
# The regression line (hyperplane) always passes through the centroid of the data set. Because of that, points that lie far from the centroid have greater influence on the regression line (leverage): they easily change the "direction" ("orientation") of the line (hyperplane). Also, the leverage of a point increases if there are fewer points nearby. As a result, leverage reflects both the distance from the centroid and the isolation of a point.
# The plot also gives contour lines for Cook’s distance, which measures how much the regression line (hyperplane) would change if a point was deleted.
# Ideally, the red smoothed line stays close to the horizontal gray dashed line and no points have a Cook’s distance larger than 0.5.
# For our model, the red line deviates from the horizontal line in the extremes, but the contour line for Cook's distance = 5 does not even show up in the plot.
# We can also plot Cook's distance per observation:
plot(lm.fit4, which=4)
# We see that the points with the highest error also have highest Cook's distance.
# Plotting studentized residuals for outlier detection
# The plots generated help us identify potential outliers by pointing us to observations with unusually high errors.
# Outliers are defined as observations with unusually high response values - but non of the above plots shows the observed response (only the fitted response).
# Plotting the "studentized residuals" makes our job easier: As a rule of thumb, every observation with a studentized residual > 3 can be safely considered an outlier.
# When trying to identify outliers, one problem that can arise is when there is a potential outlier that influences the regression model to such an extent that the estimated regression function is "pulled" towards the potential outlier. As a result, it isn't visible as an outlier using the standardized residuals. To address this issue, studentized residuals offer an alternative criterion for identifying outliers. The basic idea is to delete the observations one at a time, each time refitting the regression model on the remaining n–1 observations. Then, we compare the observed response values to their fitted values based on the models with the ith observation deleted. This produces deleted residuals. Standardizing the deleted residuals produces studentized residuals. Studentized residuals are distributed according to t distribution and the probability of being greater than 3 is less than 1%.
plot(rstudent(lm.fit4))
text(x=1:length(rstudent(lm.fit3)), y=rstudent(lm.fit3), labels=ifelse(rstudent(lm.fit3) > 3,names(rstudent(lm.fit3)),""), col="red", pos = 4)
abline(3,0, col = "red")
abline(-3,0, col = "red")
# We find 6 outliers in our dataset: observations nr. 38, 245, 323, 326, 327, 330.
# To double-check, we confirm the outliers using the outlierTest function.
# It gives us the studentized residuals of potential outliers, together with a significance value.
# For significance we need to look at the Bonferroni p-value instead of the p-value (It corrects the p-value for multiple tests.)
outlierTest(lm.fit4)
# We get a significant result for observation 323.

##### Put it all together:
# The 1st plot (residual vs fitted) is the most useful here, as it shows that the residuals are not centered around the line. The distinct U-shape indicates that a quadratic model might be a better fit to this data set.
# The 1st plot also shows that the variance of the error terms increases with the response, as we see a slight funnel shape around the red line.
# This effect hints to the suspicion that not all of the heteroscedasticity comes from non-linearity. I.e., a quadratic model may work better, but may still leave us with a bit of heteroscedasticity.
# In summary, our linear regression model should not be used, because at least 3 assumptions of OLS are violated (linearity, homoscedasticity and nomrality of error terms).

##### How to fix the problems?
# Non-linearity:
# There are 2 possible ways to go:
# 1. We can try to fit a non-linear model. Since the errors in the 1st plot (residual vs fitted) look quadratic, we may wanna play around with quadratic regression.
# 2. We could also try a non-linear transformation of the response variable and hope that we can fit a linear hyperplane afterwards. Yet, with this approach we cannot fine-tune the individual input-variables. It is less flexible than the first approach, and, as a consequence, it is more likely to fail.
# Heteroscedasticity:
# We can expect that we can get rid of the most part of heteroscedasticity by fitting a quadratic model.
# The result may leave us with a bit of heteroscedasticity. Maybe it is small enough to ignore it.
# Or, maybe it will be possible to fix it with a transformation of the response variable.
# Or, we apply an entirely different approach to regression analysis that is more flexible to fit complex, non-linear structures, such as, e.g., a tree-based approach (see later).
# Non-normality:
# Non-normality is also a direct consequence of non-linearity.
# Outliers:
# We don't look at the potential outliers in this case, since we need to change the model anyways.
# The reason is that the outlier tests we perfomed above always depend on the model that we fit!
# Fitting another model may result in different points being flagged as an outlier.
# E.g., observation 323 has an unusually high studentized residual. This means that it would change the regression line produced by our model lm.fit4 strongly if it was deleted from the dataset.
# Keep in mind: You should be always be very wary of changing your data!
# Outlier detection is meant to flag observations that MIGHT be erroneous, with the purpose that you investigate them more closely.
# It is not a license to change the data! An outlier could very well be a legitimate observation.