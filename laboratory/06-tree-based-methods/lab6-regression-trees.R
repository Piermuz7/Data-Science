#####################################################
# Applied Machine Learning
# DS-06 Tree-based Methods
# Lab 1: Regression Trees
# Gwendolin Wilke
# gwendolin.wilke@hslu.ch
#####################################################
# 1. Preliminaries
# 2. Data Exploration
# 3. Validation Set Approach
# 4. Fitting a Regression Tree (unpruned)
#     4.1 Two Predictors
#     4.2 All Predictors
# 5. Fitting a Pruned Tree
# 6. Bagging (Bootstrap Aggregation)
# 7. Random Forests
#####################################################



##############   1. Preliminaries   ##########################

# install.packages("tree")
library(ISLR)
library(ggplot2)

Ht <- Hitters # make a copy of the original data set, so that we dont mess with it

##############   2. Data Exploration   ##########################

## Read the data dictionary 
?Hitters 
# For accessing the help page, we need to use the original name of the data set.
# In this exercise, we will try to predict the credit card balance from the other variables (regression task).

## Examine Data Set
View(Ht)
str(Ht)
summary(Ht)
# Notice that summary also shows us how many missing values (NAs) are present per attribute.
# Here, the only incomplete attribute is "Salary" with 59 NAs.

## Visual exploration
library(Hmisc)
hist(Ht)
# We see that a lot of the variables are skewed, in particular also the ouput variable Salary.
# Tree-based models don't make any assumptions on the distribution of the variabels, and also not in the distribution of the errors.
# So, as long as we stick to our regression tree model, we do not care about the skewness that is present here.
pairs(Ht) 
# Notice that the pairs plot is almost at it's limit here. If we had even more variables, we couldn't use it any more.
corrplot(cor(select(Ht, -c(League, Division, NewLeague))))
# Notice that we don't need to worry about the cluster of multicollinearities we see in the input variables, since we use a regression tree: it can learn linear as well as nonlinear surfaces and is not impacted by collinear inputs.
#     We check corrplot only as part of data exploration to get a better idea about the structure of our data, i.e., to understand it better.
#     To do that, ask yourself: Regarding attribute semantics, are you surprised about the collinearity of some of the input variables, or does it confirm your understanding of attribute semantics? 
#     Data exploration is a great tool to double check our data understanding! 
#     Another reason to check corrplot here is that we might want to try linear regression on the data set later on to compare both approaches - corrplot gives us an idea, if trying out linear regression makes sense or not.
# Notice also that the Salary column shows a question mark. 
#     The reason is that we have missing values in this column. 
#     When you have an NA present in a formula (here: cor() ), the output of the formula is usually again NA (here symbolized as ?).
#     This is true for almost all machine learning algorithms also. Yet, the function tree() that we will use below is an exception: it can handle missing values.
#     We remove the NAs anyways. Let's do that now.

# Missing values 
sum(is.na(Ht)) # 59
sum(is.na(Ht$Salary)) # also 59. Thus Salary is the only column with NAs present.
Ht_nonas <- na.omit(Ht) 
# The simplest possibility to get rid of NAs is to use na.omit(). It removes all incomplete observations from the data set (i.e., all rows that have an NA in at least one of the columns).
# 
# Notice that applying na.omit() on the whole data set is usually not a sensible choice! It makes sense ONLY if one of 2 cases apply:
#
#        Case 1: The number of incomplete rows is very small. (This is not the case here: we have 18% incomplete rows.)
#        Case 2: The incomplete values only occur in the output variable. (This applies in the case of the Ht_nonas data!)
# 
#        Ad 1: When the number of incomplete rows is high, na.omit() removes a lot of data. We might be left with very little data to learn from. 
#              Instead of using na.omit(), it would be better to
#                -  find the sparse rows and columns (with over 70% NAs) and only remove those, then
#                -  impute the remaining NAs using some imputation method (simplest choice: mean or mode of the columns).
#        Ad 2: When only the output variable is incomplete, "imputing" it actually means to predict it (predict the missing values). 
#               It would not make sense to "guess" the missing output variables (e.g., by calculating the mean)
#               and then use these "guesses" as example data to train our decision tree...
#               For this reason we apply na.omit() to the Ht_nonas data set.

# Now, we can run corrplot(cor()) again:
corrplot(cor(select(Ht_nonas, -c(League, Division, NewLeague))))
# We see that not many input variables show a very high linear correlation with Salary. 
# Thus, trying out a linear regression model as an alternative to regression trees might not be worth our while...
# We also see that when going back to the pairs plot to check the Salary column there.

##############   3. Validation Set Approach ##########################

# sample 70% of the row indices for subsetting as training data
set.seed (1)
train_indx <- sample(1:nrow(Ht_nonas), 0.7*nrow(Ht_nonas))
Ht_nonas.train <- Ht_nonas[train_indx,]
Ht_nonas.test <- Ht_nonas[-train_indx,]

##############   4. Tree  ##########################
#
# We fit an (unpruned) regression tree.

# We use the function tree() from library tree.
library(tree)

##############   4.1 Two Predictors   ##########################
# 
#  We use only 2 input variables at first, so that we can visualize what's going on.

tree.HitsYears <- tree(Salary ~ Years + Hits, data = Ht_nonas.train)
#   By default, tree() uses binary recursive splitting based on RSS.
#   As a stop criterion it uses that eigther a default minimum number of observations in a node is reached, or a maximum RSS value.
#   Notice for bigger data sets: The function tree() restricts the tree depth to 31 for technical reasons (see help(tree)). An alternative to tree() is rpart().

summary(tree.HitsYears)
#   There are 4 terminal nodes in the final tree.
#   For regression trees, the “residual deviance” is just the residual sum of squares (RSS), summed over all nodes.
#   The "mean residual deviance" is the RSS divided by the number of degrees of freedom (180 = number of observations (184) minus number of leaf nodes (4)).

# Plot the tree 
plot(tree.HitsYears)
text(tree.HitsYears, cex=0.75) # cex: set character size to 0.75
#   Notice that the tree uses the attribute "Years" twice for splitting.
#   Using an attribute more than once is possible in binary recursive splitting.
#   Remember that, in a regression trees, the order of the attributes is a measure of variable importance.
#   Thus we see that the number of Years a player spent in a major leage is the most important predictor of salary.
#   Classification and Regression Trees (CART) are called "explainable models" or "white box models". 
#   The reason is that you can directly read off the reasons (criteria) for each prediction. 
#   E.g., if our tree predicts a salary of 886k for a certain player, we can explain the reason: 
#       "The reason is that the player has spent more than 4.5 years in major leagues and achieved more than 117 Hits this year. 
#        Yet, he would be expected to earn even more, if he had spent more than 12.5 years in major leagues - which does not apply in his case."

# Compare the plot with the textual description:
#   The textual output gives you the details about number of observations and deviance per node.
tree.HitsYears
#   The output of tree() is an object of class "tree":
#     node): node number
#     split: split criterion, e.g. League: A, or Years < 4.5
#     n: number of observations in that branch
#     dev: deviance (RSS) of the node (the smaller the better)
#     yval: prediction for the branch (mean value of all observations in this node)
#     * indicates a terminal node

# Plot the regions
plot(Ht_nonas.train$Years, Ht_nonas.train$Hits, col='steelblue', pch=20, xlab="Years",ylab="Hits")
partition.tree(tree.HitsYears, ordvars=c("Years","Hits"),add=TRUE,cex=1)
#   Since our input space is only 2D, we can plot the regions that correspond to the leaf nodes
#   Here, we see again that "Years" is used twice for splitting.

# Training error (RMSE) 
(tree.HitsYears.train.RMSE <- sqrt(summary(tree.HitsYears)$dev/summary(tree.HitsYears)$df)) 
#     The exact value of the deviance (RSS) is stored in summary(tree.HitsYears)$dev.
#     The degrees of freedom are stored in summary(tree.HitsYears)$df.
#   Training error: 331 - when compared to the mean of Salary (535), it seems very high - especially for a training error.

# Check if this is plausible by visually comparing the predicted Salaries to the true Salaries:
salaryHt_nonas.train <- Ht_nonas.train$Salary # True Salaries on training data
tree.HitsYears.train.predict <- predict(tree.HitsYears, Ht_nonas.train) # "Predicted" Salaries on on training data
plot(salaryHt_nonas.train, tree.HitsYears.train.predict)
abline (0 ,1) 
#   abline (0 ,1) draws the function f(x) = x (intercept 0, slope 1).
#   For a perfect prediction, all points would lie in this line. 
#   We see that the predicted values indeed deviate a lot from the true values. Thus, the high training MSE is plausible.
#   The horizontal pattern stems from the 4 regions: each region provides a prediction for all included observations.

# Test error (RMSE) 
tree.HitsYears.predict <- predict(tree.HitsYears, Ht_nonas.test) # To calculate it, we first need to predict the test data: As always, use the the generic function predict().
(tree.HitsYears.test.dev <- sum((tree.HitsYears.predict - Ht_nonas.test$Salary)^2)) # Calculate the deviance (RSS) 
(tree.HitsYears.test.RMSE <- sqrt(tree.HitsYears.test.dev/summary(tree.HitsYears)$df)) # Calculate the RMSE (divide by degrees of freedom and take the square root).
#   Test error: 239

# Use tree.control() to grow the tree deeper:
tree.HitsYears.control = tree.control(nobs = dim(Ht_nonas.train)[1], mincut=1, minsize = 3, mindev = 0)
#   tree.control() produces control parameters for tree(). 
#     nobs: The number of observations in the training set.
#     mincut: The minimum number of observations to include in a child node. Default: 5.
#     minsize: The smallest allowed node size. Default: 10.
#     mindev: The within-node deviance (RSS) must be at least this times that of the root node for the node to be split. Default: 0.01.
#     To produce a tree that fits the data perfectly (saturated tree), set mindev = 0 and minsize = 2 (provided the limit of 31 on tree depth allows such a tree).

tree.HitsYearsCon <- tree(Salary ~ Years + Hits, data = Ht_nonas.train, control=tree.HitsYears.control) 
#   grow the tree with the control parameter set.
summary(tree.HitsYearsCon)
#   Notice that we have 125 nodes for 184 observations - that is not so far from the "perfect" fit ("saturated model").
#   That's probably not what we want, since it is likely overfitting.

# Training error: Calculate the RMSE by taking the root of the mean deviance
(tree.HitsYearsCon.train.RMSE <- sqrt(summary(tree.HitsYearsCon)$dev/summary(tree.HitsYearsCon)$df))
#   Training error: 97 - it is much smaller now, as expected.

# Let's confirm visually:
salaryHt_nonas.train <- Ht_nonas.train$Salary # True Salaries on training data
tree.HitsYearsCon.train.predict <- predict(tree.HitsYearsCon, Ht_nonas.train) # "Predicted" Salaries on on training data
plot(salaryHt_nonas.train, tree.HitsYearsCon.train.predict)
abline (0 ,1) 
#   As expected, the training set is predicted very well.
# To see whether we are overfitting, we compare with the test error.

# Test error (RMSE)  
tree.HitsYearsCon.predict <- predict(tree.HitsYearsCon, Ht_nonas.test)
tree.HitsYearsCon.predict.dev <- sum((tree.HitsYearsCon.predict - Ht_nonas.test$Salary)^2)
(tree.HitsYearsCon.test.RMSE <- sqrt(tree.HitsYearsCon.predict.dev/summary(tree.HitsYearsCon)$df))
#   Test error: 628 - terribly high. We are definitely overfitting.

# Now compare the different errors:
HitsYears      <- c(tree.HitsYears.train.RMSE, tree.HitsYears.test.RMSE)
HitsYearsCon   <- c(tree.HitsYearsCon.train.RMSE, tree.HitsYearsCon.test.RMSE)
error_matrix <- data.frame(HitsYears, HitsYearsCon)
row.names(error_matrix) <- c("train", "test")
error_matrix

# Remark 1:
#   In order to find the sweet spot between over- and underfitting, we could apply cross validation for identifying the 
#   best parameter values for tree.control(): mincut, minsize and mindev.
#   Yet, cost-complexity pruning may give us better results than global thresholds do. 
#   So that's what we will do instead. (See section 5 below.)

# Remark 2:
#   Our test and training results are quite bad.
#   One reason is that we didn't fine-tune our tree yet (using cross-validation).
#   Another reason is probably that we only used 2 predictors to train our model!
#   So, before we go to pruning, let's first fit the full model...


##############   4.2 All Predictors  ##########################
#
# We use all predictors now to fit the tree.

# Fit the tree
tree.all.control = tree.control(nobs = dim(Ht_nonas.train)[1], mincut=5, minsize = 10, mindev = 0.01)
#   The choice of parameters here is kind of arbitrary: 
#   We want a deeper tree than the default tree (which has mincut = 5, minsize = 10, mindev = 0.01),
#   but not as deep as the full tree            (which has mincut = 1, minsize =  2, mindev = 0).
tree.all <- tree(Salary ~ ., data = Ht_nonas.train, control=tree.all.control) 
summary(tree.all)
#   Notice that summary() tells us now which of the attributes where actually used in tree construction.
#   Only 9 out of our 19 attributes where used!
#   Now the most important criterion to predict Salary is CHits (= overall number of Hits in a player's career). 
#   Notice that "Years" does not even show up and "Hits" (= Hits this year) is quite unimportant now (deeper down in the tree: see plot below). 

# Plot the tree 
plot(tree.all)
text(tree.all, cex=0.75) # cex: sets the character size to 0.75

# Training error (RMSE)
(tree.all.train.RMSE <- sqrt(summary(tree.all)$dev/summary(tree.all)$df))
#   233 

# Test error (RMSE)
tree.all.predict <- predict(tree.all, Ht_nonas.test)
tree.all.predict.dev <- sum((tree.all.predict - Ht_nonas.test$Salary)^2)
(tree.all.test.RMSE <- sqrt(tree.all.predict.dev/summary(tree.all)$df)) 
#   235 

# We could play with the tree.control() parameters a bit, or do cross-validation now.
# Instead, we look at cost-complexity pruning. (We will need cross-validation there aslo, to find the best pruning parameter alpha)


##############   5. Pruned Tree   ##########################
# 
# Cost-Complexity Pruning
#   Goal: Prune the tree to avoid high variance and overfitting. 
#   Expected positive effects: 
#     - smaller test errors (due to less overfitting).
#     - higher interpretability (due to smaller trees).
#   Approach: 
#     - Do cross-validation on the training+test set to find the best pruning parameter alpha.
#     - Using the best alpha, grow a pruned tree on the training set.
#     - Evaluate it on the validation set and compare the result with the validation set error from the tree obove.


# We grow the full tree now
tree.full.control <- tree.control(nobs = dim(Ht_nonas.train)[1], mincut=1, minsize = 2, mindev = 0)
# This set of parameters produce (almost) the full tree ("saturated tree", the tree that fits the data perfectly).
#   Notice:
#     In the tree above (in 4.2), we set the tree.control() parameters to mincut = 2, minsize = 10, mindev = 0.01.
#     These parameters serve as a global threshold for growth, i.e, they stop the growth at a certain point, and the stop critera apply equally to all branches of the tree.
#     In contrast, what we do now is cost complexity pruning: It takes the fully grown tree (no stop critera) and afterwards cuts back each branch.
#     The cutting is *not* the same for all branches, but is done individually - in such a way as to minimize the overall RSS.
tree.full <- tree(Salary ~ ., Ht_nonas.train, control = tree.full.control) 
summary(tree.full)

# use cross-validation to find the optimal parameter \alpha for cost-complexity pruning  
set.seed (1)
cv.tree.full = cv.tree(tree.full)
#   Runs a k-fold cross-validation experiment to determin deviance (RSS) 
#   as a function of the cost-complexity parameter alpha.

cv.tree.full
#   $size: number of terminal nodes of each tree
#     Notice that the size is decreasing (corresponding to the pruning sequence).
#   $dev: cross-validation deviance (RSS)
#   $k: cost-complexity parameter (alpha)
#     Notice that alpha is increasing (corresponding to the pruning sequence).

# Plot the cross-validation deviance as a function of size and alpha
par(mfrow=c(1,2)) # Environment variable to arrange multiple plots in one window: c(1,2)... 1 row, 2 columns 
  plot(cv.tree.full$size, cv.tree.full$dev, type="b", xlab="number of terminal nodes", ylab="deviance") # type="b": plot both, points and lines
  plot(cv.tree.full$k, cv.tree.full$dev, type="b", xlab="alpha", ylab="deviance")
par(mfrow=c(1,1)) # Set back to default.

# Find the tree with smallest CV error
mindev.idx <- which(cv.tree.full$dev == min(cv.tree.full$dev)) 
#   Index with minimal deviance
(best.size <- min(cv.tree.full$size[mindev.idx]))
#   The tree with 8 terminal nodes has lowest cross-validation error.

# Now fit the pruned tree with alpha = 8
tree.pruned <- prune.tree(tree.full, best = best.size)
# prune.tree determines the nested cost-complexity sequence  
# best=8: get the 8-node tree in the cost-complexity sequence 

summary(tree.pruned)

# Plot the pruned regression tree 
plot(tree.pruned) 
text(tree.pruned, cex=0.75) # cex: set character size to 0.75

# Training error (RMSE)
(tree.pruned.train.RMSE <- sqrt(summary(tree.pruned)$dev/summary(tree.pruned)$df))
#   214 (as compared to 202 in the from 4.2)

# Test error (RMSE)
tree.pruned.predict <- predict(tree.pruned, Ht_nonas.test)
tree.pruned.predict.dev <- sum((tree.pruned.predict - Ht_nonas.test$Salary)^2)
(tree.pruned.test.RMSE <- sqrt(tree.pruned.predict.dev/summary(tree.pruned)$df))
# 299 (as compared to 200 in the tree from 4.2)


##############   6. Bagging (Bootstrap Aggregation)  ##########################

library(randomForest)
#   Recall that bagging is just a special case of random forests with m = p.
#   Thus, we can use the function randomForest() from the library randomForest for bagging also.

# Apply bagging  
set.seed(1)
(tree.bag <- randomForest(Salary ~ ., Ht_nonas.train, mtry=19, importance =TRUE, ntree=500))
#   mtry = 19 means that we use all 19 predictors for each split of the tree - hence, do bagging.
#   importance = TRUE says that variable importance should be assessed.
#   ntree	... Number of trees to grow. This should not be set to too small a number, to ensure that every 
#             input data point gets predicted at least a few times. 

summary(tree.bag)
  # Lists the attributes of bag.all

# Training error (RMSE) based on predictions 
tree.bag.predict.train <- predict(tree.bag, Ht_nonas.train) 
  # We predict the data points of the training set using the ensemble.
  # Remember that these values are the *averaged* predictions of all trees in the ensemble. 
  # Each of the predicted data points was part of the training set of some of the trees.  
  # Thus, the error calculated from these predictions indeed corresponds to a classical "training error".
(tree.bag.train.MSE <- mean((tree.bag.predict.train - Ht_nonas.train$Salary)^2))
(tree.bag.train.RMSE <- sqrt(tree.bag.train.MSE))
#   113 - The training error is relatively small. Likely the ensemble is overfitting.

# OOB error (RMSE)
#   Recall that we actuaklly don't need to do cross validation for bagged trees to get a robust test error estimate.
#   The OOB error is the qeuivalent of the cross validation error.
tree.bag$predicted
#   bag.all$predicted holds the averaged predicted values of the input data based on out-of-bag samples. 
#   I.e., only those trees are participating in the prediction for which the data point in question was NOT part of the training set.
#   Thus, the error calculated from these predictions corresponds to a "CV test error".
(tree.bag.OOB.MSE <- mean((tree.bag$predicted - Ht_nonas.train$Salary)^2))
(tree.bag.OOB.RMSE <- sqrt(tree.bag.OOB.MSE))
#   286 - The OOB error is much higher, as expected.

# Test error (RMSE)
tree.bag.predict.test <- predict(tree.bag, Ht_nonas.test)
(tree.bag.test.MSE <- mean((tree.bag.predict.test - Ht_nonas.test$Salary)^2))
(tree.bag.test.RMSE <- sqrt(tree.bag.test.MSE))
#   321 - The test error is even higher.

# Variable Importance 
importance(tree.bag)
# Two measures are reported:
#   1. %IncMSE        ... reports how much the average MSE (estimated with out-of-bag-CV) increase
#                         over all trees when we randomly shuffle the values of this variable.
#                         The higher %IncMSE, the more important the variable.
#                         %IncMSE is a robust measure.
#   2. IncNodePurity  ... reports the total increase in node impurity that results from splits 
#                         over this variable, averaged over all trees.
#                         It is measured using the loss function by which best splits are chosen, i.e., the MSE in regression trees.
#                         The higher IncNodePurity, the more important the variable.
#                         IncNodePurity is biased, %IncMSE is preferred.
sort(importance(tree.bag)[,1], decreasing = T)
#   CHits has highest %IncMSE, followed by CAtBat. This is consistent with the decision trees we grew in sections 4 and 5.
(varImpPlot(tree.bag))
#   Plot variables importance measures

# Optimizing the ntree parameter
plot(tree.bag)
# Shows the OOB error convergence with growing number of trees.
# Remember that Bagging does not overfit, so we can increase ntree as we like.
# If you increase ntree, e.g. to 1000, you will see in the plot that the OOB 
#   error stabilizes after 500 trees, so we dont need more than that.

# Optimizing the mtry parameter
#   The default for mtry is quite sensible so there is not really a need to change it. 
#   To see if randomForest is a better option than bagging, just compare the OOB errors of both.
#   There is a function tuneRF() for optimizing the mtry parameter, yet it may cause bias. 


##############   7. Random Forests   ##########################

# Growing a random forest proceeds in exactly the same way, 
#     except that we use a smaller value of the mtry argument. 
# By default, randomForest() uses 
#     - p/3 variables when building a random forest of regression trees, and 
#     - sqrt(p) variables when building a random forest of classification trees.

# Building a random forest on the same data set using mtry = 19/3 = 6. 
set.seed(1) 
(tree.rf <- randomForest(Salary ~ ., Ht_nonas.train, mtry = 6, importance =TRUE, ntree=500))

# Training error (RMSE)
tree.rf.predict.train <- predict(tree.rf, Ht_nonas.train)
(tree.rf.train.MSE <- mean((tree.rf.predict.train - Ht_nonas.train$Salary)^2))
(tree.rf.train.RMSE <- sqrt(tree.rf.train.MSE))
#   116

# OOB error (RMSE)
(tree.rf.OOB.MSE <- mean((tree.rf$predicted - Ht_nonas.train$Salary)^2))
(tree.rf.OOB.RMSE <- sqrt(tree.rf.OOB.MSE))
#   286

# Test error (RMSE)
tree.rf.predict.test <- predict(tree.rf, Ht_nonas.test)
(tree.rf.test.MSE <- mean((tree.rf.predict.test - Ht_nonas.test$Salary)^2))
(tree.rf.test.RMSE <- sqrt(tree.rf.test.MSE))
#   314

# Variable Importance 
importance(tree.rf)
sort(importance(tree.rf)[,1], decreasing = T)
(varImpPlot(tree.rf))

# Optimizing the ntree parameter
plot(tree.rf)
  # We see that the OOB error stabilizes after 500 trees 
  # (you can put in ntree=1000 above and see this), so no need to use more than 500.


##############   Boosting   ##########################

#   We use the gbm() function from the gbm library 
library(gbm)

# Perform boosting on the training data set, treating this as a regression problem. 
set.seed (1)
(tree.boost <- gbm(Salary ~ ., Ht_nonas.train, distribution="gaussian", n.trees=1000, interaction.depth = 4, shrinkage = 0.001, verbose = F))
# distribution = "gaussian" ... refers to a regression problem. 
# n.trees	                  ... Integer specifying the total number of trees to fit (number of iterations). Default is 100.
# interaction.depth         ... refers to the maximum depth of variable interactions. 
#                               A value of 1 implies an additive model, a value of 2 implies a model with up to 
#                               2-way interactions, etc. Default is 1.
# shrinkag                  ... a shrinkage parameter, also known as the learning rate or step-size reduction; 
#                               0.001 to 0.1 usually work, but a smaller learning rate typically requires more trees. Default is 0.1.

# Training error (RMSE)
tree.boost.predict.train <- predict(tree.boost, Ht_nonas.train)
(tree.boost.train.MSE <- mean((tree.boost.predict.train - Ht_nonas.train$Salary)^2))
(tree.boost.train.RMSE <- sqrt(tree.boost.train.MSE))
#   0.002
#   Our boosted ensemble seems to overfit! - Lets check the test error.

# Test error (RMSE)
tree.boost.predict.test <- predict(tree.boost, Ht_nonas.test)
(tree.boost.test.MSE <- mean((tree.boost.predict.test - Ht_nonas.test$Salary)^2))
(tree.boost.test.RMSE <- sqrt(tree.boost.test.MSE))
#   373 
#   Definitely overfitting...

# Variable importance
summary(tree.boost)
# outputs the variable importance as a table and a plot
#   rel.inf ... relative variable importance ("permutation importance"), same as %IncMSE used for random forests.
#             It measures the importance of a predictor by the average increase in prediction error when the values  
#             of a given predictor are shuffled (permuted). The values are normalized so that they add up to 100%.
#   If you dont see a variable in your plot, expand the plot window.
# We have different variables on top here...



##############   Evaluation Sum-Up  ##############   

# Now compare the different errors:
Tree          <- c(tree.all.train.RMSE, tree.all.test.RMSE)
prunedTree    <- c(tree.pruned.train.RMSE, tree.pruned.test.RMSE)
baggedTrees   <- c(tree.bag.train.RMSE, tree.bag.OOB.RMSE)
randomForest  <- c(tree.rf.train.RMSE, tree.rf.OOB.RMSE)
boostedTrees  <- c(tree.boost.train.RMSE, tree.boost.test.RMSE)

error_matrix <- data.frame(Tree, prunedTree, baggedTrees, randomForest, boostedTrees)
row.names(error_matrix) <- c("train", "test/OOB")
error_matrix
