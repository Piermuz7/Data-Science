#### Lab 4: Classification: kNN ####
#
# TASK:
# Explore the dataset Default from the ISLR library.
# The dataset holds historic records on credit card customers who have or have not defaulted on their credit card payments.
# Explore the dataset with the goal in mind that you want to train a classification model that predicts the value of the variable default.
#

install.packages("ISLR")
install.packages("Hmisc")
install.packages("ggplot2")
install.packages("gridExtra")

library(ISLR)
library(Hmisc)
library(ggplot2)
library(gridExtra)
library(class)

#### Data exploration of the "Default" data set  ####

View(Default)
help("Default")
summary(Default)
333 / dim(Default)[1] # The data set is highly unbalanced: only about 3.33% of all customers defaulted
sum(is.na(Default)) # No outliers

plot(Default)
# We see that defaulting customers tend to have a higher balance, but average income.
# Students tend to have a lower income, but thats not directly interesting for the target.
# Surprisingly, the amount of credit card balance seems to not depend on income.
# We can't read anything from student vs default, since there are only 4 possible combinations of values, and no pattern can be seen from this.

hist.data.frame(Default)
# We confirm that the dataset is highly unbalanced in the target.
# It is also highly unbalanced in the student variable: the large majority is non-students.
# Over 600 customers (out of 10000 - thats around 6%) have a zero or close-to-zero credit card balance. The majority has a positive balance, with few people over 2000$.
# Income is a bimodal distribution with peaks at 20k and 40k per year.
# There are no obvious outliers visible.

# How can boxplots help us to visualize the influence of input variables on the target?
boxplot(Default$balance, xlab = "balance")
boxplot(Default$income, xlab = "income")
# Not like that, since the target is nowhere.
boxplot(balance ~ default, data = Default)
boxplot(income ~ default, data = Default)

# Nicer way to plot it using ggplot2()
bp1 <- ggplot(Default, aes(y = balance, fill = default)) + geom_boxplot()
bp2 <- ggplot(Default, aes(y = income, fill = default)) + geom_boxplot()
grid.arrange(bp1, bp2, nrow = 1)

# We see that the income barely has any influence on default, while balance does:
# Not surprisingly, people who default have a higher credit card balance on average.
# That is consistent with what we observed with the plot() function.

# To double-check, we can display the class labels in the numerical predictor space, and see the above conclusion confirmed
ggplot(Default, aes(x = balance, y = income)) + geom_point(aes(col = default))

# How can we check the influence of the categorical variable student in default? Best use table():
table(Default$default)
table(Default$student)
table(Default$student, Default$default)
# 127/2944 = 4.3% of students default, and 206/7056 = 2.9 % of non-students default on their credit card.


#### Standardizing the predictors ####

# In standardizing the data, we exclude columns 1 and 2, because they are categorical.
Default.stand.pred <- scale(Default[,-c(1, 2)])
View(Default.stand.pred) # it worked
str(Default.stand.pred) # not a data frame any more, but a matrix


#### Split the data set in training and test set (VSA, validation set approach) ####
set.seed(1)
(indices <- sort(sample(1:dim(Default)[1], 2000))) # create 2000 randomly sampled indices -  we use a 80% - 20% split
(test.data.pred <- Default.stand.pred[indices, ]) # select the corresponding observations for the test set
(training.data.pred <- Default.stand.pred[-indices, ]) # select the remaining observations for the training set
(test.data.class <- Default$default[indices]) # store the class labels for the test set in a separate vector
(training.data.class <- Default$default[-indices]) # store the class labels for the training set in a separate vector


#### K-Nearest Neighbors ####

# Fit the k-NN model with k=1

# We set a random seed before we apply knn() because if several observations are tied as nearest neighbors,
# then R will randomly break the tie. Therefore, a seed must be set in order to ensure reproducibility of results.
set.seed (1)
(knn.pred.1 <- knn(training.data.pred, test.data.pred, training.data.class, k = 1))

# Estimate the test error rate
mean(knn.pred.1 != test.data.class)
# The error rate on the 2000 test observations is 4.65%. At first glance, this may appear to be very good.
# However, since only 3.33% of all customers defaulted, we could get the error rate down to 3.33 % by always
# predicting NO, regardless of the values of the predictors! (This is called the "trivial classifier".)

# Confusion Matrix
table(knn.pred.1, test.data.class)

# Fit the k-NN model with different parameters k
knn.pred.3 <- knn(training.data.pred, test.data.pred, training.data.class, k = 3)
knn.pred.10 <- knn(training.data.pred, test.data.pred, training.data.class, k = 10)
knn.pred.20 <- knn(training.data.pred, test.data.pred, training.data.class, k = 20)
knn.pred.30 <- knn(training.data.pred, test.data.pred, training.data.class, k = 30)
knn.pred.40 <- knn(training.data.pred, test.data.pred, training.data.class, k = 40)
knn.pred.50 <- knn(training.data.pred, test.data.pred, training.data.class, k = 50)
knn.pred.100 <- knn(training.data.pred, test.data.pred, training.data.class, k = 100)
knn.pred.200 <- knn(training.data.pred, test.data.pred, training.data.class, k = 100)

# Calculate the corrsponding test error rates
mean(knn.pred.3 != test.data.class)
mean(knn.pred.10 != test.data.class)
mean(knn.pred.20 != test.data.class)
mean(knn.pred.30 != test.data.class)
mean(knn.pred.40 != test.data.class)
mean(knn.pred.50 != test.data.class)
mean(knn.pred.100 != test.data.class)
mean(knn.pred.200 != test.data.class)

# k=20 is the sweet spot between overfitting and underfitting.
# The best error rate we achieve is 2.85 %, which is ca. 0.5% better than the trivial classifier always predicting NO, whose error rate is 3.33 %.
# These results are not excellent, but better than nothing on a very difficult unbalanced data set.

# The overall success rate of predicting if a customer will default or not is 97%
conf.matrix.20 <- table(knn.pred.20, test.data.class) # Confusion Matrix
(conf.matrix.20[1, 1] + conf.matrix.20[2, 2]) / 2000

# The success rate in catching the bad guys is 31%
conf.matrix.20[2, 2] / (conf.matrix.20[2, 2] + conf.matrix.20[1, 2]) # TPR (True Positive Rate)