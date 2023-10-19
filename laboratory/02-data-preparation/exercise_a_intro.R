# Using the build in data set "mtcars"

# Packages to be installed and activated
install.packages("ggplot2")
library(ggplot2)
install.packages("corrplot")
library(corrplot)

mtcars # Print dataset
str(mtcars) # Display structure of the dataset "mtcars"

# Compute the mean, median, and mode of column "wt"
mean(mtcars$wt)
median(mtcars$wt)
# mode(mtcars$wt) # Not correct!
y <- table(mtcars$wt)
names(y)[which(y==max(y))] # Correct statistical mode

# To quickly summarize the values in the data set "mtcars"
summary(mtcars)

# Plot the column "wt"
ggplot(mtcars) + geom_point(mapping = aes(x = 1:length(wt), y = wt))

# Plot the column "wt" against "drat"
ggplot(mtcars) + geom_point(mapping = aes(x = drat, y = wt))

# Boxplot of column "wt"
boxplot(mtcars$wt)

# Histogram of column "wt"
hist(mtcars$wt)

# Q-Q plot of column "wt"
qqnorm(mtcars$wt)

# Q-Q plot with assumed line of column "wt"
qqline(mtcars$wt)

# Correlation of columns of data set "mtcars"
#cor(mtcars)
round(cor(mtcars),2)

# Plot of the correlation of columns of data set "mtcars"
corrplot(cor(mtcars), type = "upper", order = "hclust", tl.col = "black", tl.srt = 45)