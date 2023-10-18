# Load advertising dataset
adv <-
  read.csv("~/Documents/github/Data-Science/example_advertising_dataset/Advertising.csv")

View(adv)

# Check details of advertising dataset
dim(adv)
summary(adv)

# Check linear relationship between TV and sales
plot(adv$TV, adv$sales)

# Simple linear regression between sales and TV
lm.fit = lm(adv$sales~adv$TV)

# or attaching the dataset
attach(adv)
lm.fit = lm(sales~TV)

summary(lm.fit)

# Confidence interval of the coefficients
confint(lm.fit)
