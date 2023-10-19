#### Missing Values ####

# Packages to be installed
install.packages("nycflights13")
remove.packages("rlang")
install.packages("rlang")
install.packages("tidyverse", dependencies = TRUE)
install.packages("ggplot2")

# Packages activation
library(nycflights13)
library(tidyverse)
library(dplyr)
library(ggplot2)


# See the data "flights"
flights
count(flights)

# Filtering the data
filter(flights, month == 1)
filter(flights, month == 1 & day == 1)

# Identifying missing values: deleting the data row
filter(flights, !is.na(dep_time))
count(filter(flights, !is.na(dep_time)))

# Remembering the data set
my_flights <- filter(flights, !is.na(dep_time))

# Identifying and replacing missing values
flights_with_replaced_dep_time <- flights
flights_with_replaced_dep_time$dep_time <- 
  ifelse(is.na(flights$dep_time),
  flights$sched_dep_time, flights$dep_time)

# An example searching N18120
filter(flights, tailnum == "N18120") # Here there are NAs in dep_time
filter(flights_with_replaced_dep_time, tailnum == "N18120") # There are not NAs in dep_time

# NAs replacament with 1200
replacement <- 1200
flights_with_replaced_dep_time$dep_time <- 
  ifelse(is.na(flights$dep_time),
  replacement, flights$dep_time)

# Nas replacement by the mean
replacement <- as.integer(mean(flights$dep_time, na.rm = TRUE))
flights_with_replaced_dep_time$dep_time <- 
  ifelse(is.na(flights$dep_time),
  replacement, flights$dep_time)


#### Outliers ####

# Identifying and eliminating outliers
ggplot(flights) + geom_point(mapping = aes(x = flight, y = dep_delay))

arrange(flights, dep_delay) # Flights which really depart earlier than scheduled

minus_delay <- filter(flights, dep_delay <= 0) # Catch them

# Analysing the distribution of the negative departure delay
boxplot(minus_delay$dep_delay)

my_flights <- filter(flights, dep_delay > -29)

new_minus_delay <- filter(my_flights, dep_delay <= 0)

boxplot(new_minus_delay$dep_delay)

ggplot(my_flights) + geom_point(mapping = aes(x = flight, y = dep_delay))