#### Data Transformation ####

# Packages to be installed
install.packages("nycflights13")
remove.packages("rlang")
install.packages("rlang")
install.packages("tidyverse", dependencies = TRUE)

# Packages activation
library(nycflights13)
library(dplyr)
library(tidyverse)

# Deleting columns: example "dep_delay" column
my_flight <- subset(flights,select=-dep_delay)
my_flight

# Deleting several columns, example "dep_delay" and "flight"
my_flights <- subset(flights,select=-c(dep_delay,flight))
my_flights

# Transforming departure delay
my_flights <- filter(flights, ! is.na(dep_time)) # Remove all rows with missing values
my_flights <- filter(my_flights, dep_delay > -29) # and remove rows with extreme negative delay

# Analysing the distribution
hist(my_flights$dep_delay) # Looks imbalanced, looks like a logarithmic distribution
# And then converting it to a more uniform distribution ..
hist(log(my_flights$dep_delay)) # Now it looks better, but we produce NA (for the negative delays; log is not defined for negative inputs)

# In order to remove negative values, without deleting them, we simply shift the delays by the most negative value (i.e. the minimum). Then all values are positive.
minimum <- min(my_flights$dep_delay,na.rm = TRUE)
hist(log(my_flights$dep_delay - minimum))

my_flights$dep_delay <- log(my_flights$dep_delay - minimum)

#### Data Normalization ####
my_flights <- filter(flights, !is.na(dep_time)) # Remove all the rows with missing values

# Apply the min-max normalization to "dep_time" (assuming a range of 0000-2400)
my_flights$dep_time <- my_flights$dep_time / 2400
my_flights

# However the coding of time in integer is not continuous. 
# E.g. 1178 would never exists.
# We need a (self-defined) conversion function "time_conversion", which translates that into continuous numbers

time_conversion <- function(x) {
  h <- trunc(x/100,0)
  m <- x-(h*100)
  r <- m+(h*60)
  return(r) 
}

my_flights$dep_time <- time_conversion(my_flights$dep_time) / (24*60)
my_flights

