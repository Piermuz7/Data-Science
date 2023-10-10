#### Variable Transformation ####

titanic <- data.frame(Titanic)
titanic
my_titanic <- titanic

# Transforming the categorial (nominal) variable "Survived"
f <- factor(titanic$Survived)
levels(f) # possible values of f
typeof(f)
as.integer(f)
my_titanic$Survived <- as.integer(f)

# "Survived" is not ordered. Also "Sex" is not ordered. 
# Therefore we should not transform it into a factor. But we can create a unique (boolean) column for each value.
install.packages("fastDummies")
library(fastDummies)

# dummy_cols selects the column "Sex", remove it, and add for each value a new (0/1) columns, i.e. two colums "Sex_Male" and "Sex_Female".
my_titanic <- dummy_cols(my_titanic, select_columns="Sex", remove_selected_columns = TRUE)

my_titanic

# Transforming the ordinal variable "Age"

# Age is qualitative but ordered.
# This time we would like to influence how the different values are translated into number.
# An Adult is older than a child. Therefore Child = 1, Adult = 2
ordered(my_titanic$Age, levels= c("Child", "Adult"))

# Converting it into integers
as.integer(ordered(my_titanic$Age, levels= c("Child", "Adult")))

my_titanic$Age <- as.integer(ordered(my_titanic$Age, levels=c("Child", "Adult")))
my_titanic

# Transforming the (partly) ordinal variable "Class"

# While the three values ("1st", "2nd", "3rd") are obviously ordered, is the value "Crew" a little bit separated from that.
# Therefore we need to have a Boolean column for "Crew" but an ordered column for the other three values.

# Creating a new different column for "Crew"
ifelse(my_titanic$Class =="Crew",1,0)
my_titanic$Class_Crew <- ifelse(my_titanic$Class =="Crew",1,0)

# Ordering "Class"
ordered(my_titanic$Class, levels= c("Crew", "3rd", "2nd","1st"))

# Converting it into integers
as.integer(ordered(my_titanic$Class, levels= c("Crew", "3rd", "2nd","1st")))

# Subtracting 1 to have from 0 to 3 instead from 1 to 4
my_titanic$Class <- as.integer(ordered(my_titanic$Class, levels= c("Crew", "3rd", "2nd","1st"))) - 1

my_titanic
