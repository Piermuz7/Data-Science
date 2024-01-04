# Assignment1: Prediction Task

This file shows how the project is structured with regard to assignment 1 for the prediction task.

## FHNW_DS_Assignment1_Group1.pdf
This is the pdf describing our report for this task.

## Assignment1.Rmd
The entire project was realised in an R markdown file, in order to have the possibility of visualising graphs at any time, unlike a simple R file. This file contains all the steps, from data exploration, preprocessing to algorithm evaluation and saving the best model.

## best_model_xgb.rds
In our case, we got the XGBoost as the best model, saved in an rds file.

## preprocessing_old_version.R
This R file contains all the steps of the initial preprocessing version.

## preprocessing_final_version.R
This R file contains all the steps of the final preprocessing version.

## main.R
This file is equivalent to the Assignment.Rmd file but modularized. In fact it contains lines of code to set the LC dataset and then calls the preprocessing_final_version.R file to apply the preprocessing. Once the data has been preprocessed, the learning_lgorithms file is called up to evaluate our models on that dataset. Then the civ of the preprocessed dataset, the training and the testing set are saved. Finally, the best model is saved in rds format.

## learning_algorithms.R
This R file contains the same models of Assignment1 for the learning phase.

## reality_check.R
This file is used exclusively for the reality check. To use it, simply change the string 'secret_data.csv' to your own dataset.
Once loaded, the preprocessing_final_version.R script will be called up to preprocess the data, the previously saved best rds model will be loaded, and finally predictions will be made on this dataset.