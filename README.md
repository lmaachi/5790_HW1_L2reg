# Overview
In this assignment, I implement an L2 regularized linear regression (i.e., ridge regression) with λ ranging from 0 to 150, and λ ranging from 1 to 150 on datasets 100-100, 50(1000)-100, 100(1000)-100. My code includes functions for data processing, computing weight calculation and mean-squared error, and 10-fold cross-validation analysis. 
# Installation
Please ensure the following Python libraries are also installed:
-	pandas
-	numpy
-	matplotlib
# Usage
## Functions
`def process_defs(df_train, df_test=None)`

This function converts the dataframes into matrices (i.e., arrays) for calculations. It also adds a column of ones (‘1’) to the input matrix if not present, for the purpose of evening out the sizes and ensuring matrix calculation can be done.

`def compute_weight(x_mtx,trn, y_mtx_trn, start, end)`

This function computes the weight based on the formula w = (XTX + λI)−1XTy and iterates through a range of lambda values (start to end). 

`def compute_mse(x_trn, w, y_trn)`

This function computes the mean-squared error (MSE) between the predicted and actual values.

`def ten_fold_cv(x_trn, y_trn, k_folds=10, lambda_max=150)`

This function performs 10-fold CV (cross-validation) analysis and calculates the average mean-squared error for each lambda value.
## Main Loop
A for-loop iterates through different datasets, calculating the weights, mean-squared-errors, and performing a cross-validation analysis. It also generates plots for training and testing mean-squared errors for lambda values. I also wrote an additional loop for the calculations of λ1-150 for datasets 100-100,  50(1000)-100, and 100(1000)-100. 
# Results
For each dataset, the code outputs the lambda value that minimizes the testing mean-squared error and the corresponding minimum mean-squared error from the training set. It also displays plots showing the training and testing mean-squared errors across different lambda values.
