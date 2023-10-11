#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 1 15:49:58 2023

@author: lulumaachi
"""

import pandas as pd #For importing .csv files.#
import numpy as np #For arithmetic calculations.#
import matplotlib.pyplot as plt #For plotting data/outputting graphs.#

df1_test = pd.read_csv(r'/Users/lulumaachi/Downloads/HW1_dataset/test-100-10.csv')
df1_train = pd.read_csv(r'/Users/lulumaachi/Downloads/HW1_dataset/train-100-10.csv')
df1_train = df1_train.iloc[:,:-2]

df2_test = pd.read_csv(r'/Users/lulumaachi/Downloads/HW1_dataset/test-100-100.csv')
df2_train = pd.read_csv(r'/Users/lulumaachi/Downloads/HW1_dataset/train-100-100.csv')

df3og_test = pd.read_csv(r'/Users/lulumaachi/Downloads/HW1_dataset/test-1000-100.csv')
df3og_train = pd.read_csv(r'/Users/lulumaachi/Downloads/HW1_dataset/train-1000-100.csv')

#Taking the first 50, 100, and 150 instances of from [train-1000-100.csv] to create three new training datasets.#
df3og_train[0:50].to_csv('train-50(1000)-100.csv', index=False)
df3og_train[0:100].to_csv('train-100(1000)-100.csv', index=False)
df3og_train[0:150].to_csv('train-150(1000)-100.csv', index=False)

df3new_50 = pd.read_csv('train-50(1000)-100.csv')
df3new_100 = pd.read_csv('train-100(1000)-100.csv')
df3new_150 = pd.read_csv('train-150(1000)-100.csv')

#Function for processing dataframes into matrices (i.e. arrays) for calculations.#
def process_dfs(df_train, df_test=None):
    def df_mtx(df):
        if '1' not in df.columns:
            df.insert(0, '1', 1)            
        x_trn = df.iloc[:, :-1]
        x_mtx_trn = x_trn.values
        y_trn = df.iloc[:, -1:]
        y_mtx_trn = y_trn.values
        return x_mtx_trn, y_mtx_trn

    x_train, y_train = df_mtx(df_train)
    x_test, y_test = None, None
    
    if df_test is not None:
        x_test, y_test = df_mtx(df_test)
    
    return x_train, y_train, x_test, y_test

#Storing the converted matrices into these lists for later use.#
data_list_test = [df1_test, df2_test, df3og_test, df3og_test, df3og_test, df3og_test]
data_list_train = [df1_train, df2_train, df3og_train, df3new_50, df3new_100, df3new_150]
    
#Function for computing weight calculation based on the formula w = (XTX + λI)−1XTy.#
def compute_weight(x_mtx_trn, y_mtx_trn, start, end):
    lambda_vals = np.arange(start, end + 1)
    w_lambda = []

    x_trn_trans = np.transpose(x_mtx_trn)
    xtx_trn = np.dot(x_trn_trans, x_mtx_trn)
    iden_mtx = np.identity(len(xtx_trn))

    for lambda_value in lambda_vals:
        lambda_iden = lambda_value * iden_mtx
        xtx_iden_sum = xtx_trn + lambda_iden
        xtx_iden_sum_inv = np.linalg.inv(xtx_iden_sum)
        xty_trn = np.dot(x_trn_trans, y_mtx_trn)
        w = np.dot(xtx_iden_sum_inv, xty_trn)
        w_lambda.append(w.flatten())

    w_trn_data = np.transpose(np.array(w_lambda))
    return w_trn_data
    
#Function for calculating the mean-squared-error.#
def compute_mse(x_trn, w, y_trn):
    y_pred = np.dot(x_trn, w)
    sum_error = 0.0

    for i in range(len(y_trn)):
        y_pred_error = y_pred[i] - y_trn[i]
        sum_error += (y_pred_error ** 2)
    mean_sq_error = sum_error / float(len(y_trn))
    return mean_sq_error  

#Function for completing 10-fold CV analysis.#
def ten_fold_cv(x_trn, y_trn, k_folds=10, lambda_max=150):
    fold_size = int(len(y_trn) / k_folds)
    mse_sum = 0
    
    for i in range(k_folds):
        x_test_fold = x_trn[i * fold_size:(i + 1) * fold_size]
        y_test_fold = y_trn[i * fold_size:(i + 1) * fold_size]

        x_train_fold = np.concatenate((x_trn[:i * fold_size], x_trn[(i + 1) * fold_size:]), axis=0)
        y_train_fold = np.concatenate((y_trn[:i * fold_size], y_trn[(i + 1) * fold_size:]), axis=0)

        weights = compute_weight(x_train_fold, y_train_fold, start=0, end=150)
        
        mse_sum += compute_mse(x_test_fold, weights, y_test_fold)

    mse_avg = mse_sum / k_folds
    return mse_avg

#For-loop to iterate through each element in 'data_list_test' and 'data_list_train' and calculate λ0-150, MSEs, and graphs.#
for i in range(6):
    df_test = data_list_test[i]
    df_train = data_list_train[i]
    x_train, y_train, x_test, y_test = process_dfs(df_train, df_test)
    w = compute_weight(x_train, y_train, start=0, end=150)
    mse_train = compute_mse(x_train, w, y_train)
    mse_test = compute_mse(x_test, w, y_test)
    lam_min = mse_test.argmin()
    mse_min = mse_test[lam_min]
    cv_analysis = ten_fold_cv(x_train, y_train)
    optimal_lam = np.argmin(cv_analysis) + lam_min
    mse_opt = mse_train[optimal_lam]
    print("The λ value", lam_min, "gives the least MSE value for", mse_min, "in dataset", i+1)
    print("The best lambda value for dataset", i+1, "is:", optimal_lam, "and the accompanying MSE value is",mse_opt, "\n")  
    mse_train_plot = plt.plot(mse_train, label='Training MSE', color='red')
    mse_test_plot = plt.plot(mse_test, label='Testing MSE', color='orange')
    plt.title(f"Train and Test : Dataset {i + 1}")
    plt.xlabel('Lambda Values')
    plt.ylabel('Mean-Squared-Error')
    plt.legend()
    plt.show()
 
#For-loop to start from index 3 and iterate through 'data_list_test' and 'data_list_train' to calculate λ1-150, MSEs, and graphs.#  
for i in range(3, 6):
    df_test = data_list_test[i]
    df_train = data_list_train[i]
    x_train, y_train, x_test, y_test = process_dfs(df_train, df_test)
    w = compute_weight(x_train, y_train, start=1, end=150)
    mse_train = compute_mse(x_train, w, y_train)
    mse_test = compute_mse(x_test, w, y_test)
    lam_min = mse_test.argmin()
    mse_min = mse_test[lam_min]
    cv_analysis = ten_fold_cv(x_train, y_train)
    optimal_lam = np.argmin(cv_analysis) + lam_min
    mse_opt = mse_train[optimal_lam]
    mse_train_plot = plt.plot(mse_train, label='Training MSE', color='red')
    mse_test_plot = plt.plot(mse_test, label='Testing MSE', color='orange')
    plt.title(f"Train and Test w/λ1-150 : Dataset {i + 1}")
    plt.xlabel('Lambda Values')
    plt.ylabel('Mean-Squared-Error')
    plt.legend()
    plt.show()