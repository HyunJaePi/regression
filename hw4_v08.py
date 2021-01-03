# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:25:04 2020

hw4 code 1 -- 
This program tests multiple combination of regularization and outlier detection models
and find the optimal values of hyper-parameters

@author: HyunJae Pi
"""

# import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

import warnings
warnings.filterwarnings('ignore')

# calculate RMSE using K-fold cross validation
def rmse_cross_validation(model, x_train, y_train, folds):
    mse = cross_val_score(model, x_train, y_train, scoring='neg_mean_squared_error', cv=folds)
    return np.sqrt(np.abs(mse))

# remove indices of outliers using Isolation Forest
def indices_inliers_by_isolation_forest(data, contamination_factor):
    iso = IsolationForest(contamination=contamination_factor)
    indices_outliers = iso.fit_predict(data)
    # select all rows that are not outliers
    mask = indices_outliers != -1
    return mask

# remove indices of outliers using SVM
def indices_inliers_by_svm(data, nu):
    ocs = OneClassSVM(nu=nu)
    indices_outliers = ocs.fit_predict(data)
    mask = indices_outliers != -1 # select all rows that are not outliers
    return mask

# get filenames in the HW4data folder
def filenames():
    folder1 = ['a','b']
    folder2 = ['1','2','3','4','5']
    str0 = './HW4data/'

    filenames=[]
    for f1 in folder1:
        if f1 == 'a':
            str_tmp =""
        else:
            str_tmp ="X"
            
        for f2 in folder2:
                filenames.append(str0 + f1 + '/' + f2 + '/' + 'Train'+str_tmp+'_'+f2+f1+'.csv')
            
    return filenames


# base model: linear regression without outlier detection and regularization
def base_model(filename, folds):
    data = pd.read_csv(filename).to_numpy()
    X = data[:, 0:-1]
    y = data[:, -1]
    return np.average(rmse_cross_validation(LinearRegression(), X, y, folds))

# find the combination of optimal hyper-parameters
def find_hyperparams_for_min_rmse(filename, regularization_method, regularization_factor_range, outlier_detection_method, outlier_detection_factor_range, folds):
    data = pd.read_csv(filename).to_numpy()
    X = data[:, 0:-1]
    y = data[:, -1]

    # intialization
    ave_rmse = np.empty((len(outlier_detection_factor_range),len(regularization_factor_range)))
    #const_outlier_detection = np.empty((len(outlier_detection_factor_range),len(regularization_factor_range)))
    #const_regularization = np.empty((len(outlier_detection_factor_range),len(regularization_factor_range)))
    
    # outlier detection factor
    i=0
    for odf in outlier_detection_factor_range:
        if outlier_detection_method == 'IF': # isolation forest
            mask = indices_inliers_by_isolation_forest(X, odf)
        elif outlier_detection_method == 'SVM': # SVM
            mask = indices_inliers_by_svm(X, odf)
        else:
            warning("only two options: IF or SVM")
        X_, y_ = X[mask, :], y[mask]
    
        # regularization factor
        j=0
        for rf in regularization_factor_range:
            if regularization_method == 'L1':
                rmse = rmse_cross_validation(Lasso(alpha = rf, max_iter=1000, tol=0.0001), X_, y_, folds)
            elif regularization_method == 'L2':
                rmse = rmse_cross_validation(Ridge(alpha = rf, max_iter=1000, tol=0.0001), X_, y_, folds)
            else:
                warning("only two options: L1 or L2")
            ave_rmse[i][j] =np.average(rmse)
            j=j+1
        i=i+1

    ij = np.where(ave_rmse == np.min(ave_rmse))
    return (outlier_detection_factor_range[ij[0]][0], regularization_factor_range[ij[1]][0], ave_rmse[ij[0][0],ij[1][0]])


# k-fold cross validation
folds = KFold(n_splits = 5, shuffle = True, random_state = 1001)

# 1. testing a base model    
rmse_base=[]
for f in filenames():
    rmse_base.append(base_model(f, folds))    
print("\nBase model\n")
print(rmse_base)
 
   
# 2. L1 and Isolation Forest
contamination = np.arange(0, 0.5, 0.01) # contamination range = [0, 0.5]
alpha = np.arange(0, 0.5, 0.01) # regularization factor
RMSEs=[]
for f in filenames():
    rmse = find_hyperparams_for_min_rmse(f, 'L1', alpha, 'IF', contamination, folds)
    RMSEs.append(rmse)    
print("\nL1 & Isolation Forest\n")    
print(RMSEs)   


# 3. L2 and Isolation Forest  ------------ check alpha_range
contamination = np.arange(0, 0.5, 0.01) # contamination range = [0, 0.5]
alpha = np.arange(0, 0.5, 0.01) # regularization factor
RMSEs=[]
for f in filenames():
    rmse = find_hyperparams_for_min_rmse(f, 'L2', alpha, 'IF', contamination, folds)
    RMSEs.append(rmse)    
print("\nL2 & Isolation Forest\n")       
print(RMSEs)   


# 4. L1 and SVM
nu = np.arange(0.01, .5, .01) # SVM factor
alpha = np.arange(0, 0.5, 0.01) # regularization factor
RMSEs=[]
for f in filenames():
    rmse = find_hyperparams_for_min_rmse(f, 'L1', alpha, 'SVM', nu, folds)
    RMSEs.append(rmse)
print("\nL1 & SVM\n")    
print(RMSEs) 


# 5. L2 and SVM
nu = np.arange(0.01, .5, .01) # SVM factor
alpha = np.arange(0, 0.5, 0.01) # regularization factor
RMSEs=[]
for f in filenames():
    rmse = find_hyperparams_for_min_rmse(f, 'L2', alpha, 'SVM', nu, folds)
    RMSEs.append(rmse)
print("\nL2 & SVM\n")    
print(RMSEs)   